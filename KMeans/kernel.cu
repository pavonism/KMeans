#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sys/types.h> 
#include <sys/stat.h>
#include <fcntl.h>
#include <cassert>
#include <limits>
#include <chrono>
#include <random>

#define MAX_CHAR_PER_LINE 128
#define THREADS_PER_BLOCK 1024
#define MAX_ITERATIONS 500
#define DEBUG 1

cudaError_t kMeansThrust(float* points, int clustersCount, int pointsCount, int dimNum, float threshold, float** clusters, int* iterations, int** memberships);
cudaError_t kMeansReduce(float* points, int clustersCount, int pointsCount, int dimNum, float threshold, float** clusters, int* iterations, int** memberships);
float* readFile(
	char* filename,
	int* pointsCount,
	int* dimNum);
int writeFile(char* filename,
	int numClusters,
	int numObjs,
	int numCoords,
	float* clusters,
	int* membership);
void coalesceData(float** points, int pointsCount, int dimNum);
void unCoalesceData(float** points, int pointsCount, int dimNum);
void getMinMax(float* points, int pointsCount, int dimNum, float** maxCoordinates, float** minCoordinates);
void generateClusters(float* minCoordinates, float* maxCoordinates, float** clusters, int clusterCount, int dimNum);
void printClusters(float* data, int numObjs, int numCoords);
void generateRandomPoints(char* path, int pointsCount, int dimNum);
void generateRandomPointsNormal(char* path, int pointsCount, int dimNum, int groups);
template <class T>
cudaError_t reduce(T* data, int n);
template <class T>
cudaError_t reduceChunks(T* values, int n, cudaStream_t* streams, int k);

__global__ void pointsDistance(float* points, float* clusters, int* membership, int* membershipDims, int* membershipChanged, int clustersCount, int pointsCount, int dimNum)
{
	int pointNum = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	float minDistance = FLT_MAX;
	int currentMembership = -1;

	if (pointNum >= pointsCount)
		return;

	for (size_t i = 0; i < clustersCount; i++)
	{
		float distance = 0;

		for (size_t j = 0; j < dimNum; j++)
		{
			float diff = clusters[j * clustersCount + i] - points[j * pointsCount + pointNum];
			distance += diff * diff;

		}

		if (distance < minDistance) {
			minDistance = distance;
			currentMembership = i;
		}
	}

	__syncthreads();
	membershipChanged[pointNum] = membership[pointNum] != currentMembership;

	if (membershipDims != NULL) {
		__syncthreads();
		for (size_t i = 0; i < dimNum; i++)
		{
			membershipDims[i * pointsCount + pointNum] = i * clustersCount + currentMembership;
		}
	}
	__syncthreads();
	membership[pointNum] = currentMembership;
}

__global__ void updateClusters(float* clusters, float* clusterSums, int* clusterMembersCount, int* clusterMembersCountKeys, int clusterCount, int dimNum, int usedClusters) {

	int threadId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (threadId >= usedClusters)
		return;

	int clusterId = clusterMembersCountKeys[threadId];
	int members = clusterMembersCount[threadId];

	for (size_t i = 0; i < dimNum; i++)
	{
		clusters[i * clusterCount + clusterId] = clusterSums[i * usedClusters + threadId] / members;
	}
}

template <unsigned int blockSize, class T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize, class T>
__global__ void reduceGlobal(T* g_idata, T* g_odata, unsigned int n) {
	extern __shared__ __align__(sizeof(T)) unsigned char memory[];
	T* sdata = reinterpret_cast<T*>(memory);
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize, T>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void pickCoordinate(float* points, float* clusters, int* membership, float* sums, int* membersCount, int pointsCount, int pointsCountPowerOfTwo, int clustersCount, int dimNum) {

	extern __shared__ float pointData[];
	int threadId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int member = -1;
	float value = 0;

	if (threadId > pointsCountPowerOfTwo)
		return;

	if (threadId < pointsCount) {
		member = membership[threadId];
	}

	for (size_t dimension = 0; dimension < dimNum; dimension++)
	{
		if (threadId < pointsCount) {
			value = points[dimension * pointsCount + threadId];
		}

		for (size_t cluster = 0; cluster < clustersCount; cluster++)
		{
			float currentValue = member == cluster ? value / membersCount[cluster] : 0;
			sums[((long long)cluster * dimNum + dimension) * pointsCount + threadId] = currentValue;
		}
	}
}

__global__ void pickMembership(int* membership, int* sums, int pointsCount, int pointsCountPowerOfTwo, int clustersCount) {

	extern __shared__ int data[];

	int threadId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (threadId > pointsCountPowerOfTwo)
		return;

	for (size_t k = 0; k < clustersCount; k++)
	{
		int member = threadId < pointsCount ? membership[threadId] : -1;
		member = member == k ? 1 : 0;
		sums[k * pointsCountPowerOfTwo + threadId] = member;
	}
}

template <class T>
__global__ void gather(T* data, int chunkSizes, int chunks) {

	if (threadIdx.x > chunks)
		return;

	data[threadIdx.x] = data[(long long)threadIdx.x * chunkSizes];;
}

int main(int argc, char** argv)
{
	// TODO: dwie wersje mogą się odpalać jedno po drugim
	if (argc < 5)
	{
		printf("Wrong number of arguments!\n");
		return EXIT_FAILURE;
	}

	char* fileName = argv[1];

	int clustersCount = atoi(argv[2]);
	int pointsCount = 0, dimNum;
	float threshold = atof(argv[3]);
	float* minCoordinates = NULL;
	float* maxCoordinates = NULL;
	float* clusters = NULL;
	int* memberships = NULL;
	cudaError_t cudaStatus;

	printf("Reading file...\n");
	auto cpuStart = std::chrono::high_resolution_clock::now();
	float* points = readFile((char*)fileName, &pointsCount, &dimNum);
	auto cpuEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = cpuEnd - cpuStart;
	printf("Elapsed %fs...\n", diff.count());

	coalesceData(&points, pointsCount, dimNum);
	getMinMax(points, pointsCount, dimNum, &maxCoordinates, &minCoordinates);
	// TODO: ziarno dla generatora na inpucie
	generateClusters(minCoordinates, maxCoordinates, &clusters, clustersCount, dimNum);
	printClusters(clusters, clustersCount, dimNum);
	int iterations;

	if (strcmp(argv[4], "r") == 0) {
		cudaStatus = kMeansReduce(points, clustersCount, pointsCount, dimNum, threshold, &clusters, &iterations, &memberships);
	}
	else {
		cudaStatus = kMeansThrust(points, clustersCount, pointsCount, dimNum, threshold, &clusters, &iterations, &memberships);
	}

	if (cudaStatus == cudaSuccess) {
		printf("Iterations: %d\n", iterations);
		unCoalesceData(&clusters, clustersCount, dimNum);
		writeFile(fileName, clustersCount, pointsCount, dimNum, clusters, memberships);
	}

	delete minCoordinates;
	delete maxCoordinates;
	delete clusters;
	delete memberships;
	return EXIT_SUCCESS;
}

cudaError_t kMeansThrust(float* points, int clustersCount, int pointsCount, int dimNum, float threshold, float** clusters, int* iterations, int** memberships)
{
	float* dev_points = NULL;
	float* dev_points_sums = NULL;
	float* dev_clusters = NULL;
	float* dev_clustersSums = NULL;
	int* dev_membershipDims = NULL;
	int* dev_membership = NULL;
	int* dev_currentMembership = NULL;
	int* dev_membershipChanged = NULL;
	int* dev_clusterSizes = NULL;
	int* dev_clusterSizesKeys = NULL;
	float* dev_delta = NULL;
	float delta = FLT_MAX;
	cudaError_t cudaStatus;

	int blocksCount = (int)ceil((float)pointsCount / THREADS_PER_BLOCK);
	int newClustersBlocksCount = (int)ceil((float)clustersCount / THREADS_PER_BLOCK);

	int nearestPowerOfTwo = 1;
	while (nearestPowerOfTwo < clustersCount)
		nearestPowerOfTwo *= 2;
	int iterationNumber = 0;

	printf("Preparing memory...\n");

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_points, pointsCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_points_sums, pointsCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusters, clustersCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_clustersSums, clustersCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membershipDims, dimNum * pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membership, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_currentMembership, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membershipChanged, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_delta, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusterSizes, clustersCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusterSizesKeys, clustersCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_points, points, pointsCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_clusters, *clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(dev_membership, -1, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Calculating on GPU...\n");
	cudaEventRecord(start, 0);
	do {
		printf(".");
		pointsDistance << <blocksCount, THREADS_PER_BLOCK >> > (dev_points, dev_clusters, dev_membership, dev_membershipDims, dev_membershipChanged, clustersCount, pointsCount, dimNum);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "pointsDistance launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_points_sums, dev_points, pointsCount * dimNum * sizeof(int), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointsDistance!\n", cudaStatus);
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_currentMembership, dev_membership, pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		thrust::sort(thrust::device, dev_currentMembership, dev_currentMembership + pointsCount);
		auto pair = thrust::reduce_by_key(thrust::device, dev_currentMembership, dev_currentMembership + pointsCount, thrust::make_constant_iterator(1), dev_clusterSizesKeys, dev_clusterSizes);
		int usedClusters = pair.first - dev_clusterSizesKeys;
		thrust::sort_by_key(thrust::device, dev_membershipDims, dev_membershipDims + pointsCount * dimNum, dev_points_sums);
		thrust::reduce_by_key(thrust::device, dev_membershipDims, dev_membershipDims + pointsCount * dimNum, dev_points_sums, thrust::make_discard_iterator(), dev_clustersSums);

		updateClusters << <newClustersBlocksCount, THREADS_PER_BLOCK >> > (dev_clusters, dev_clustersSums, dev_clusterSizes, dev_clusterSizesKeys, clustersCount, dimNum, usedClusters);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "pointsDistance launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		delta = (float)thrust::reduce(thrust::device, dev_membershipChanged, dev_membershipChanged + pointsCount);
		delta /= pointsCount;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointsDistance!\n", cudaStatus);
			goto Error;
		}

		iterationNumber++;
	} while (delta > threshold && iterationNumber < MAX_ITERATIONS);
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed %fs\n", time / 1000);

	*iterations = iterationNumber;

	cudaStatus = cudaMemcpy(*clusters, dev_clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	*memberships = new int[pointsCount];

	cudaStatus = cudaMemcpy(*memberships, dev_membership, pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_points);
	cudaFree(dev_points_sums);
	cudaFree(dev_clusters);
	cudaFree(dev_clustersSums);
	cudaFree(dev_membershipDims);
	cudaFree(dev_membership);
	cudaFree(dev_currentMembership);
	cudaFree(dev_membershipChanged);
	cudaFree(dev_clusterSizes);
	cudaFree(dev_clusterSizesKeys);
	cudaFree(dev_delta);

	return cudaStatus;
}

cudaError_t kMeansReduce(float* points, int clustersCount, int pointsCount, int dimNum, float threshold, float** clusters, int* iterations, int** memberships) {

	float* dev_points = NULL;
	float* dev_sums = NULL;
	float* dev_clusters = NULL;
	int* dev_membership = NULL;
	int* dev_membership_sums = NULL;
	int* dev_membershipChanged = NULL;
	int* dev_membership_result = NULL;
	cudaStream_t* streams = new cudaStream_t[clustersCount * dimNum];
	int* membersCount = new int[clustersCount];
	float delta = FLT_MAX;
	cudaError_t cudaStatus;

	for (size_t k = 0; k < clustersCount * dimNum; k++)
	{
		cudaStreamCreate(&streams[k]);
	}

	int blocksCount = (int)ceil((float)pointsCount / THREADS_PER_BLOCK);
	int pointsCountPowerOfTwo = 1;
	while (pointsCountPowerOfTwo < pointsCount) pointsCountPowerOfTwo *= 2;
	int reduceBlocksCount = (int)ceil((float)pointsCountPowerOfTwo / THREADS_PER_BLOCK);
	int iterationNumber = 0;

	printf("Preparing memory...\n");

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_points, pointsCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_sums, clustersCount * dimNum * pointsCountPowerOfTwo * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusters, clustersCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membership, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membership_result, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membership_sums, clustersCount * pointsCountPowerOfTwo * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membershipChanged, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_points, points, pointsCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_clusters, *clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(dev_membership, -1, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Calculating on GPU...\n");
	cudaEventRecord(start, 0);

	do
	{
		printf(".");
		pointsDistance << <blocksCount, THREADS_PER_BLOCK >> > (dev_points, dev_clusters, dev_membership, NULL, dev_membershipChanged, clustersCount, pointsCount, dimNum);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "pointsDistance launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointsDistance!\n", cudaStatus);
			goto Error;
		}

		delta = (float)thrust::reduce(thrust::device, dev_membershipChanged, dev_membershipChanged + pointsCount);
		delta /= pointsCount;

		if (delta < threshold || iterationNumber++ > MAX_ITERATIONS)
			break;
		//printf("Delta: %f", delta);

		pickMembership << <reduceBlocksCount, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int) >> > (
			dev_membership,
			dev_membership_sums,
			pointsCount,
			pointsCountPowerOfTwo,
			clustersCount
			);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "countMembers launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching countMembers!\n", cudaStatus);
			goto Error;
		}

		cudaStatus = reduceChunks(dev_membership_sums, pointsCountPowerOfTwo, streams, clustersCount);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduce launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}


		pickCoordinate << <reduceBlocksCount, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (
			dev_points,
			dev_clusters,
			dev_membership,
			dev_sums,
			dev_membership_sums,
			pointsCount,
			pointsCountPowerOfTwo,
			clustersCount,
			dimNum
			);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "countMembers launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching countMembers!\n", cudaStatus);
			goto Error;
		}

		cudaStatus = reduceChunks(dev_sums, pointsCountPowerOfTwo, streams, clustersCount * dimNum);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduce launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaMemcpy(dev_clusters, dev_sums, clustersCount * dimNum * sizeof(float), cudaMemcpyDeviceToDevice);

	} while (1);
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed %fs\n", time / 1000);

	*iterations = iterationNumber;

	cudaStatus = cudaMemcpy(*clusters, dev_clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	*memberships = new int[pointsCount];

	cudaStatus = cudaMemcpy(*memberships, dev_membership, pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_membership);
	cudaFree(dev_membership_sums);
	cudaFree(dev_membership_result);
	cudaFree(dev_clusters);
	cudaFree(dev_membershipChanged);

	for (size_t k = 0; k < clustersCount; k++)
	{
		cudaStreamDestroy(streams[k]);
	}
	delete streams;
	delete membersCount;

	return cudaStatus;
}

template <class T>
cudaError_t reduce(T* data, int n) {

	int currentThreads = n;
	cudaError_t cudaStatus;

	while (currentThreads > 1) {
		//printf("currentThreads: %d\n", currentThreads);
		int dimGrid = (int)ceil((float)currentThreads / THREADS_PER_BLOCK);
		int dimBlock = dimGrid == 1 ? currentThreads : THREADS_PER_BLOCK;
		int	smemSize = dimBlock * sizeof(T);

		switch (currentThreads / 2)
		{
		default:
			reduceGlobal<512> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 256:
			reduceGlobal<256> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 128:
			reduceGlobal<128> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 64:
			reduceGlobal< 64> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 32:
			reduceGlobal< 32> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 16:
			reduceGlobal< 16> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 8:
			reduceGlobal< 8> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 4:
			reduceGlobal< 4> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 2:
			reduceGlobal< 2> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		case 1:
			reduceGlobal< 1> << < dimGrid, dimBlock, smemSize >> > (data, data, currentThreads); break;
		}

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduceGlobal launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceGlobal!\n", cudaStatus);
			return cudaStatus;
		}

		currentThreads /= THREADS_PER_BLOCK;
	}

	return cudaSuccess;
}

template <class T>
cudaError_t reduceChunks(T* values, int n, cudaStream_t* streams, int k) {

	int currentThreads = n;
	cudaError_t cudaStatus;

	while (currentThreads > 1) {
		//printf("currentThreads: %d\n", currentThreads);
		int dimGrid = (int)ceil((float)currentThreads / THREADS_PER_BLOCK);
		int dimBlock = dimGrid == 1 ? currentThreads : THREADS_PER_BLOCK;
		int	smemSize = dimBlock * sizeof(T);

		for (size_t i = 0; i < k; i++)
		{
			T* data = values + i * n;

			switch (currentThreads / 2)
			{
			default:
				reduceGlobal<512> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 256:
				reduceGlobal<256> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 128:
				reduceGlobal<128> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 64:
				reduceGlobal< 64> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 32:
				reduceGlobal< 32> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 16:
				reduceGlobal< 16> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 8:
				reduceGlobal< 8> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 4:
				reduceGlobal< 4> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 2:
				reduceGlobal< 2> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			case 1:
				reduceGlobal< 1> << < dimGrid, dimBlock, smemSize, streams[i] >> > (data, data, currentThreads); break;
			}

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "reduceGlobal launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return cudaStatus;
			}
		}

		currentThreads /= THREADS_PER_BLOCK;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching countMembers!\n", cudaStatus);
		return cudaStatus;
	}

	int blocks = (int)ceil((float)k / THREADS_PER_BLOCK);
	gather << <blocks, THREADS_PER_BLOCK >> > (values, n, k);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "reduceGlobal launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching countMembers!\n", cudaStatus);
		return cudaStatus;
	}

	return cudaSuccess;
}

void coalesceData(float** points, int pointsCount, int dimNum) {

	float* coalesced = new float[pointsCount * dimNum];
	int64_t indx = 0;

	for (size_t i = 0; i < dimNum; i++)
	{
		for (size_t j = 0; j < pointsCount; j++)
		{
			coalesced[indx++] = (*points)[j * dimNum + i];
		}
	}

	free(*points);
	*points = coalesced;
}

void unCoalesceData(float** points, int pointsCount, int dimNum) {

	float* unCoalesced = new float[pointsCount * dimNum];

	for (size_t i = 0; i < pointsCount; i++)
	{
		for (size_t j = 0; j < dimNum; j++)
		{
			unCoalesced[i * dimNum + j] = (*points)[j * pointsCount + i];
		}
	}

	free(*points);
	*points = unCoalesced;
}

void generateClusters(float* minCoordinates, float* maxCoordinates, float** clusters, int clusterCount, int dimNum) {

	*clusters = new float[clusterCount * dimNum];


	for (size_t i = 0; i < dimNum; i++)
	{
		float diff = maxCoordinates[i] - minCoordinates[i];

		for (size_t j = 0; j < clusterCount; j++)
		{
			float random = (float)rand() / (float)RAND_MAX;
			(*clusters)[i * clusterCount + j] = random * diff + minCoordinates[i];
		}
	}

}

void getMinMax(float* points, int pointsCount, int dimNum, float** maxCoordinates, float** minCoordinates) {

	(*minCoordinates) = (float*)malloc(dimNum * sizeof(float));
	(*maxCoordinates) = (float*)malloc(dimNum * sizeof(float));
	memset(*minCoordinates, FLT_MAX, dimNum * sizeof(float));
	memset(*maxCoordinates, FLT_MIN, dimNum * sizeof(float));


	for (size_t i = 0; i < dimNum; i++)
	{
		for (size_t j = 0; j < pointsCount; j++)
		{
			auto coordinate = points[i * pointsCount + j];

			if ((*maxCoordinates)[i] < coordinate)
				(*maxCoordinates)[i] = coordinate;
			if ((*minCoordinates)[i] > coordinate)
				(*minCoordinates)[i] = coordinate;
		}
	}
}

float* readFile(
	char* filename,
	int* pointsCount,
	int* dimNum
)
{
	float* points;
	int     i, j, len;
	FILE* infile;
	char* line, * ret;
	int   lineLen;

	if ((infile = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error: no such file (%s)\n", filename);
		return NULL;
	}

	/* first find the number of objects */
	lineLen = MAX_CHAR_PER_LINE;
	line = (char*)malloc(lineLen);
	assert(line != NULL);

	(*pointsCount) = 0;
	while (fgets(line, lineLen, infile) != NULL) {
		/* check each line to find the max line length */
		while (strlen(line) == lineLen - 1) {
			/* this line read is not complete */
			len = (int)strlen(line);
			fseek(infile, -len, SEEK_CUR);

			/* increase lineLen */
			lineLen += MAX_CHAR_PER_LINE;
			line = (char*)realloc(line, lineLen);
			assert(line != NULL);

			ret = fgets(line, lineLen, infile);
			assert(ret != NULL);
		}

		if (strtok(line, " \t\n") != 0)
			(*pointsCount)++;
	}
	rewind(infile);

	/* find the no. objects of each object */
	(*dimNum) = 0;
	while (fgets(line, lineLen, infile) != NULL) {
		if (strtok(line, " \t\n") != 0) {
			/* ignore the id (first coordiinate): numCoords = 1; */
			while (strtok(NULL, " ,\t\n") != NULL) (*dimNum)++;
			break; /* this makes read from 1st object */
		}
	}
	rewind(infile);
	if (DEBUG) {
		printf("Points = %d\n", *pointsCount);
		printf("Dimensions = %d\n", *dimNum);
	}

	/* allocate space for objects[][] and read all objects */
	len = (*pointsCount) * (*dimNum);
	points = (float*)malloc((*pointsCount) * (*dimNum) * sizeof(float*));


	i = 0;
	/* read all objects */
	while (fgets(line, lineLen, infile) != NULL) {
		if (strtok(line, " \t\n") == NULL) continue;
		for (j = 0; j < (*dimNum); j++) {
			float coordinate = (float)atof(strtok(NULL, " ,\t\n"));

			points[i * (*dimNum) + j] = coordinate;
		}
		i++;
	}

	fclose(infile);
	free(line);

	return points;
}


void printClusters(float* points, int pointsCount, int dimNum) {

	for (size_t i = 0; i < pointsCount; i++)
	{
		printf("%d. ", i + 1);

		for (size_t j = 0; j < dimNum; j++)
		{
			printf("%f ", points[i * dimNum + j]);
		}

		printf("\n");
	}
}

int writeFile(char* filename,
	int clustersCount,
	int pointsCount,
	int dimNum,
	float* clusters,
	int* membership)
{
	FILE* fptr;
	int   i, j;
	char  outFileName[1024];

	sprintf(outFileName, "%s.cluster_centres", filename);
	printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
		clustersCount, outFileName);
	fptr = fopen(outFileName, "w");
	for (i = 0; i < clustersCount; i++) {
		fprintf(fptr, "%d ", i);
		for (j = 0; j < dimNum; j++)
			fprintf(fptr, "%f ", clusters[i * dimNum + j]);
		fprintf(fptr, "\n");
	}
	fclose(fptr);

	sprintf(outFileName, "%s.membership", filename);
	printf("Writing membership of N=%d data objects to file \"%s\"\n",
		pointsCount, outFileName);
	fptr = fopen(outFileName, "w");
	for (i = 0; i < pointsCount; i++)
		fprintf(fptr, "%d %d\n", i, membership[i]);
	fclose(fptr);

	return 1;
}

void generateRandomPoints(char* path, int pointsCount, int dimNum) {

	FILE* outFile = NULL;

	if ((outFile = fopen(path, "w")) == NULL) {
		fprintf(stderr, "Error: Cannot create a new file (%s)\n", path);
		return;
	}

	float min = -5;
	float max = 5;
	float diff = max - min;

	for (size_t i = 0; i < pointsCount; i++)
	{
		fprintf(outFile, "%d ", i);

		for (size_t j = 0; j < dimNum; j++)
		{
			float random = (float)rand() / (float)RAND_MAX;
			fprintf(outFile, "%f ", random * diff + min);
		}

		fprintf(outFile, "\n");
	}
}

void generateRandomPointsNormal(char* path, int pointsCount, int dimNum, int groups) {

	FILE* outFile = NULL;

	if ((outFile = fopen(path, "w")) == NULL) {
		fprintf(stderr, "Error: Cannot create a new file (%s)\n", path);
		return;
	}

	float min = -5;
	float max = 5;
	float diff = max - min;

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	//for (size_t i = 0; i < pointsCount; i++)
	//{
	//	fprintf(outFile, "%d ", i);

	//	for (size_t j = 0; j < dimNum; j++)
	//	{
	//		float random = (float)rand() / (float)RAND_MAX;
	//		fprintf(outFile, "%f ", random * diff + min);
	//	}

	//	fprintf(outFile, "\n");
	//}

	for (size_t g = 0; g < groups; g++)
	{
		std::normal_distribution<> d{ -(float)groups / 2 + g, 0.2 };

		for (size_t i = 0; i < pointsCount / groups; i++)
		{
			fprintf(outFile, "%d ", i);

			for (size_t j = 0; j < dimNum; j++)
			{
				fprintf(outFile, "%f ", d(gen));
			}

			fprintf(outFile, "\n");
		}
	}

}