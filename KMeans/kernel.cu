#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <chrono>
#include <random>

#define MAX_CHAR_PER_LINE 128
#define THREADS_PER_BLOCK 1024
#define MAX_ITERATIONS 500
#define DEBUG 1

#define MIN_ARGUMENTS 4
#define MSG_WRONG_ARG "Wrong argument: %s\n"
#define MSG_WRONG_NUMBER_OF_ARGUMENTS "Wrong number of arguments: %d\n"
#define MSG_USAGE "\
Usage: %s <path> <clusters> <threshold> [-c] [-s <value>]\n\
\tpath\t\t- path to a file with points \n\
\tclusters\t\t- number of clusters \n\
\tthreshold\t\t- stop condition - percentage of points that changed membership\n\
\t-c\t\t- shows also a result calculated on cpu\n\
\t-s value\t\t- set value as custom seed before generating clusters starting points\n"
#define ARG_SET_SEED "-s"
#define MSG_WRONG_SET_SEED "Please provide a value for set seed\n"
#define MSG_WRONG_CLUSTERS_COUNT "Wrong number of clusters. Please provide a positive integer\n"
#define MSG_WRONG_THRESHOLD "Wrong threshold. Please provide a positive float, ex. 0.0001\n"
#define ARG_SHOW_CPU "-c"

typedef struct arguments {
	char* filePath;
	char runCpu;
	int seed;
	int clustersCount;
	float threshold;
} arguments_t;

typedef struct kMeansArgs {
	float* points;
	int clustersCount;
	int pointsCount;
	int dimNum;
	float threshold;
	float* clusters;
	float* outClusters;
	int* iterations;
	int* memberships;
} kMeansArgs_t;

arguments_t initializeArguments(int argc, char** argv);
cudaError_t kMeansThrust(kMeansArgs_t* args);
cudaError_t kMeansReduce(kMeansArgs_t* args);
template <class T> cudaError_t reduce(T* data, int n);
void kMeansCpu(kMeansArgs_t* args);
void coalesceData(float** points, int pointsCount, int dimNum);
void unCoalesceData(float** points, int pointsCount, int dimNum);
void getMinMax(float* points, int pointsCount, int dimNum, float** maxCoordinates, float** minCoordinates);
float* generateClusters(float* minCoordinates, float* maxCoordinates, int seed, int clusterCount, int dimNum);
void printClusters(float* data, int numObjs, int numCoords);
void generateRandomPoints(char* path, int pointsCount, int dimNum);
void generateRandomPointsNormal(char* path, int pointsCount, int dimNum, int groups);
float* readFile(
	char* filename,
	int* pointsCount,
	int* dimNum);
int writeFile(char* filename,
	int numClusters,
	int numObjs,
	int numCoords,
	float* clusters,
	int* membership,
	char* specialSufix);

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

__global__ void pickCoordinate(float* points, float* clusters, int* membership, float* sums, int membersCount, int pointsCount, int pointsCountPowerOfTwo, int clustersCount, int dimNum, int dimension, int cluster) {

	extern __shared__ float pointData[];
	int threadId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int member = -1;
	float value = 0;

	if (threadId > pointsCountPowerOfTwo)
		return;

	if (threadId < pointsCount) {
		member = membership[threadId];
	}

	if (threadId < pointsCount) {
		value = points[dimension * pointsCount + threadId];
	}
	value = member == cluster ? value / membersCount : 0;

	sums[threadId] = value;
}

__global__ void pickMembership(int* membership, int* sums, int pointsCount, int pointsCountPowerOfTwo, int clustersCount, int cluster) {

	extern __shared__ int data[];

	int threadId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

	if (threadId > pointsCountPowerOfTwo)
		return;

	int member = threadId < pointsCount ? membership[threadId] : -1;
	member = member == cluster ? 1 : 0;
	sums[threadId] = member;
}

int main(int argc, char** argv)
{
	arguments_t args = initializeArguments(argc, argv);
	int pointsCount, dimNum;
	float* minCoordinates = NULL;
	float* maxCoordinates = NULL;
	int iterations = 0;

	kMeansArgs_t kMeansArgs = { 0 };

	printf("Reading file...\n");
	auto cpuStart = std::chrono::high_resolution_clock::now();
	kMeansArgs.points = readFile(args.filePath, &pointsCount, &dimNum);
	if (kMeansArgs.points == NULL) {
		return EXIT_FAILURE;
	}
	auto cpuEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = cpuEnd - cpuStart;
	printf("Elapsed %fs...\n", diff.count());

	coalesceData(&(kMeansArgs.points), pointsCount, dimNum);
	getMinMax(kMeansArgs.points, pointsCount, dimNum, &maxCoordinates, &minCoordinates);

	kMeansArgs.clustersCount = args.clustersCount;
	kMeansArgs.pointsCount = pointsCount;
	kMeansArgs.dimNum = dimNum;
	kMeansArgs.clusters = generateClusters(minCoordinates, maxCoordinates, args.seed, args.clustersCount, dimNum);;
	kMeansArgs.threshold = args.threshold;
	kMeansArgs.outClusters = new float[args.clustersCount * dimNum];
	kMeansArgs.iterations = &iterations;
	kMeansArgs.memberships = new int[pointsCount];

	cudaError_t cudaStatus;

	if (args.runCpu) {
		printf("===== Running CPU version ====\n");
		cpuStart = std::chrono::high_resolution_clock::now();
		kMeansCpu(&kMeansArgs);
		cpuEnd = std::chrono::high_resolution_clock::now();
		diff = cpuEnd - cpuStart;
		printf("Elapsed %fs...\n", diff.count());
		printf("Iterations: %d\n", iterations);
		unCoalesceData(&(kMeansArgs.outClusters), args.clustersCount, dimNum);
		writeFile(args.filePath, args.clustersCount, pointsCount, dimNum, (kMeansArgs.outClusters), kMeansArgs.memberships, (char*)"CPU");
	}

	printf("===== Running Thrust version on GPU =====\n");
	cudaStatus = kMeansThrust(&kMeansArgs);
	if (cudaStatus == cudaSuccess) {
		printf("Iterations: %d\n", iterations);
		unCoalesceData(&(kMeansArgs.outClusters), args.clustersCount, dimNum);
		writeFile(args.filePath, args.clustersCount, pointsCount, dimNum, (kMeansArgs.outClusters), kMeansArgs.memberships, (char*)"GPUThrust");
	}

	printf("===== Running custom version on GPU =====\n");
	cudaStatus = kMeansReduce(&kMeansArgs);
	if (cudaStatus == cudaSuccess) {
		printf("Iterations: %d\n", iterations);
		unCoalesceData(&(kMeansArgs.outClusters), args.clustersCount, dimNum);
		printf("WritingData...\n");
		writeFile(args.filePath, args.clustersCount, pointsCount, dimNum, (kMeansArgs.outClusters), kMeansArgs.memberships, (char*)"GPU");
	}

	printf("Terminating...\n");
	delete[] minCoordinates;
	delete[] maxCoordinates;
	delete[] kMeansArgs.clusters;
	delete[] kMeansArgs.memberships;
	delete[] kMeansArgs.outClusters;
	delete[] kMeansArgs.points;

	return EXIT_SUCCESS;
}

arguments_t initializeArguments(int argc, char** argv) {

	if (argc < MIN_ARGUMENTS) {
		fprintf(stderr, MSG_WRONG_NUMBER_OF_ARGUMENTS, argc);
		fprintf(stderr, MSG_USAGE, argv[0]);
		exit(EXIT_FAILURE);
	}

	arguments_t args = { argv[1], false, 0 };

	if ((args.clustersCount = atoi(argv[2])) == 0) {
		fprintf(stderr, MSG_WRONG_CLUSTERS_COUNT);
		fprintf(stderr, MSG_USAGE, argv[0]);
		exit(EXIT_FAILURE);
	}

	if ((args.threshold = (float)atof(argv[3])) == 0.0f) {
		fprintf(stderr, MSG_WRONG_CLUSTERS_COUNT);
		fprintf(stderr, MSG_USAGE, argv[0]);
		exit(EXIT_FAILURE);
	}

	for (size_t i = 4; i < argc; i++)
	{
		if (strcmp(argv[i], ARG_SET_SEED) == 0) {

			if (i + 1 < argc) {
				int seed = 0;
				if ((seed) = atoi(argv[i + 1]) != 0) {
					args.seed = seed;
					i++;
				}
				else {
					fprintf(stderr, MSG_WRONG_SET_SEED);
					fprintf(stderr, MSG_USAGE, argv[0]);
					exit(EXIT_FAILURE);
				}
			}
			else {
				fprintf(stderr, MSG_WRONG_SET_SEED);
				fprintf(stderr, MSG_USAGE, argv[0]);
				exit(EXIT_FAILURE);
			}
		}
		else if (strcmp(argv[i], ARG_SHOW_CPU) == 0) {
			args.runCpu = true;
		}
		else {
			fprintf(stderr, MSG_WRONG_ARG, argv[i]);
			fprintf(stderr, MSG_USAGE, argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	return args;
}

void kMeansCpu(kMeansArgs_t* args)
{
	int pointsCount = args->pointsCount, dimNum = args->dimNum, clustersCount = args->clustersCount;
	float* newCluster = new float[dimNum];
	float delta = 0;
	int iterationNumber = 0;
	for (size_t i = 0; i < clustersCount * dimNum; i++)
	{
		args->outClusters[i] = args->clusters[i];
	}

	do {
		delta = 0;
		iterationNumber++;
		printf(".");
		fflush(stdout);

		for (int point = 0; point < pointsCount; point++)
		{
			float minDistance = FLT_MAX;
			int currentMembership = -1;

			for (int k = 0; k < clustersCount; k++)
			{
				float distance = 0;

				for (int n = 0; n < dimNum; n++)
				{
					float diff = args->outClusters[n * clustersCount + k] - args->points[n * pointsCount + point];
					distance += diff * diff;
				}

				if (distance < minDistance) {
					minDistance = distance;
					currentMembership = k;
				}
			}

			if (args->memberships[point] != currentMembership) {
				delta++;
				args->memberships[point] = currentMembership;
			}
		}

		delta /= pointsCount;

		if (delta < args->threshold || iterationNumber > MAX_ITERATIONS)
			break;

		for (size_t k = 0; k < clustersCount; k++)
		{
			int members = 0;
			memset(newCluster, 0, dimNum * sizeof(float));

			for (size_t point = 0; point < pointsCount; point++)
			{
				if (args->memberships[point] == k)
				{
					members++;

					for (size_t n = 0; n < dimNum; n++)
					{
						newCluster[n] += args->points[n * pointsCount + point];
					}
				}
			}

			for (size_t n = 0; n < dimNum; n++)
			{
				args->outClusters[n * clustersCount + k] = newCluster[n] / members;
			}
		}

	} while (1);

	printf("\n");
	delete[] newCluster;
	*(args->iterations) = iterationNumber;
}

cudaError_t kMeansThrust(kMeansArgs_t* args)
{
	int pointsCount = args->pointsCount, dimNum = args->dimNum, clustersCount = args->clustersCount;
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
	float delta = FLT_MAX;
	cudaError_t cudaStatus;

	int blocksCount = (int)ceil((float)pointsCount / THREADS_PER_BLOCK);
	int newClustersBlocksCount = (int)ceil((float)clustersCount / THREADS_PER_BLOCK);

	int nearestPowerOfTwo = 1;
	while (nearestPowerOfTwo < clustersCount)
		nearestPowerOfTwo *= 2;
	int iterationNumber = 0;

	printf("Allocating memory...\n");

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

	cudaStatus = cudaMemcpy(dev_points, args->points, pointsCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_clusters, args->clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
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
		iterationNumber++;
		printf(".");
		pointsDistance << <blocksCount, THREADS_PER_BLOCK >> > (dev_points, dev_clusters, dev_membership, dev_membershipDims, dev_membershipChanged, clustersCount, pointsCount, dimNum);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "pointsDistance launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaMemcpyAsync(dev_points_sums, dev_points, pointsCount * dimNum * sizeof(int), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		delta = (float)thrust::reduce(thrust::device, dev_membershipChanged, dev_membershipChanged + pointsCount);
		delta /= pointsCount;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointsDistance!\n", cudaStatus);
			goto Error;
		}

		if (delta < args->threshold || iterationNumber > MAX_ITERATIONS)
			break;

		cudaStatus = cudaMemcpy(dev_currentMembership, dev_membership, pointsCount * sizeof(int), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		thrust::sort(thrust::device, dev_currentMembership, dev_currentMembership + pointsCount);
		auto pair = thrust::reduce_by_key(thrust::device, dev_currentMembership, dev_currentMembership + pointsCount, thrust::make_constant_iterator(1), dev_clusterSizesKeys, dev_clusterSizes);
		int usedClusters = (int)(pair.first - dev_clusterSizesKeys);
		thrust::sort_by_key(thrust::device, dev_membershipDims, dev_membershipDims + pointsCount * dimNum, dev_points_sums);
		thrust::reduce_by_key(thrust::device, dev_membershipDims, dev_membershipDims + pointsCount * dimNum, dev_points_sums, thrust::make_discard_iterator(), dev_clustersSums);

		updateClusters << <newClustersBlocksCount, THREADS_PER_BLOCK >> > (dev_clusters, dev_clustersSums, dev_clusterSizes, dev_clusterSizesKeys, clustersCount, dimNum, usedClusters);
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

	} while (1);
	printf("\n");
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed %fs\n", time / 1000);

	*(args->iterations) = iterationNumber;

	cudaStatus = cudaMemcpy(args->outClusters, dev_clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(args->memberships, dev_membership, pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
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

	return cudaStatus;
}

cudaError_t kMeansReduce(kMeansArgs_t* args)
{
	int pointsCount = args->pointsCount, dimNum = args->dimNum, clustersCount = args->clustersCount;
	float* dev_points = NULL;
	float* dev_sums = NULL;
	float* dev_clusters = NULL;
	int* dev_membership = NULL;
	int* dev_membershipSums = NULL;
	int* dev_membershipChanged = NULL;
	int membersCount = 0;
	float delta = FLT_MAX;
	cudaError_t cudaStatus;

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

	cudaStatus = cudaMalloc((void**)&dev_sums, pointsCountPowerOfTwo * sizeof(float));
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

	cudaStatus = cudaMalloc((void**)&dev_membershipSums, pointsCountPowerOfTwo * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membershipChanged, pointsCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_points, args->points, pointsCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_clusters, args->clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
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
		iterationNumber++;
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

		if (delta < args->threshold || iterationNumber > MAX_ITERATIONS)
			break;

		for (int k = 0; k < clustersCount; k++)
		{
			pickMembership << <reduceBlocksCount, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int) >> > (
				dev_membership,
				dev_membershipSums,
				pointsCount,
				pointsCountPowerOfTwo,
				clustersCount,
				k
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

			cudaStatus = reduce(dev_membershipSums, pointsCountPowerOfTwo);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "reduce launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}

			cudaStatus = cudaMemcpy(&membersCount, dev_membershipSums, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}

			if (membersCount > 0)
			{
				for (int i = 0; i < dimNum; i++)
				{
					pickCoordinate << <reduceBlocksCount, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (
						dev_points,
						dev_clusters,
						dev_membership,
						dev_sums,
						membersCount,
						pointsCount,
						pointsCountPowerOfTwo,
						clustersCount,
						dimNum,
						i,
						k
						);

					cudaStatus = cudaDeviceSynchronize();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching countCoordinate!\n", cudaStatus);
						goto Error;
					}

					cudaStatus = reduce(dev_sums, pointsCountPowerOfTwo);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "reduce launch failed: %s\n", cudaGetErrorString(cudaStatus));
						goto Error;
					}

					cudaStatus = cudaMemcpy(&(dev_clusters[i * clustersCount + k]), dev_sums, sizeof(float), cudaMemcpyDeviceToDevice);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
						goto Error;
					}
				}
			}
		}
	} while (1);
	printf("\n");
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed %fs\n", time / 1000);

	*(args->iterations) = iterationNumber;

	cudaStatus = cudaMemcpy(args->outClusters, dev_clusters, clustersCount * dimNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(args->memberships, dev_membership, pointsCount * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_membership);
	cudaFree(dev_membershipSums);
	cudaFree(dev_clusters);
	cudaFree(dev_membershipChanged);

	return cudaStatus;
}

template <class T>
cudaError_t reduce(T* data, int n) {

	int currentThreads = n;
	cudaError_t cudaStatus;

	while (currentThreads > 1) {
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

		currentThreads /= THREADS_PER_BLOCK;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceGlobal!\n", cudaStatus);
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

	delete[] * points;
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

	delete[] * points;
	*points = unCoalesced;
}

float* generateClusters(float* minCoordinates, float* maxCoordinates, int seed, int clusterCount, int dimNum) {

	float* clusters = new float[clusterCount * dimNum];

	if (seed == 0) {
		srand((unsigned)time(NULL));
	}
	else {
		srand(seed);
	}

	for (size_t i = 0; i < dimNum; i++)
	{
		float diff = maxCoordinates[i] - minCoordinates[i];

		for (size_t j = 0; j < clusterCount; j++)
		{
			float random = (float)rand() / (float)RAND_MAX;
			clusters[i * clusterCount + j] = random * diff + minCoordinates[i];
		}
	}

	return clusters;
}

void getMinMax(float* points, int pointsCount, int dimNum, float** maxCoordinates, float** minCoordinates) {

	(*minCoordinates) = new float[dimNum];
	(*maxCoordinates) = new float[dimNum];

	for (size_t i = 0; i < dimNum; i++)
	{
		(*minCoordinates)[i] = FLT_MAX;
		(*maxCoordinates)[i] = FLT_MIN;
	}

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
	char* line;
	int   lineLen;

	if ((infile = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error: no such file (%s)\n", filename);
		return NULL;
	}

	lineLen = MAX_CHAR_PER_LINE;
	line = (char*)malloc(lineLen);
	assert(line != NULL);

	(*pointsCount) = 0;
	while (fgets(line, lineLen, infile) != NULL) {
		while (strlen(line) == lineLen - 1) {
			len = (int)strlen(line);
			fseek(infile, -len, SEEK_CUR);

			lineLen += MAX_CHAR_PER_LINE;
			line = (char*)realloc(line, lineLen);
			assert(line != NULL);
			assert(fgets(line, lineLen, infile) != NULL);
		}

		if (strtok(line, " \t\n") != 0)
			(*pointsCount)++;
	}
	rewind(infile);

	(*dimNum) = 0;
	while (fgets(line, lineLen, infile) != NULL) {
		if (strtok(line, " \t\n") != 0) {
			while (strtok(NULL, " ,\t\n") != NULL) (*dimNum)++;
			break;
		}
	}
	rewind(infile);
	if (DEBUG) {
		printf("Points = %d\n", *pointsCount);
		printf("Dimensions = %d\n", *dimNum);
	}

	len = (*pointsCount) * (*dimNum);
	points = new float[(*pointsCount) * (*dimNum)];


	i = 0;
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

	for (int i = 0; i < pointsCount; i++)
	{
		printf("%d. ", i + 1);

		for (int j = 0; j < dimNum; j++)
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
	int* membership,
	char* specialSufix)
{
	FILE* fptr;
	int   i, j;
	char  outFileName[1024];

	sprintf(outFileName, "%s%s.cluster_centres", filename, specialSufix);
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

	sprintf(outFileName, "%s%s.membership", filename, specialSufix);
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

	for (int i = 0; i < pointsCount; i++)
	{
		fprintf(outFile, "%d ", i);

		for (int j = 0; j < dimNum; j++)
		{
			float random = (float)rand() / (float)RAND_MAX;
			fprintf(outFile, "%f ", random * diff + min);
		}

		fprintf(outFile, "\n");
	}

	fclose(outFile);
}

void generateRandomPointsNormal(char* path, int pointsCount, int dimNum, int groups) {

	FILE* outFile = NULL;

	if ((outFile = fopen(path, "a")) == NULL) {
		fprintf(stderr, "Error: Cannot create a new file (%s)\n", path);
		return;
	}

	float min = -5;
	float max = 5;
	float diff = max - min;

	int rows = (int)sqrt(groups);
	int columns = rows;

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	for (size_t x = 0; x < groups; x++)
	{
		for (size_t y = 0; y < groups; y++)
		{
			for (size_t z = 0; z < groups; z++)
			{
				std::normal_distribution<> dx{ -5 + x * (float)10 / groups, 0.3 };
				std::normal_distribution<> dy{ -5 + y * (float)10 / groups, 0.5 };
				std::normal_distribution<> dz{ -5 + z * (float)10 / groups, 0.3 };

				for (int i = 0; i < pointsCount / pow(groups, 3); i++)
				{
					fprintf(outFile, "%d ", i);
					fprintf(outFile, "%f ", dx(gen));
					fprintf(outFile, "%f ", dy(gen));
					fprintf(outFile, "%f ", dz(gen));

					fprintf(outFile, "\n");
				}
			}

		}
	}

	fclose(outFile);
}