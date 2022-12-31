#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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

#define MAX_CHAR_PER_LINE 128
#define THREADS_PER_BLOCK 1024
#define MAX_ITERATIONS 500
#define DEBUG 1

cudaError_t kMeansCuda(float* points, int clustersCount, int pointsCount, int dimNum, float threshold, float** clusters, int* iterations);
float* readFile(
	char* filename,
	int* numObjs,
	int* numCoords);
int writeFile(char* filename,
	int numClusters,
	int numObjs,
	int numCoords,
	float* clusters,
	int* membership);
void coalesceData(float** points, int pointsCount, int dimNum);
void unCoalesceData(float** points, int pointsCount, int dimNum);
void printClusters(float* data, int numObjs, int numCoords);

__global__ void pointsDistance(float* points, float* clusters, int* membership, int* membershipChanged, int clustersCount, int pointsCount, int dimNum)
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
			// jakby wraz z przynależnością wpisywać wymiar, to ułatwiłoby to zastosowanie thrust::reduce_by_key
			// np. od 0 do clustersCount to pierwszy wymiar, potem od clustersCount do 2 * clustersCount to drugi wymiar itd
			currentMembership = i;
		}
	}

	__syncthreads();
	if (membership[pointNum] != currentMembership) {
		membershipChanged[pointNum] = 1;
	}
	else {
		membershipChanged[pointNum] = 0;
	}
	__syncthreads();
	membership[pointNum] = currentMembership;
}

__global__ void newClustersMean(float* points, int* membership, float* newClusters, int clustersCount, int pointsCount, int dimNum)
{
	int clusterId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int members = 0;

	extern __shared__ float newClustersSums[];

	if (clusterId >= clustersCount)
		return;

	for (size_t j = 0; j < dimNum; j++)
	{
		newClustersSums[j * clustersCount + clusterId] = 0;
	}

	for (size_t i = 0; i < pointsCount; i++)
	{
		if (membership[i] == clusterId)
			for (size_t j = 0; j < dimNum; j++)
			{
				newClustersSums[j * clustersCount + clusterId] += points[j * pointsCount + i];
				members++;
			}
	}

	__syncthreads();
	if (members != 0)
		for (size_t j = 0; j < dimNum; j++)
		{
			newClusters[j * clustersCount + clusterId] = newClustersSums[j * clustersCount + clusterId] / members;
		}
}

int main()
{
	// TODO: Wczytywanie z parametrów threshold, liczby centroidów (K), wyznaczanie początkowych punktów (losowo z zakresu wczytanych danych??)
	const char* fileName = "C:\\Users\\spawl\\Desktop\\MiNI_5\\GPU\\kmeans_demo\\kmeans\\Image_data\\color100.txt";
	int clustersCount = 10;
	int pointsCount = 0, dimNum;
	float threshold = 0.000001;

	printf("Reading file...\n");
	auto cpuStart = std::chrono::high_resolution_clock::now();
	float* points = readFile((char*)fileName, &pointsCount, &dimNum);
	auto cpuEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = cpuEnd - cpuStart;
	printf("Elapsed %fs...\n", diff.count());

	coalesceData(&points, pointsCount, dimNum);
	float* clusters;
	int iterations;
	auto cudaStatus = kMeansCuda(points, clustersCount, pointsCount, dimNum, threshold, &clusters, &iterations);
	if (cudaStatus == cudaSuccess) {
		printf("Iterations: %d\n", iterations);
		unCoalesceData(&clusters, clustersCount, dimNum);
		printClusters(clusters, clustersCount, dimNum);
	}


	return EXIT_SUCCESS;
}

cudaError_t kMeansCuda(float* points, int clustersCount, int pointsCount, int dimNum, float threshold, float** clusters, int* iterations)
{
	float* dev_points = NULL;
	float* dev_clusters = NULL;
	float* dev_newClusters = NULL;
	int* dev_membership = NULL;
	int* dev_membershipChanged = NULL;
	float* dev_delta = NULL;
	float delta = FLT_MAX;
	cudaError_t cudaStatus;

	int blocksCount = (int)ceil((float)pointsCount / THREADS_PER_BLOCK);
	int newClustersBlocksCount = (int)ceil((float)clustersCount / THREADS_PER_BLOCK);

	int nearestPowerOfTwo = 1;
	while (nearestPowerOfTwo < clustersCount)
		nearestPowerOfTwo *= 2;
	int iterationNumber = 0;

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

	cudaStatus = cudaMalloc((void**)&dev_clusters, clustersCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_newClusters, clustersCount * dimNum * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_membership, pointsCount * sizeof(int));
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

	cudaStatus = cudaMemcpy(dev_points, points, pointsCount * dimNum * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaMemset(dev_membership, -1, pointsCount * sizeof(int));


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Calculating on GPU...\n");
	cudaEventRecord(start, 0);
	do {
		float* temp = dev_clusters;
		dev_clusters = dev_newClusters;
		dev_newClusters = temp;

		pointsDistance << <blocksCount, THREADS_PER_BLOCK >> > (dev_points, dev_clusters, dev_membership, dev_membershipChanged, clustersCount, pointsCount, dimNum);
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
		// obliczenie nowych współrzędnych centroidów - środek ciężkości tych, które należą 
		newClustersMean << <newClustersBlocksCount, THREADS_PER_BLOCK, clustersCount* dimNum * sizeof(float) >> > (dev_points, dev_membership, dev_newClusters, clustersCount, pointsCount, dimNum);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "newClustersMean launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching newClustersMean!\n", cudaStatus);
			goto Error;
		}
		// obliczenie delty między poprzednim rozwiązaniem a obecnym - średnia róznica odległości między poprzednimi centroidami a nowymi 
		delta = (float)thrust::reduce(thrust::device, dev_membershipChanged, dev_membershipChanged + pointsCount);
		delta /= pointsCount;

	} while (delta > threshold && ++iterationNumber < MAX_ITERATIONS);
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed %fs\n", time / 1000);

	*iterations = iterationNumber;
	*clusters = new float[clustersCount * dimNum];

	cudaStatus = cudaMemcpy(*clusters, dev_newClusters, clustersCount * dimNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_newClusters);
	cudaFree(dev_membership);
	cudaFree(dev_delta);

	return cudaStatus;
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
	int64_t indx = 0;

	for (size_t i = 0; i < pointsCount; i++)
	{
		for (size_t j = 0; j < dimNum; j++)
		{
			unCoalesced[i * dimNum + j] = (*points)[j*pointsCount + i];
		}
	}

	free(*points);
	*points = unCoalesced;
}

float* readFile(
	char* filename,
	int* pointsCount,       /* no. data objects (local) */
	int* dimNum)     /* no. coordinates */
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
		for (j = 0; j < (*dimNum); j++)
			points[i * (*dimNum) + j] = (float)atof(strtok(NULL, " ,\t\n"));
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
