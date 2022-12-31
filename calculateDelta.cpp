__global__ void calculateDelta(float* clusters, float* newClusters, int clustersCount, int dimNum, int distancesSize, float* delta) {

	int clusterId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	extern __shared__ float distances[];

	if (clusterId >= clustersCount)
		return;

	if (threadIdx.x < distancesSize)
		distances[clusterId] = 0;

	for (size_t i = 0; i < dimNum; i++)
	{
		float diff = clusters[i * clustersCount + clusterId] - newClusters[i * dimNum + clusterId];
		distances[clusterId] += diff * diff;
	}

	int workingThreads = distancesSize;
	auto stepSize = 1;
	__syncthreads();

	while (workingThreads > 0)
	{
		if (threadIdx.x < workingThreads) // still alive?
		{
			const auto fst = threadIdx.x * stepSize * 2;
			const auto snd = fst + stepSize;
			distances[fst] += distances[snd];
		}

		stepSize <<= 1;
		workingThreads >>= 1;
	}

	if (threadIdx.x == 0)
	{
		delta[threadIdx.x] = distances[threadIdx.x];
	}
}