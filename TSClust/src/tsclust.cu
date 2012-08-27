//CUDA implementation of k-means using DTW
//Author: Jeffrey Wong
//Last Modified: 27 May 2011
//Status: Working
//To Do: Code decomposition.  Want to write DTW, kmeans, and tsdist as separate
//files.  Problem, NVCC does not allow a kernel to use a device function from
//another file.  How can we have the kmeans kernel refer to a DTW file?

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<math.h>
#include<cuda.h>
#include"cuPrintf.cu"

//////
//MISC
//////

int checkCUDAError()
{
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("Error during initialization: %s\n", cudaGetErrorString(error));
    return -1;
  }
  return 1;
}

/////////////
//DTW HELPERS
/////////////

__host__ __device__ float min(float x, float y, float z)
{
  float smallest = x;
  if(smallest > y)
    smallest = y;
  if(smallest > z)
    smallest = z;
  return smallest;
}

//local distance for dtw
__host__ __device__ float distance(float x, float y)
{
  if(x > y)
    return (x-y);
  else return (y-x);
}

//////////
//CUDA DTW
//////////

__device__ float cudaDTW(float* x, float* y, int length, float* DTW, int index)
{
  int n = length+1;
  float* dtw = DTW + index*n;
  for(int i = 0; i < n; i++)
  {
    dtw[i] = 0;
  }

  float cost;
  float next;

  for(int i=0; i < n; i++)
  {
    dtw[i] = 9999;
  }
  float prev = 9999;
  dtw[0] = 0;

  for(int i = 0; i < length; i++)
  {
    for(int j = 0; j < length; j++)
    {
      cost = distance(x[i], y[j]);
      next = cost + min(dtw[j+1], prev, dtw[j]);
      if(i == length - 1 && j == length-1)
      {
        return next;
      }
      dtw[j+1] = next;
      dtw[j] = prev;
      prev = next;
    }
  }
  return next;
}

float DTW(float* x, float* y, int length)
{
  int n = (length+1);
  float* DTW = (float*) malloc(n * sizeof(float));
  if(DTW == NULL) { return -1; }
  for(int i = 0; i < n; i++)
  {
    DTW[i] = 0;
  }

  float cost;
  float next;

  for(int i=0; i < n; i++)
  {
    DTW[i] = 9999;
  }
  float prev = 9999;
  DTW[0] = 0;

  for(int i = 0; i < length; i++)
  {
    for(int j = 0; j < length; j++)
    {
      cost = distance(x[i], y[j]);
      next = cost + min(DTW[j+1], prev, DTW[j]);
      DTW[j+1] = next;
      DTW[j] = prev;
      prev = next;
    }
  }

  free(DTW);
  return next;
}

////////////////
//KMEANS KERNELS
////////////////

__global__ void cudaKMeansDistances(float* d_data, float* d_centers, float* DTW, int numTS, int ts_length, int k, int* d_clusters)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= numTS) return;

  float* ts = d_data + i*ts_length;
  float shortestDistance = cudaDTW(ts, d_centers, ts_length, DTW, i);
  int shortestDistanceIndex = 0;
  for(int j = 1; j < k; j++)
  {
    float* center = d_centers + j*ts_length;
    float distance = cudaDTW(ts, center, ts_length, DTW, i);
    if(distance < shortestDistance)
    {
      shortestDistance = distance;
      shortestDistanceIndex = j;
    }
  }
  d_clusters[i] = shortestDistanceIndex;
}            

__global__ void cudaKMeansCenters(float* d_data, float* d_centers, int numTS, int ts_length, int k, int* d_clusters)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= ts_length) return;

  extern __shared__ int clusterSize[];
  if(threadIdx.x == 0) 
  {
    for(int j = 0; j < k; j++)
    {
      clusterSize[j] = 0;
    }
    for(int j = 0; j < numTS; j++)
    {
      int cluster = d_clusters[j];
      clusterSize[cluster] = clusterSize[cluster] + 1;
    }
  }
  __syncthreads();

  for(int j = 0; j < k; j++)
  {
    float* center = d_centers + j*ts_length;
    center[i] = 0;
  }
  for(int j = 0; j < numTS; j++)
  {
    float* ts = d_data + j*ts_length;
    int cluster = d_clusters[j];
    float* center = d_centers + cluster*ts_length;
    center[i] = center[i] + ts[i];
  }
  for(int j = 0; j < k; j++)
  {
    float* center = d_centers + j*ts_length;
    center[i] = center[i] / clusterSize[j];
  }
}

////////////////////
//R KMEANS WITH CUDA
////////////////////

//Entry point for R to run cudaKMeans
extern "C" void cudaRKMeans(float* data, float* centers, int* numTSR, int* ts_lengthR, int* kR, int* clusters, float* withinss, int* success)
{
  int numTS = numTSR[0];
  int ts_length = ts_lengthR[0];
  int k = kR[0];

  float* d_data;
  float* d_centers;
  int* d_clusters;
  float* dtw;

  if(cudaSuccess != cudaMalloc( (void**)&d_data, numTS * ts_length * sizeof(float)))
  {
    printf("Could not cudaMalloc data\n");
    success[0] = -1;
    return;
  }
  if(cudaSuccess != cudaMalloc( (void**)&d_centers, k * ts_length * sizeof(float)))
  {
    printf("Could not cudaMalloc centers\n");
    cudaFree(d_data);
    success[0] = -1;
    return;
  }
  if(cudaSuccess != cudaMalloc( (void**)&d_clusters, numTS * sizeof(int)))
  {
    printf("Could not cudaMalloc clusters\n");
    cudaFree(d_data);
    cudaFree(d_centers);
    success[0] = -1;
    return;
  }
  if(cudaSuccess != cudaMalloc( (void**)&dtw, numTS * (ts_length+1) * sizeof(float)))
  {
    printf("Could not cudaMalloc DTW\n");
    cudaFree(d_data);
    cudaFree(d_centers);
    cudaFree(d_clusters);
    success[0] = -1;
    return;
  }

  cudaMemcpy(d_data, data, numTS*ts_length*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_centers, centers, k*ts_length*sizeof(float), cudaMemcpyHostToDevice);
 
  if(checkCUDAError() == -1)
  {
    return;
  }

  printf("Computing k-means...\n");

  int numIters = 10;
  int blocksize = 512;
  int gridsizeDistances = numTS / blocksize;
  int gridsizeCenters = ts_length / blocksize;
  for(int i = 0; i < numIters; i++)
  {
    printf("Begin iteration %d\n", i);
    cudaThreadSynchronize();  
    cudaKMeansDistances<<<gridsizeDistances + 1, blocksize>>>(d_data, d_centers, dtw, numTS, ts_length, k, d_clusters);
    cudaThreadSynchronize();
    cudaKMeansCenters<<<gridsizeCenters + 1, blocksize, k * sizeof(float)>>>(d_data, d_centers, numTS, ts_length, k, d_clusters); 
  }
  cudaThreadSynchronize();
  
  cudaMemcpy(clusters, d_clusters, numTS*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(centers, d_centers, k * ts_length * sizeof(float), cudaMemcpyDeviceToHost);

  //Compute withinss
  for(int i = 0; i < numTS; i++)
  {
    float* ts = data + ts_length*i;
    int cluster = clusters[i];
    float* center = centers + cluster*ts_length;
    float distanceToCenter = DTW(ts, center, ts_length);
    withinss[cluster] = withinss[cluster] + (distanceToCenter * distanceToCenter);
  }

  for(int i = 0; i < numTS; i++)
  {
    printf("%d ", clusters[i]);
  }
  printf("\n");

  cudaFree(d_data);
  cudaFree(d_centers);
  cudaFree(d_clusters);
  cudaFree(dtw);

  success[0] = 1;
  if(checkCUDAError() == -1)
  {
    return;
  }
  cudaThreadExit();
}

/////////////////
//R KMEANS WITH C
/////////////////

//Entry point for R to run host KMeans
extern "C" void RKMeans(float* data, float* centers, int* numTSR, int* ts_lengthR, int* kR, int* clusters, float* withinss, int* success)
{
  int numTS = numTSR[0];
  int ts_length = ts_lengthR[0];
  int k = kR[0];

  int* ktable = (int*)malloc(sizeof(int) * k);
  for(int i = 0; i < k; i++)
  {
    ktable[i] = 0;
  }
      
  int numIters = 10;
  for(int rep = 0; rep < numIters; rep++)
  {
    printf("Begin iteration %d\n", rep);
    //Compute cluster assignments and cluster size
    for(int i = 0; i < numTS; i++)
    {
      float* ts = data + ts_length*i;
      float smallestDistance = DTW(ts, centers, ts_length);
      if(smallestDistance == -1) {success[0] = -1; return;}
      int smallestDistanceIndex = 0;
      //printf("DTW for TS %d with center 0: %f\n", i, smallestDistance);
      for(int j = 1; j < k; j++)
      {
        float* center = centers + ts_length*j;
        float dtw = DTW(ts, center, ts_length);
        if(dtw == -1) {success[0] = -1; return;}
        //printf("DTW for TS %d with center %d: %f\n", i, j, dtw);
        if(dtw < smallestDistance)
        {
          smallestDistance = dtw;
          smallestDistanceIndex = j;
        }
      }
      //printf("Assinging TS %d to cluster %d\n", i, smallestDistanceIndex);
      clusters[i] = smallestDistanceIndex;
      ktable[smallestDistanceIndex] = ktable[smallestDistanceIndex] + 1;
    }

    //Reset centers
    for(int i = 0; i < k*ts_length; i++)
    {
      centers[i] = 0;
    }

    //Set centers to be center of data
    for(int i = 0; i < numTS; i++)
    {
      float* ts = data + ts_length*i;
      int cluster = clusters[i];
      float* center = centers + ts_length*cluster;
      for(int j = 0; j < ts_length; j++)
      {
        center[j] = center[j] + ts[j];
      }
    }

    for(int i = 0; i < k; i++)
    {
      float* center = centers + ts_length*i;
      int clusterSize = ktable[i];
      for(int j = 0; j < ts_length; j++)
      {
        center[j] = center[j] / clusterSize;
      }
    }

    for(int i = 0; i < k; i++)
    {
      ktable[i] = 0;
    }
  }

  //Final Steps:
  //Compute withinss
  //Reindex cluster assignments to start with 1
  //Clean up memory
  for(int i = 0; i < numTS; i++)
  {
    float* ts = data + ts_length*i;
    int cluster = clusters[i];
    float* center = centers + cluster*ts_length;
    float distanceToCenter = DTW(ts, center, ts_length);
    withinss[cluster] = withinss[cluster] + (distanceToCenter*distanceToCenter);
  }

  free(ktable);
  success[0] = 1;
}

//////////////////
//R TS DIST WITH C
//////////////////

//Entry point for R to run host TS distance matrix
extern "C" void RTSDist(float* data, int* numTSR, int* ts_lengthR, float* distances)
{
  int numTS = numTSR[0];
  int ts_length = ts_lengthR[0];
  for(int i = 0; i < numTS; i++)
  {
    float* ts1 = data + i*ts_length;
    for(int j = i+1; j < numTS; j++)
    {
      float* ts2 = data + j*ts_length;
      float distance = DTW(ts1, ts2, ts_length);
      if(i == 0)
        distances[j-1] = distance;
      else
        distances[numTS*i - ((i+1)*(i+2)/2) + j] = distance;
    }
  }
}

//////////////////////
//R TS DIST WITH CUDA
//////////////////////

__global__ void cudaTSDistKer(float* d_data, float* DTW, int numTS, int ts_length, int totalComparisons, float* d_distances)
{
  int z = blockDim.x * blockIdx.x + threadIdx.x;
  if(z >= totalComparisons) return;

  int i, j;

  float a = -1;
  float b = -1 + 2*numTS;
  float c = -2*z;

  float i1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
  i = (int)i1;
  float j1 = z - i*numTS + ((i1+1)*(i1+2)/2);
  j = (int)j1;

  float* ts1 = d_data + i * ts_length;
  float* ts2 = d_data + j * ts_length;

  float dtw = cudaDTW(ts1, ts2, ts_length, DTW, z);
  d_distances[z] = dtw;
}

//Entry point for R to run GPU TS distance matrix
extern "C" void cudaRTSDist(float* data, int* numTSR, int* ts_lengthR, float* distances, int* success)
{
  int numTS = numTSR[0];
  int ts_length = ts_lengthR[0];

  float* d_data;
  float* d_distances;
  float* dtw;

  int totalComparisons = ((float)numTS/2) * ((float)numTS-1);
  printf("%d ", totalComparisons);
  if(cudaSuccess != cudaMalloc( (void**)&d_data, numTS * ts_length * sizeof(float)))
  {
    printf("Could not cudaMalloc data\n");
    return;
  }
  if(cudaSuccess != cudaMalloc( (void**)&d_distances, totalComparisons * sizeof(float)))
  {
    printf("Could not cudaMalloc distances\n");
    cudaFree(d_data);
    return;
  }
  if(cudaSuccess != cudaMalloc( (void**)&dtw, totalComparisons * (ts_length + 1) * sizeof(float)))
  {
    printf("Could not cudaMalloc dtw\n");
    cudaFree(d_data);
    cudaFree(d_distances);
    return;
  }

  cudaMemcpy(d_data, data, numTS * ts_length * sizeof(float), cudaMemcpyHostToDevice);

  int blocksize = totalComparisons;
  if(blocksize > 512)
    blocksize = 512;
  int gridsize = totalComparisons / blocksize;
  printf("Calculating distances...\n");
  cudaTSDistKer<<<gridsize+1, blocksize>>>(d_data, dtw, numTS, ts_length, totalComparisons, d_distances);
  cudaThreadSynchronize();
  cudaMemcpy(distances, d_distances, totalComparisons * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_distances);
  cudaFree(dtw);

  cudaThreadExit();
  success[0] = 1;
}

///////////////////////////////////
//KMEANS ON THE HOST FOR REFERENCE
///////////////////////////////////

int* cpuKMeans(float* data, float* centers, int numTS, int ts_length, int k)
{
  int* clusters = (int*)malloc(sizeof(int) * numTS);
  int* ktable = (int*)malloc(sizeof(int) * k);
  for(int i = 0; i < k; i++)
  {
    ktable[i] = 0;
  }
     
  int numIters = 10;
  for(int rep = 0; rep < numIters; rep++)
  {
    printf("Begin iteration %d\n", rep);
    //Compute cluster assignments and cluster size
    for(int i = 0; i < numTS; i++)
    {
      float* ts = data + ts_length*i;
      float smallestDistance = DTW(ts, centers, ts_length);
      int smallestDistanceIndex = 0;
      //printf("DTW for TS %d with center 0: %f\n", i, smallestDistance);
      for(int j = 1; j < k; j++)
      {
        float* center = centers + ts_length*j;
        float dtw = DTW(ts, center, ts_length);
        //printf("DTW for TS %d with center %d: %f\n", i, j, dtw);
        if(dtw < smallestDistance)
        {
          smallestDistance = dtw;
          smallestDistanceIndex = j;
        }
      }
      //printf("Assinging TS %d to cluster %d\n", i, smallestDistanceIndex);
      clusters[i] = smallestDistanceIndex;
      ktable[smallestDistanceIndex] = ktable[smallestDistanceIndex] + 1;
    }

    //Reset centers
    for(int i = 0; i < k*ts_length; i++)
    {
      centers[i] = 0;
    }

    //Set centers to be center of data
    for(int i = 0; i < numTS; i++)
    {
      float* ts = data + ts_length*i;
      int cluster = clusters[i];
      float* center = centers + ts_length*cluster;
      for(int j = 0; j < ts_length; j++)
      {
        center[j] = center[j] + ts[j];
      }
    }

    for(int i = 0; i < k; i++)
    {
      float* center = centers + ts_length*i;
      int clusterSize = ktable[i];
      for(int j = 0; j < ts_length; j++)
      {
        center[j] = center[j] / clusterSize;
      }
    }

    for(int i = 0; i < k; i++)
    {
      ktable[i] = 0;
    }
  }

  //Final Steps:
  //Clean up memory
  
  free(ktable);
  return clusters;
}

int main(int argc, char** argv)
{
  if(argc != 5)
  {
    printf("Usage: ./kmeans <-cpu or -gpu> <numTS> <ts_length> <k>\n");
    return -1;
  }
  srand(100);
  int numTS = atoi(argv[2]);
  int ts_length = atoi(argv[3]);
  int k = atoi(argv[4]);

  float* data = (float*)malloc(sizeof(float) * numTS * ts_length);
  for(int i = 0; i < numTS*ts_length; i++)
  {
    if(i < numTS*ts_length / 2)
      data[i] = rand() % 300;
    else data[i] = (rand() % 10) * (rand() % 10);
  }

  float* centers = (float*)malloc(sizeof(float) * k * ts_length);
  for(int i = 0; i < k * ts_length; i++) 
  {
    if(i < k*ts_length)
      centers[i] = rand() % 150;
    else centers[i] = (rand() % 10) * (rand() % 10);
  }

  if(strcmp(argv[1], "-cpu") == 0)
  {
    int* clusters = cpuKMeans(data, centers, numTS, ts_length, k);
    for(int i = 0; i < numTS; i++)
    {
      printf("%d ", clusters[i]);
    }
    printf("\n");

    free(data);
    free(centers);
    free(clusters);

    return 0;
  }
  else if(strcmp(argv[1], "-gpu") == 0)
  {
    int* clusters = (int*)malloc(numTS * sizeof(int));
    float* withinss = (float*)malloc(k * sizeof(float));
    int* success = (int*)malloc( sizeof(int) );
    cudaRKMeans(data, centers, &numTS, &ts_length, &k, clusters, withinss, success);

    free(data);
    free(centers);
    free(clusters);
    free(withinss);
    free(success);
    return 0;
  }
}
