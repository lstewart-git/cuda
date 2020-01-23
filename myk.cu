#include <stdio.h>
#include <iostream>
#include <math.h> 
using namespace std;

#define PI 3.14159265

// GPU ERROR CHECKING MACRO
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// COMPUTE KERNEL
__global__
void mykern(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] =(a*x[i] + y[i]);
}

int main(void)
{
  cout << "LES CUDA Kernel Test\n" << std::flush;
  int N = 100000000;

  float my_const = 0.5;

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  gpuErrchk(cudaMalloc(&d_x, N*sizeof(float))); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 2.0f;
    y[i] = 0.0f;
  }

  cout << "Setup Done\n" << std::flush;

  for( int index = 0; index < 50; index+=1 ) {  
    // COPY MEMORY TO DEVICE
    gpuErrchk( cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform KERNEL on all elements
    mykern<<<(N+255)/256, 256>>>(N, my_const, d_x, d_y);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // COPY DEVICE MEMORY TO CPU 
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (index%10 ==0){
      cout << " index:" << index << "\n" << std::flush;}
  }

  cout << "GPU Computations Done\n" << std::flush;

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-50.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}

