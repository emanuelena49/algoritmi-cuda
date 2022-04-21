
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
# pragma once
#include <memory>
#include <stdio.h>

#include <math.h>

#include "gestione_errori.cu"



/// <summary>
/// Calcola quanti blocchi si deve lanciare per processare 
/// un vettore di lungh n, con threadsPerBlock thread per blocco
/// </summary>
/// <param name="n"></param>
/// <param name="threadsPerBlock"></param>
/// <returns></returns>
int _calculateB(int n, int threadsPerBlock) {
	return (int) ceil(
		(double) n / 
		(double) threadsPerBlock
	);
}

__global__ void countOccurrenciesV1_kernel(int* v, int n, int x, int* result) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n) {
		if (v[tid] == x) {
			// atomicAdd(result, 1);
		}
	}
}

__global__ void a() {

}


/// <summary>
/// Conta le occorrenze di x nel vettore v
/// </summary>
/// <param name="v"></param>
/// <param name="n"></param>
/// <param name="x"></param>
/// <return>Il numero di occorrenze osservate</return>
int countOccurrenciesV1(int* v, int n, int x) {

	// allocazione memoria su device
	// - n spazi per il vettore di input
	// - 1 spazio finale per il risultato
	int *vDevice;
	HANDLE_ERROR(cudaMalloc((void**)&vDevice, sizeof(int) * (n+1)));
	int *resultDevice = &vDevice[n];

	// copia vettore di input nel device + inizializzazione a 0 spazio risultato
	HANDLE_ERROR(cudaMemcpy(vDevice, v, sizeof(int) * n, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(resultDevice, 0, sizeof(int)));

	// calcolo numero blocchi
	const int threadsPerBlock = 256;
	int nBlocks = _calculateB(n, threadsPerBlock);

	// lancio kernel
	// countOccurrenciesV1_kernel <<< nBlocks, threadsPerBlock>>>(vDevice, n, x, resultDevice
	// a<<<nBlocks, threadsPerBlock>>>();
	cudaDeviceSynchronize();
	checkKernelError("kernel v1");

	// copia risultato su host
	int result;
	HANDLE_ERROR(cudaMemcpy(&result, resultDevice, sizeof(int), cudaMemcpyDeviceToHost));
	
	// liberazione memoria device
	HANDLE_ERROR(cudaFree(vDevice));

	return result;
}