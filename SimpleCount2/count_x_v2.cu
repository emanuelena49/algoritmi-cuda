#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_errors.cuh"
#include <math.h>


/// <summary>
/// SOLUZIONE 2.1: 1 thread per dato, uso di atomicAdd.
/// Se i thread non bastano per analizzare la quantità 
/// di dati richiesta, la strategia d'accesso è coalescente.
/// </summary>
/// <param name="v"></param>
/// <param name="n"></param>
/// <param name="x"></param>
/// <param name="result"></param>
/// <returns></returns>
__global__ void countOccurrenciesV2_coalescent_kernel(int* v, int n, int x, int* result, const int nThreads) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int remainingData = n;
	int round = 0;

	do {
		__syncthreads();

		int index = tid + round*nThreads;

		if (index < n) {
			if (v[index] == x) {
				atomicAdd(result, 1);
			}
		}

		round ++;
		remainingData -= nThreads;

	} while (remainingData > 0);
}

/// <summary>
/// SOLUZIONE 2.2: 1 thread per dato, uso di atomicAdd.
/// Se i thread non bastano per analizzare la quantità 
/// di dati richiesta, la strategia d'accesso è in stride 
/// (ogni thread ha la sua sequenza).
/// </summary>
/// <param name="v"></param>
/// <param name="n"></param>
/// <param name="x"></param>
/// <param name="result"></param>
/// <returns></returns>
__global__ void countOccurrenciesV2_stride_kernel(int* v, int n, int x, int* result, const int nThreads) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int remainingData = n;
	int round = 0;
	int startPosition = tid * (int) ceil((float)n/(float)nThreads);

	do {
		__syncthreads();

		int index = startPosition + round;

		if (index < n) {
			if (v[index] == x) {
				atomicAdd(result, 1);
			}
		}

		round++;
		remainingData -= nThreads;

	} while (remainingData > 0);
}



/// <summary>
/// Strategia di accesso alla memoria in caso di size > nThreads
/// 
/// * può essere coalescente (ad ogni round si legge un blocco contiguo)
/// * oppure stride (i thread leggono la "sequenza" a loro assegnata, ad ogni 
///		round si accede quindi a tante posizioni frammentate)
/// </summary>
enum MemoryAccessStrategy { COALESCENCE, STRIDE };

/// <summary>
/// Conta le occorrenze di x nel vettore v, specificando manualmente il numero
/// di blocchi e i thread per blocco richiesti.
/// </summary>
/// <param name="v"></param>
/// <param name="n"></param>
/// <param name="x"></param>
/// <param name="nBlocks">Quanti blocchi voglio utilizzare?</param>
/// <param name="threadPerBlock">Quanti thread per blocco?</param>
/// <param name="threadPerBlock">Strategia di accesso alla memoria: coalescente o stride?</param>
/// <return>Il numero di occorrenze osservate</return>
int countOccurrenciesV2(int* v, int n, int x, int nBlocks, int threadsPerBlock, MemoryAccessStrategy memoryAccessStrategy) {

	// allocazione memoria su device
	// - n spazi per il vettore di input
	// - 1 spazio finale per il risultato
	int* vDevice;
	HANDLE_ERROR(cudaMalloc((void**)&vDevice, sizeof(int) * (n + 1)));
	int* resultDevice = &vDevice[n];

	// copia vettore di input nel device + inizializzazione a 0 spazio risultato
	HANDLE_ERROR(cudaMemcpy(vDevice, v, sizeof(int) * n, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(resultDevice, 0, sizeof(int)));

	// (non serve calcolare il numero di blocchi)

	if (memoryAccessStrategy == COALESCENCE) {
		countOccurrenciesV2_coalescent_kernel << <nBlocks, threadsPerBlock >> > (vDevice, n, x, resultDevice, nBlocks * threadsPerBlock);
	}
	else {
		countOccurrenciesV2_stride_kernel << <nBlocks, threadsPerBlock >> > (vDevice, n, x, resultDevice, nBlocks * threadsPerBlock);
	}
	cudaDeviceSynchronize();
	checkKernelError("kernel v1");

	// copia risultato su host
	int result;
	HANDLE_ERROR(cudaMemcpy(&result, resultDevice, sizeof(int), cudaMemcpyDeviceToHost));

	// liberazione memoria device
	HANDLE_ERROR(cudaFree(vDevice));

	return result;
}


