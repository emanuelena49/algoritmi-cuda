#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

#include <stdio.h>

# define MAX_THREADS_PER_BLOCK 1024

# define DEBUG_OCC_MATRIX_CALC false
# define DEBUG_SUM_CALC true

__global__ void occurrences_matrix(int* dataDevice, int* occurrencesMatrixDevice, int dataLength, int n) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	// printf("%i = %i * %i + %i\n", threadId, blockDim.x, blockIdx.x, threadIdx.x);

	if (threadId < dataLength) {
		int observedValue = dataDevice[threadId];
		int destination = threadId * n + observedValue;
		occurrencesMatrixDevice[destination] = 1;

		if (DEBUG_OCC_MATRIX_CALC) printf("%i=%i*%i+%i\tin:%i=%i*%i+%i\tscrivo:%i\n",
			threadId, blockDim.x, blockIdx.x, threadIdx.x, 
			destination, threadId, n, observedValue, occurrencesMatrixDevice[destination]);
	}
}

__global__ void sum_pair_vector(int* occurrencesMatrixDevice, int n, int dataLength, int step) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < dataLength) {

		int startFirst = (2 * threadId) * n;
		int startSecond = (2 * threadId + 1) * n;

		int j = startSecond;
		for (int i = startFirst; i < startFirst + n; i++) {

			occurrencesMatrixDevice[i] += occurrencesMatrixDevice[j];
		}
	}
}

__global__ void sum_pair(int* occurrencesMatrixDevice, int nRealThreads, int offset) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < nRealThreads) {

		occurrencesMatrixDevice[threadId] += occurrencesMatrixDevice[threadId + offset];
	}
}


/// <summary>
/// Dato un vettore di interi, i cui valori sono in 0..n-1, 
/// calcola il numero di occorrenze ci ciascuna chiave.
/// </summary>
/// <param name="data">
///		vettore di dataLength elementi, interi in 0..n-1)
/// </param>
/// <param name="occurrences">
///		vettore dove verranno inserite le occorrenze. Ci si aspetta sia 
///		già inizializzato con tutti i valori a 0.
/// </param>
/// <param name="dataLength">Dimensione del vettore di input</param>
/// <param name="n"></param>
void histogram_parallel(int* data, int* occurrences, int dataLength, int n) { 
	
	// copia dati sul device
	int *dataDevice;
	cudaMalloc((void**) &dataDevice, dataLength * sizeof(int));
	cudaMemcpy(dataDevice, data, dataLength * sizeof(int), cudaMemcpyHostToDevice);

	// matrice dataLength * n dei risultati di ogni singolo thread
	// in pratica, è una lista di lunghezza dataLength * n dove 
	// ogni blocco di dimensione n, corrisponde a quello dove andrà ad 
	// operare 1 thread per la frequenza osservata su un dato.
	int *occurrencesMatrixDevice;
	cudaMalloc((void**) &occurrencesMatrixDevice, dataLength * n * sizeof(int));
	cudaMemset(occurrencesMatrixDevice, 0, dataLength * n * sizeof(int));

	// lancio di dataLength threads, ciascuno di essi (i) analizza
	// un dato unico (dataDevice[i]) e incrementa il suo valore corrispettivo
	// nella lista occurrencesMatrixDevice[i*n + dataDevice[i]]

	int nBlocks = dataLength / MAX_THREADS_PER_BLOCK;
	if (dataLength % MAX_THREADS_PER_BLOCK != 0) {
		nBlocks++;
	}

	int nThreadsPerBlock;
	if (nBlocks > 1) {
		nThreadsPerBlock = MAX_THREADS_PER_BLOCK;
	}
	else {
		nThreadsPerBlock = dataLength;
	}

	occurrences_matrix<<<nBlocks, nThreadsPerBlock >>>(dataDevice, occurrencesMatrixDevice, dataLength, n);
	cudaDeviceSynchronize();

	// output: sequenza di dataLength liste di dimensione n, ciascuna 
	// contenente la frequenza osservata da 1 thread

	cudaFree(dataDevice);

	int* o = (int*) malloc(n * dataLength * sizeof(int));

	if (DEBUG_SUM_CALC) {
		cudaMemcpy(o, occurrencesMatrixDevice, n * dataLength * sizeof(int), cudaMemcpyDeviceToHost);
		printf("\n\nRisultati del calcolo della matrice:\n");
		for (int i = 0; i < dataLength; i++) {

			int cumsum = 0;

			printf("%i)\t", i);

			for (int j = 0; j < n; j++) {
				printf("%i ", o[i * n + j]);
				cumsum += o[i * n + j];
			}
			printf("\tcumsum: %i\n", cumsum);
		}
	}

	// ora aggrego sommando a due a due le liste, fintanto 
	// che non me ne rimane una sola (l'output finale)

	// ogni thread si occupa di sommare un singolo termine

	int remainingData = dataLength*n;

	while (remainingData > n)
	{
		// lancio di offset thread, 
		// che sommano offset/n liste 
		
		int offset = remainingData / 2;
		
		nBlocks = offset / MAX_THREADS_PER_BLOCK;
		if (offset % MAX_THREADS_PER_BLOCK != 0) {
			nBlocks++;
		}

		if (nBlocks > 1) {
			nThreadsPerBlock = MAX_THREADS_PER_BLOCK;
		}
		else {
			nThreadsPerBlock = offset;
		}

		sum_pair <<<nBlocks, nThreadsPerBlock >>> (occurrencesMatrixDevice, offset, offset);
		cudaDeviceSynchronize();

		// output: (offset/n)/2 liste ancora da operare

		if (DEBUG_SUM_CALC) {
			cudaMemcpy(o, occurrencesMatrixDevice, n * offset * sizeof(int), cudaMemcpyDeviceToHost);
			printf("\n\nOffset:%i, lanciati n.threads:%i\n", offset, nBlocks*nThreadsPerBlock);
			for (int i = 0; i < offset / n; i++) {

				int cumsum = 0;

				printf("%i)\t", i);

				for (int j = 0; j < n; j++) {
					printf("%i ", o[i * n + j]);
					cumsum += o[i * n + j];
				}
				printf("\tcumsum: %i\n", cumsum);
			}
		}

		remainingData = offset;
	}

	// output: nelle prime dataLength posizioni di occurrencesMatrixDevice ora
	// si trova la soluzione

	cudaMemcpy(occurrences, occurrencesMatrixDevice, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(occurrencesMatrixDevice);
}



