//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <memory>
//#include <stdio.h>
//
//
//__global__ void frequences_on_segments(int* data, int* output, int) {
//
//}
//
//void histogram_parallel_v1(int* data, int* output, int m, int symbolsPerThread) {
//	
//	// data:	(puntatore a) vettore di lunghezza m, contiene valori interi in {0..n}
//	// output:	(puntatore a) area di memoria vuota di dim. n*sizeof(int)
//
//	// def. della prima potenza di due più >= m 
//	int m2 = pow(2, ceil(log2((double) m)));
//
//	// -------------------------------------------------
//	// def. di un po' di parametri
//
//	// thread a disposizione
//	const int maxThreadsP1 = 1024 * 16;
//	// quanti simboli (celle del vettore) deve processare un thread?
//	const int symbolsPerThread = m2 < maxThreadsP1 ? 1 : m2/maxThreadsP1;
//	// quanti thread devo lanciare?
//	const int nThreadsP1 = m2 / symbolsPerThread;
//	// e in quanti blocchi?
//	const int threadsPerBlockP1 = 1024;
//	const int nBlocksP1 = ceil(((double) m2)/((double) threadsPerBlockP1));
//
//	// -------------------------------------------------
//	// copia dei dati sul device + allocazione spazio 
//	// per l'output (non aggregato)
//
//	int* dataDevice;
//	cudaMalloc((void**)&dataDevice, m * sizeof(int));
//	cudaMemcpy(dataDevice, data, m * sizeof(int), cudaMemcpyHostToDevice);
//
//	int* outputDevice;
//	cudaMalloc((void**)&outputDevice, nThreadsP1 * n * sizeof(int));
//	cudaMemset(outputDevice, nThreadsP1 * n * sizeof(int), 0);
//
//	// outputDevice: sequenza di sequenze di lungh n, inizializzata a 0 
//	// (1 seq. per thread => nThreadsP1 => lunghezza totale=nThreadsP1*n)
//
//	// ------------------------------------------------
//	// lancio prima procedura P1: calcolo delle 
//	// frequenze osservate iniziali
//
//
//
//
//
//
//
//}