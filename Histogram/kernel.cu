
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdio.h>
#include <memory>
#include <time.h>

#include "histogram.cpp"
#include "histogram_parallel_naive.cpp"

int main() {

	int length = 128;
	int n = 10;

	int* data = (int*) malloc(length*sizeof(int));

	srand(time(NULL));
	
	for (size_t i = 0; i < length; i++)
	{
		data[i] = rand() % n;
		printf("%i, ", data[i]);
	}

	printf("\n\n");

	int* result = (int*) calloc(n, sizeof(int));
	histogram(data, result, length);

	int* resultParallel = (int*)malloc(n * sizeof(int));
	histogram_parallel(data, resultParallel, length, n);

	printf("\n\nrisultati\tseriale\tparallelo");
	for (size_t i = 0; i < n; i++)
	{
		printf("%i:\t%i\t\t%i\n", i, result[i], resultParallel[i]);
	}
}