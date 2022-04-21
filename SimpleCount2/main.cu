#include <time.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <iostream>
#include "count_x_v1.cuh"
#include "count_x_v2.cuh"


/// <summary>
/// Conta le occorrenze di x nel vettore v
/// </summary>
/// <param name="v"></param>
/// <param name="n"></param>
/// <param name="x"></param>
/// <return>Il numero di occorrenze osservate</return>
int countOccurrenciesV0(int* v, int n, int x) {

	int xOccurrencies = 0;

	for (size_t i = 0; i < n; i++)
	{
		if (v[i] == x) {
			xOccurrencies++;
		}
	}

	return xOccurrencies;
}

/// <summary>
/// Riempi lo spazio v (di dimensione n) con valori casuali
/// appartenenti all'intervallo [startValue, endValue]
/// </summary>
/// <param name="v"></param>
/// <param name="n"></param>
/// <param name="startValue"></param>
/// <param name="endValue"></param>
void randomArray(int* v, int n, int startValue, int endValue) {

	time_t t;
	srand((unsigned)time(&t));

	for (size_t i = 0; i < n; i++)
	{
		v[i] = (rand() % (endValue - startValue)) + startValue;
	}
}

int main() {

	int n, start, end, x, nBlocks, threadsPerBlocks;
	bool defaultOptions = false;

	std::cout << "Inserisci la dimensione del vettore [-1 per usare opzioni di default]:\t";
	std::cin >> n;
	// n = 10000;
	if (n<=0) {
		n = 10000; start = 0; end = 10; x = 5; nBlocks = 8; threadsPerBlocks = 128;
		defaultOptions = true;

		std::cout << "Caricamento opzioni di default: vettore casuale di " << n
			<< " elementi, compresi in {" << start << ".." << end << "-1}. Conta occorrenze di "
			<< x << ". Quanto posso scegliere, uso " << nBlocks << " blocchi da "
			<< threadsPerBlocks << "thread ciascuno.";
	}
	else {
		std::cout << "\nInserisci l'intervallo dei valori del vettore.\nValore minimo:\t";
		std::cin >> start;
		// start = 0;
		std::cout << "Valore massimo:\t";
		std::cin >> end;
		// end = 100;

		std::cout << "\nInserisci valore da cercare:\t";
		std::cin >> x;
		// x = 0;

		std::cout << "\nInserisci numero di blocchi e thread per blocco da usare nella computazione parallela V2.\nNumero di Blocchi:\t";
		std::cin >> nBlocks;
		std::cout << "Thread per blocco:\t";
		std::cin >> threadsPerBlocks;
	}

	int* v = (int*)malloc(sizeof(int) * n);
	randomArray(v, n, start, end);

	printf("\nNumero di occorrenze (seriale):%i\t", countOccurrenciesV0(v, n, x));
	printf("\nNumero di occorrenze (parallelo v1):%i\t", countOccurrenciesV1(v, n, x));
	printf("\nNumero di occorrenze (parallelo v2):%i\t", countOccurrenciesV2(v, n, x, nBlocks, threadsPerBlocks, COALESCENCE));

	return 0;
}
