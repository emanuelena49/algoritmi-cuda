
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include "count_x_v1.cpp"


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
	srand((unsigned) time(&t));
	
	for (size_t i = 0; i < n; i++)
	{
		v[i] = (rand() % (endValue - startValue)) + startValue;
	}
}

int main() {

	int n, start, end, x;
	
	printf("Inserisci la dimensione del vettore:\t");
	// cin >> n;
	n = 10000;

	printf("Inserisci l'intervallo dei valori del vettore:\nValore minimo:\t");
	// cin >> start;
	start = 0;
	printf("Valore massimo:\t");
	// cin >> end;
	end = 100;

	int* v = (int*) malloc(sizeof(int) * n);
	randomArray(v, n, start, end);

	printf("Inserisci valore da cercare:\t");
	// cin >> x;
	x = 0;

	printf("\nNumero di occorrenze (seriale):%i\t", countOccurrenciesV0(v, n, x));
	printf("\nNumero di occorrenze (parallelo):%i\t", countOccurrenciesV1(v, n, x));

	return 0;
}