#pragma once

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
void histogram(int* data, int* occurrences, int dataLength) {

	for (size_t i = 0; i < dataLength; i++)
	{
		occurrences[data[i]]++;
		// printf("%i", occurrences[data[i]]);
	}
}