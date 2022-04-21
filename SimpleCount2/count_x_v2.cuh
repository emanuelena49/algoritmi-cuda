# ifndef COUNT_OCCURENCES_V2
# define COUNT_OCCURENCES_V2

/// <summary>
/// Strategia di accesso alla memoria in caso di size > nThreads
/// 
/// * può essere coalescente (ogni thread legge una sequenza consecutiva di dati)
/// * oppure stride (ogni thread legge una posizione che in %nThreads è sempre == tid)
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
int countOccurrenciesV2(int* v, int n, int x, int nBlocks, int threadsPerBlock, MemoryAccessStrategy memoryAccessStrategy);

#endif // !COUNT_OCCURENCES_V2
