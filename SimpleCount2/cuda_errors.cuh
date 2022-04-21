#ifndef CUDA_ERRORS_HANDLING
#define CUDA_ERRORS_HANDLING

#include <driver_types.h>

/// <summary>
/// Funzione per la gestione d'errore. Nel caso in cui il codice d'output passato 
/// sia un errore, stampa a console un messaggio d'errore (indicando file e riga)
/// </summary>
/// <param name="cudaFunctionOutput">
///	L'output di una funzione CUDA, può essere un codice d'errore o uno di successo
/// </param>
/// <param name="file">Il file dove è accaduto l'errore</param>
/// <param name="line">La linea di codice che ha causato l'errore</param>
void HandleError(cudaError_t cudaFunctionOutput, const char* file, int line);

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


// NOTA: poi potrei avere macro diverse per dev e prod 
// (rimuovendo controlli, introducendo gest. degli errori, ecc.)

// NOTA: la macro non può essere usata per l'esecuzione del kerner
// (dovrei andarmi a controllare i varlori di rirorno di ogni singolo thread...)


/// <summary>
/// Funzione per controllare che l'ultima esecuzione del kernel sia
/// andata a buon fine. Chiamare al termine della chiamata al kernel
/// (dopo previa sincronizzazione)
/// </summary>
/// <param name="msg">Messaggio libero per permettermi di capire a che riga sono</param>
void checkKernelError(const char* msg);

//NOTA: occhio, non è detto che vengano colti i classici errori di accesso alla memoria


#endif // !CUDA_ERRORS_HANDLING