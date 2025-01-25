/*------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iohb.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#define IMAX 15000
#define ERRO 1e-6

/*-----------------------------------------------------------------*/
void  le_matriz( char *arquivo, int *M, int *N, int *naozeros, int **colptr, int **linhas, double **valores){
	  
	int retorno, nrhs;
	char *tipo; 
	retorno = readHB_info(arquivo, M, N, naozeros, &tipo, &nrhs);
    
	if (retorno == 0){
	        printf("Erro ao ler as informaçõess da matriz!\n");
		exit(-1);
	}

	printf("Linhas: %d \t Colunas: %d \t Não Zeros: %d \n\n", *M, *N, *naozeros);
	
	*valores = (double *) malloc (*naozeros * sizeof(double));
	*linhas  = (int *) malloc (*naozeros * sizeof(int));
	*colptr  = (int *) malloc ((*N+1) * sizeof(int));

	retorno = readHB_mat_double(arquivo, *colptr, *linhas, *valores);

	     
	if (retorno == 0){
        	printf("Erro ao ler os valores da matriz!\n");
        	exit(-1);
    	}
}
/*-----------------------------------------------------------------*/
void escreve_matriz(int M, int N, int naozeros, int *colptr, int *linhas, double *valores){
	int i;
	
	printf("VALORES:\n");
	for(i=0; i<naozeros; i++){
		printf("%f ", valores[i]);
	}
	printf("\n\n");

	printf("LINHAS:\n");
	for(i=0; i<naozeros; i++){
		printf("%d ", linhas[i]);
	}
	printf("\n\n");

	printf("PTR:\n");
	for(i=0; i<M+1; i++){
		printf("%d ", colptr[i]);
	}
	printf("\n");

}

void imprime_matriz_esparsa(int M, int N, int *colptr, int *linhas, double *valores) {
    printf("Matriz esparsa:\n");
    for (int j = 0; j < N; j++) {
        for (int k = colptr[j] - 1; k < colptr[j + 1] - 1; k++) {
            int linha = linhas[k] - 1;  
            printf("A[%d, %d] = %f\n", linha + 1, j + 1, valores[k]);  
        }
    }
    printf("\n");
}


void transpose_esparsa(int* colptr, int* linhas, double* valores, int** colptrT, int** linhasT, double** valoresT, int M, int N, int naozeros) {
    *colptrT = (int*) malloc((N + 1) * sizeof(int));
    *linhasT = (int*) malloc(naozeros * sizeof(int));
    *valoresT = (double*) malloc(naozeros * sizeof(double));

    int* row_counts = (int*) calloc(N, sizeof(int));
    for (int k = 0; k < naozeros; k++) {
        row_counts[linhas[k] - 1]++;
    }

    (*colptrT)[0] = 1;
    for (int i = 1; i <= N; i++) {
        (*colptrT)[i] = (*colptrT)[i - 1] + row_counts[i - 1];
    }

    int* current_position = (int*) calloc(N, sizeof(int)); 
    for (int j = 0; j < N; j++) {
        for (int k = colptr[j] - 1; k < colptr[j + 1] - 1; k++) {
            int i = linhas[k] - 1;
            int pos = (*colptrT)[i] - 1 + current_position[i]++;
            (*linhasT)[pos] = j + 1;
            (*valoresT)[pos] = valores[k];
        }
    }

    free(row_counts);
    free(current_position);
}

void matmul_esparsa(int* colptr, int* linhas, double* valores, double* x, double* result, int M, int N, int nth) {
    // Inicializa o vetor result com zeros
    omp_set_num_threads(nth);  // Define o número de threads
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        result[i] = 0.0;
    }

    // Realiza a multiplicação da matriz esparsa com o vetor
    omp_set_num_threads(nth);  // Define o número de threads para a segunda região paralela
    #pragma omp parallel for
    for (int j = 0; j < N; j++) {
        for (int k = colptr[j] - 1; k < colptr[j + 1] - 1; k++) {
            int i = linhas[k] - 1;
            #pragma omp atomic
            result[i] += valores[k] * x[j];
        }
    }
}

double dot_product(double* a, double* b, int m, int nth) {
    double result = 0.0;
    omp_set_num_threads(nth);
    #pragma omp parallel for reduction(+:result)  
      for (int i = 0; i < m; i++) {
          result += a[i] * b[i];
      }
    return result;
}

void vector_add(double* a, double* b, double* result, double alpha, int m, int nth) {
    omp_set_num_threads(nth);
    #pragma omp parallel for
      for (int i = 0; i < m; i++) {
          result[i] = a[i] + alpha * b[i];
      }
}

void scalar_multiply(double* a, double scalar, double* result, int m, int nth) {
    omp_set_num_threads(nth);
    #pragma omp parallel for
      for (int i = 0; i < m; i++) {
          result[i] = scalar * a[i];
      }
}
void vector_subtract(double* a, double* b, double* result, int m, int nth) {
    omp_set_num_threads(nth);
    #pragma omp parallel for
      for (int i = 0; i < m; i++) {
          result[i] = a[i] - b[i];
      }
}

void bicg_esparsa(int* colptr, int* linhas, double* valores, int* colptrT, int* linhasT, double* valoresT, double* b, double* x, int m, int n, int* niter, int nth) {
    double *r = (double*)malloc(n * sizeof(double));
    double *r2 = (double*)malloc(n * sizeof(double));
    double *p = (double*)malloc(n * sizeof(double));
    double *p2 = (double*)malloc(n * sizeof(double));
    double *v = (double*)malloc(n * sizeof(double));
    double *temp = (double*)malloc(n * sizeof(double));
    double *temp2 = (double*)malloc(n * sizeof(double));

    matmul_esparsa(colptr, linhas, valores, x, r, m, n, nth);  // r = A * x
    
    for (int i = 0; i < m; i++) {
        r[i] = b[i] - r[i];  
        r2[i] = r[i];         
        p[i] = 0.0;           
        p2[i] = 0.0;        
    }

    double rho = 1.0;
    double rho0, beta, alpha;

    *niter = 1;

    while (*niter < IMAX) {
        rho0 = rho;
        rho = dot_product(r2, r, m, nth);
        beta = rho / rho0;
        
        vector_add(r, p, temp, beta, m, nth);
        omp_set_num_threads(nth);
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            p[i] = temp[i];
        }

        vector_add(r2, p2, temp, beta, m, nth);
        
        omp_set_num_threads(nth);
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            p2[i] = temp[i];
        }

        matmul_esparsa(colptr, linhas, valores, p, v, m, n, nth);

        alpha = rho / dot_product(p2, v, m, nth); 
        
        scalar_multiply(p, alpha, temp, m, nth);
        vector_add(x, temp, temp, 1.0, m, nth);
        
        omp_set_num_threads(nth);
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            x[i] = temp[i];
        }

        if (dot_product(r, r, m, nth) < ERRO * ERRO) {
            break;
        }

        if (*niter < IMAX) {
            scalar_multiply(v, alpha, temp, m, nth);
            vector_subtract(r, temp, temp, m, nth);
            
            omp_set_num_threads(nth);
            #pragma omp parallel for
            for (int i = 0; i < m; i++) {
                r[i] = temp[i];
            }

            matmul_esparsa(colptrT, linhasT, valoresT, p2, v, m, n, nth);
            scalar_multiply(v, alpha, temp, m, nth);
            vector_subtract(r2, temp, temp, m, nth);
            
            omp_set_num_threads(nth);
            #pragma omp parallel for
            for (int i = 0; i < m; i++) {
                r2[i] = temp[i];
            }
        }

        (*niter)++;
    }

    free(r);
    free(r2);
    free(p);
    free(p2);
    free(v);
    free(temp);
    free(temp2);
}

int main(int argc, char **argv) { 
    double *valores = NULL, *valoresT = NULL;
    int *linhas = NULL, *colptr = NULL, *linhasT = NULL, *colptrT = NULL;
    int M, N, naozeros;

    if (argc != 3) {
        printf("%s <Arquivo HB> <N_Threads>\n", argv[0]);
        exit(-1);
    }

    le_matriz(argv[1], &M, &N, &naozeros, &colptr, &linhas, &valores);   

    escreve_matriz(M, N, naozeros, colptr, linhas, valores);

    imprime_matriz_esparsa(M, N, colptr, linhas, valores);
    
    transpose_esparsa(colptr, linhas, valores, &colptrT, &linhasT, &valoresT, M, N, naozeros);
    
    int nth = atol(argv[2]);
    
    double* b = (double*) malloc(M * sizeof(double));
    double* x = (double*) malloc(M * sizeof(double));
    int niter;

    for (int i = 0; i < M; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }
  
    double ti = omp_get_wtime();

    bicg_esparsa(colptr, linhas, valores, colptrT, linhasT, valoresT, b, x, M, N, &niter, nth);
    
    double tf = omp_get_wtime();

    printf("Solução x:\n");
    for (int i = 0; i < M; i++) {
        printf("%.10lf\n", x[i]);
    }

    double* b_calculado = (double*)malloc(M * sizeof(double));
    matmul_esparsa(colptr, linhas, valores, x, b_calculado, M, N, nth);
    
    printf("Número de iterações: %d\n", niter);
    printf("Verificação (b = A * x):\n");
    for (int i = 0; i < M; i++) {
        printf("%lf\n", b_calculado[i]);
    }
    
    printf("Tempo: %f\n", tf-ti);
    free(valores);
    free(linhas);
    free(colptr);
    free(valoresT);
    free(linhasT);
    free(colptrT);
    free(b);
    free(x);
    free(b_calculado);

    return 0;
}
/*------------------------------------------------------*/
