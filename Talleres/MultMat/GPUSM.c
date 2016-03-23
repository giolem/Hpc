#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h> 
#include <cuda.h>

#define rows 64
#define cols 64

#define TILE_WIDTH 32

__global__ void multi_MatTiled(int *m1_a, int *m2_a, int *m3_a){
    
    __shared__ int m1a_ds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int m2a_ds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = by * TILE_WIDTH + ty;
    int row = bx * TILE_WIDTH + tx;

    float acu = 0;

    for(int m = 0; m < cols / TILE_WIDTH; ++m){
    	m1a_ds[ty][tx] = m1_a[col*cols + m*TILE_WIDTH + tx];
    	m2a_ds[ty][tx] = m2_a[(m*TILE_WIDTH + ty) * cols + row];
    	__syncthreads();

    	for(int k = 0; k < TILE_WIDTH; ++k){
    		acu += m1a_ds[ty][k] * m2a_ds[k][tx];	    
    	}
    	__syncthreads();
    }
    m3_a[col*cols+row] = acu;
}


__global__ void multi_matricesDevice(int *m1_a, int *m2_a, int *m3_a){
	
  int i,j,acu;

  j = blockIdx.y*blockDim.y+threadIdx.y;
  i = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(i<cols && j<rows){
    	acu=0;
	   for(int k=0;k<cols;k++){
		 acu+= m1_a[j*cols+k] * m2_a[k*cols+i];
		}
		m3_a[i*cols+j]=acu;
	}
}

void multi_matricesHost(int *mat1, int *mat2, int *mat3){
	int i,j,k,acu;
	for (i = 0; i < rows; i++){
	     for (k = 0; k < cols; k++){
				acu=0;
					for (j = 0; j < rows; j++){
		    		acu += mat1[i*cols+j] * mat2[j*cols+k];
		   			 mat3[i*cols+k] = acu; 
					}
				}
		}
}

void llenar(int *mat){
	
	
	for (int i = 0; i < rows; i++){
	   for (int j = 0; j < cols; j++){
				mat[i*cols+j]=(2+i);		
		}
	}
}

void mostrar(int *mat){

	for (int i = 0; i < rows; i++){		
		for (int j = 0; j < cols; j++){
			printf("%d ",mat[i*cols+j]);
		}
		printf("\n");		
	}
}



int main(){
	int *m1,*m2,*m3;
	int *m1_a,*m2_a,*m3_a;
	int sizem = rows*cols*(sizeof(int));
	
	clock_t start_t, end_t, start_t_GPU, end_t_GPU;
	double total_t, total_t_GPU;	
	
	 cudaError_t error = cudaSuccess;

	/*---------------Secuencial-----------------------------*/
  
	m1= (int *)malloc(sizem);		
	m2= (int *)malloc(sizem);
	m3= (int *)malloc(sizem);			
  
  llenar(m1);
	llenar(m2);

	start_t=clock();	
		
	multi_matricesHost(m1,m2,m3);
  
	//mostrar(m3);
	end_t= clock();	

  	total_t= ((double)(end_t - start_t))/CLOCKS_PER_SEC;

  	printf("Tiempo Secuencial: %f \n",total_t);
  
  
  
  
  	/*---------------------------------------------------------------*/


  	/*-------------Algoritmo paralelo-----------------------------*/
  	
  	//Asignación de memoria
	error=cudaMalloc((void **)&m1_a,(sizem));
	if(error != cudaSuccess){
        	printf("Ocurrio un error reservando memoria para la matriz 2");
        	exit(0);
    	}
    	
	error=cudaMalloc((void **)&m2_a,(sizem));
	if(error != cudaSuccess){
        	printf("Ocurrio un error reservando memoria para la matriz 2");
        	exit(0);
    	}
    	
	error=cudaMalloc((void **)&m3_a,(sizem));
	if(error != cudaSuccess){
        	printf("Ocurrio un error reservando memoria para la matriz 2");
        	exit(0);
    	}
    	
	start_t_GPU=clock();
	
	//Errores de copia si los hay
  
	error=cudaMemcpy(m1_a,m1,sizem,cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
        	printf("Error copiando datos de la matriz m1 a la matriz m1_a");
        	exit(0);
    	}
    	
	error=cudaMemcpy(m2_a,m2,sizem,cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
        	printf("Error copiando datos de la matriz m2 a la matriz m2_a");
        	exit(0);
    	}

	int blockSize=32;
	dim3 dimBlock(blockSize,blockSize,1);
  	dim3 dimGrid(ceil(cols/float(blockSize)),ceil(cols/float(blockSize)),1);

	
	multi_MatTiled<<<dimGrid,dimBlock>>>(m1_a,m2_a,m3_a);
	cudaMemcpy(m3,m3_a,sizem,cudaMemcpyDeviceToHost);	

	end_t_GPU=clock();
	/*---------------------------------------------------------------*/

	/*imprimir(m3);
	printf("\n");*/

	total_t_GPU= ((double)(end_t_GPU - start_t_GPU))/CLOCKS_PER_SEC;
	printf("Tiempo Paralelo: %f",total_t_GPU);
	
	cudaFree(m1_a);
	cudaFree(m2_a);
	cudaFree(m3_a);
	free(m1);
	free(m2);
	free(m3);

	return 0;
	
}
