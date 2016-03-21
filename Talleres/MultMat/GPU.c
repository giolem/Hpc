#include<cuda.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<unistd.h> 
#include<cuda.h>

#define rows 50
#define cols 50


__global__ void multiplicar(int *m1_a, int *m2_a, int *m3_a){
	
  	int i,j;

  	j = blockIdx.y*blockDim.y+threadIdx.y;
	  i = blockIdx.x*blockDim.x+threadIdx.x;
	
		if(i<cols && j<rows){
    		int acu=0;
	  	 for(int k=0;k<cols;k++){
			 acu+= m1_a[i*cols+k] * m2_a[k*cols+j];
			}
			m3_a[i*cols+j]=acu;
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
  
  //CPU memory and GPU 
  
	int *m1,*m2,*m3,*m1_g,*m2_g,*m3_g;
	int sizem = rows*cols*(sizeof(int));
	
	clock_t start_t,end_t;
	double total_t;

	m1= (int *)malloc(sizem);		
	m2= (int *)malloc(sizem);
	m3= (int *)malloc(sizem);

	llenar(m1);
	llenar(m2);
  
	start_t=clock();
  
	cudaMalloc((void **)&m1_g,(sizem));
	cudaMalloc((void **)&m2_g,(sizem));
	cudaMalloc((void **)&m3_g,(sizem));

	cudaMemcpy(m1_g,m1,sizem,cudaMemcpyHostToDevice);
	cudaMemcpy(m2_g,m2,sizem,cudaMemcpyHostToDevice);

	int blockSize=32;
	dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(cols/float(blockSize)),ceil(cols/float(blockSize)),1);

 	multiplicar<<<dimGrid,dimBlock>>>(m1_g,m2_g,m3_g);

	cudaMemcpy(m3,m3_g,sizem,cudaMemcpyDeviceToHost);

	//multiplicar(m1_g,m2_g,m3_g);

	end_t= clock();
  
  //printf("Matriz 1\n");
 	//mostrar(m1);  	
  //printf("Matriz 2\n");
  //mostrar(m2);
  //printf("Multiplicación \n");
	//mostrar(m3);
  
  
	//printf("\n");
  
	total_t= ((double)(end_t - start_t))/CLOCKS_PER_SEC;
	printf("Tiempo: \n %f",total_t);
	
	cudaFree(m1_g);
	cudaFree(m2_g);
	cudaFree(m3_g);
  
	free(m1);
	free(m2);
	free(m3);

	return 0;
	
}
