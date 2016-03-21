#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>


#define rows 50
#define cols 50

//m1[i][j] => A [i*w+j];



void mostrar(int *mat){

	for (int i = 0; i < rows; i++){		
		for (int j = 0; j < cols; j++){
			printf("%d ",mat[i*cols+j]);
		}
		printf("\n");		
	}
}



void llenar(int *mat){
	
	
	for (int i = 0; i < rows; i++){
	   for (int j = 0; j < cols; j++){
				mat[i*cols+j]=(2+i);		
		}
	}
}

void multi(int *mat1, int *mat2, int *mat3){	

	    for (int i = 0; i < rows; i++){			
	           for (int j = 0; j < cols; j++){
				      int acu=0;				      
					for (int k = 0; k < cols; k++){
		    		acu += mat1[i*cols+k] * mat2[k*rows+j];		   			 
								}
					mat3[i*cols+j] = acu; 
					}
			}
	}




int main(){
	
	int *m1,*m2,*m3;
  
    clock_t start_t, end_t;
  
	double total_t;
	
	

	m1= (int *)malloc(rows*cols*sizeof(int *));		
	m2= (int *)malloc(rows*cols*sizeof(int *));
	m3= (int *)malloc(rows*cols*sizeof(int *));
	
	llenar(m1);
	llenar(m2);
	
	start_t = clock();
	multi(m1,m2,m3);
	
	end_t= clock();
  /*
	printf("Matriz 1\n");
	mostrar(m1);

	printf("Matriz 2\n");
	mostrar(m2);
  
	printf("Multiplicación\n");
	mostrar(m3);
	*/
	
	
	total_t= ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
	
	printf("%f \n \n \n",total_t );
	
	free(m1);
	free(m2);
	free(m3);
	
	return 0;
	
}
