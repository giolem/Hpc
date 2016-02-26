#include <stdio.h>
#include <malloc.h>
#include <unistd.h> 
#include <time.h>


#define N 100

void llenarVector(int *a){

	for(int i=0; i<N; i++){
		
		a[i]=i;
		
	}
	
}


void sumarVector(int *a, int *b, int *c){
	
	for(int i=0; i<N; i++){
	
		c[i]=a[i]+b[i];
	
	}
	
}


void print(int *a){
	
	for(int i=0; i<N; i++){
	
		printf("%d\n",a[i]);
	
	}

}

int main(){

clock_t start = clock();  
	

	
	int *a = (int*)malloc(sizeof(int)*N) ;
	int *b = (int*)malloc(sizeof(int)*N);
	int *c = (int*)malloc(sizeof(int)*N);
	
	llenarVector(a);
	llenarVector(b);
	
	sumarVector(a,b,c);
	
	print(c);

	printf("Tiempo transcurrido: %f", ((double)clock() - start) / CLOCKS_PER_SEC);  
   
	
}
   

