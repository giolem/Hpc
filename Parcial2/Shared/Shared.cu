#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0
#define TILE_SIZE 32

#define MASK_WIDTH 3

__constant__ char M[MASK_WIDTH*MASK_WIDTH];
__constant__ char d_Mt[MASK_WIDTH*MASK_WIDTH];



using namespace cv;

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}


__global__ void sobelX(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){
        	
        	
    __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];
    
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+MASK_WIDTH-1), destX = dest % (TILE_SIZE+MASK_WIDTH-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    
    
        
    __syncthreads();

    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(accum);
    __syncthreads();
}


__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}


__global__ void sobelY(unsigned char *imageInput, int width, int height, unsigned int maskWidth, unsigned char *ImageOutput){
	__shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];			//se establecen la submatriz y queda en memoria compartida
																																										//el tamaño del array en memoria global debe ser mas largo que el vector normal para darle espacio a los elementos de la izquierda, centro y derecha en total es TILE_SIZE + MASK_WIDTH - 1
    int n = maskWidth/2;
    
    //------Cargar los elementos de la matriz de la matriz de entrada en memoria compartida------
    //Cargar elementos izquierda derecha
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x;
	int destY = dest / (TILE_SIZE+MASK_WIDTH-1);
	int destX = dest % (TILE_SIZE+MASK_WIDTH-1);
    int srcY = blockIdx.y * TILE_SIZE + destY - n;
	int srcX = blockIdx.x * TILE_SIZE + destX - n;
    int src = (srcY * width + srcX);
	
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)		//si srcY es negativo son elementos fantasmas, si srcX es negativo son elementos fantasmas
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;					//asigna en 0 los elementos fantasmas

    //Cargar elementos del centro
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1);
	destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
	
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();
    //------Termina de cargar los elementos de la matriz de la matriz de entrada en memoria compartida------

		//-----llenamos la matriz de salida
    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * d_Mt[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
       ImageOutput[(y * width + x)] = clamp(accum);
    __syncthreads();
    //-----terminamos de llenar la matriz de salida

}

__global__ void sobel(unsigned char *imageSobelX, unsigned char *imageSobelY, int width, int height, unsigned char *imageOutput){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

	if((row < height) && (col < width)){
		imageOutput[row* width+col] = __powf((__powf(imageSobelX[row*width+col],2) + __powf(imageSobelY[row*width+col],2)),0.5);
	}

}






int main(int argc, char **argv){
    cudaSetDevice(0);//GTX980
    
    clock_t startGPU, endGPU;
    double  gpu_time_used;
    char h_M[] = {1,0,-1,2,0,-2,1,0,-1};
    char h_Mt[] = {1,-1,1,0,0,0,-1,-2,-1};
    char* imageName = argv[1];
    unsigned char *dataRawImage, *d_dataRawImage,*h_dataRawImage, *d_imageOutput, *h_imageOutput, * h_SobelOutput, *d_sobelOutput, *d_SobelX, *d_SobelY;
    Mat image;
    image = imread(imageName, 1);

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;

	

    dataRawImage = (unsigned char*)malloc(size);
    h_imageOutput = (unsigned char *)malloc(sizeGray);
    h_dataRawImage = (unsigned char *)malloc(sizeGray);
    h_SobelOutput = (unsigned char *)malloc(sizeGray);
    
    
    
    
    
    cudaMalloc((void**)&d_sobelOutput,sizeGray);
    cudaMalloc((void**)&d_SobelX,size);
	  cudaMalloc((void**)&d_SobelY,sizeGray);
	  cudaMalloc((void**)&d_imageOutput,sizeGray);
	  cudaMalloc((void**)&d_dataRawImage,sizeGray);
    
    
    dataRawImage = image.data;

    
	startGPU = clock();

    cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M,h_M,sizeof(char)*MASK_WIDTH*MASK_WIDTH);
    cudaMemcpyToSymbol(d_Mt,h_Mt,sizeof(char)*MASK_WIDTH*MASK_WIDTH);

    int blockSize = TILE_SIZE;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    sobelX<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,MASK_WIDTH,d_sobelOutput);
    cudaMemcpy(h_imageOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);
    endGPU = clock();

    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    //start = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    //end = clock();

/*
	namedWindow(imageName, WINDOW_NORMAL);
   namedWindow("Gray Image CUDA", WINDOW_NORMAL);
   namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);
   imshow(imageName,image);
   imshow("Gray Image CUDA", gray_image);
   imshow("Sobel Image OpenCV",abs_grad_x);
   waitKey(0);*/


    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    
    //cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    
    printf("%.10f\n",gpu_time_used);

    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(M);
    cudaFree(d_sobelOutput);
    return 0;
}
