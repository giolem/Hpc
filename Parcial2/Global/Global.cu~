#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}



__global__ void sobelFilterX(unsigned char *imageInput, int width, int height, unsigned int maskWidth,\
        char *M,unsigned char *imageOutput){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}


__global__ void sobelFilter(unsigned char *imageInputX, unsigned char *imageInputY, int width, int height, unsigned char *imageOutput){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

	if((row < height) && (col < width)){
		imageOutput[row* width+col] = __powf((__powf(imageInputX[row*width+col],2) + __powf(imageInputY[row*width+col],2)),0.5);
	}

}


__global__ void sobelFilterY(unsigned char *imageInput, int width, int height, unsigned int maskWidth,\
        char *M,unsigned char *imageOutput){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}

__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}


int main(int argc, char **argv){
    
    clock_t startGPU, endGPU;
    double gpu_time_used;
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1}, *d_M;
    char h_Mt[] = {-1,-2,-1,0,0,0,1,2,1}, *d_Mt;
    char* imageName = argv[1];
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *h_sobelOutput,*d_sobelOutputx, *d_sobelOutputy, *d_sobelOutput;
    Mat image;
    image = imread(imageName, 1);

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;


    dataRawImage = (unsigned char*)malloc(size);
    cudaMalloc((void**)&d_dataRawImage,size);
    h_imageOutput = (unsigned char *)malloc(sizeGray);
    h_sobelOutput = (unsigned char *)malloc(sizeGray);
    cudaMalloc((void**)&d_sobelOutputx,sizeGray);
    cudaMalloc((void**)&d_sobelOutputy,sizeGray);
    cudaMalloc((void**)&d_sobelOutput,sizeGray);
   	cudaMalloc((void**)&d_imageOutput,sizeGray);
  	cudaMalloc((void**)&d_M,sizeof(char)*9);
  	  	cudaMalloc((void**)&d_Mt,sizeof(char)*9);
  	cudaMalloc((void**)&d_sobelOutput,sizeGray);
    dataRawImage = image.data;

    startGPU = clock();
    
		cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);
		
		cudaMemcpy(d_M,h_M,sizeof(char)*9, cudaMemcpyHostToDevice);
		
		cudaMemcpy(d_Mt,h_Mt,sizeof(char)*9, cudaMemcpyHostToDevice);
		
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    sobelFilterX<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_M,d_sobelOutputx);
    sobelFilterY<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_Mt,d_sobelOutputy);
    
    sobelFilter<<<dimGrid,dimBlock>>>(d_sobelOutputx,d_sobelOutputy,width,height,d_sobelOutput);
    
    cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);
    endGPU = clock();
    cudaMemcpy(h_sobelOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);
    endGPU = clock();
// aca
    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;
    
    Mat sobel_image;
    sobel_image.create(height,width,CV_8UC1);
    sobel_image.data = h_sobelOutput;
    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image CUDA", WINDOW_NORMAL);
    namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);
    imshow(imageName,image);
    waitKey(0);
    imshow("Gray Image CUDA", gray_image);
    waitKey(0);
    imshow("Sobel Image OpenCV",sobel_image);
    waitKey(0);

//aca
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    
   
    
    printf("%.10f;\n",gpu_time_used);

    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(d_M);
    cudaFree(d_sobelOutput);
    return 0;
}
