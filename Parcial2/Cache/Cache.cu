#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_WIDTH 3

__constant__ char M[MASK_WIDTH*MASK_WIDTH];
__constant__ char Mt[MASK_WIDTH*MASK_WIDTH];

using namespace cv;

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}


__global__ void sobelFilterX(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

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



__global__ void sobelFilterY(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

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



__global__ void sobelFilter(unsigned char *imageSobelX, unsigned char *imageSobelY, int width, int height, unsigned char *imageSobel){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

	if((row < height) && (col < width)){
		imageSobel[row* width+col] = __powf((__powf(imageSobelX[row*width+col],2) + __powf(imageSobelY[row*width+col],2)),0.5);
	}

}


int main(int argc, char **argv){
    
    clock_t  startGPU, endGPU;
    double  gpu_time_used;
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1};
    char h_Mt[] = {-1,-2,-1,0,0,0,1,2,1};
    char* imageName = argv[1];
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;
    unsigned char *h_imageSobel, *d_imageSobel, *d_imageSobelX,  *d_imageSobelY;
    Mat image;
    image = imread(imageName, 1);

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;


    dataRawImage = (unsigned char*)malloc(size);    
    h_imageOutput = (unsigned char *)malloc(sizeGray);
    h_imageSobel = (unsigned char *)malloc(sizeGray);

  cudaMalloc((void**)&d_imageOutput,sizeGray);
	cudaMalloc((void**)&d_sobelOutput,sizeGray);
	cudaMalloc((void**)&d_imageSobel,sizeGray);
	cudaMalloc((void**)&d_imageSobelX,sizeGray);
	cudaMalloc((void**)&d_imageSobelY,sizeGray);
	cudaMalloc((void**)&d_dataRawImage,sizeGray);
	cudaMalloc((void**)&h_imageSobel,sizeGray);
	
	
	
	dataRawImage = image.data;

    startGPU = clock();
    
    cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M,h_M,sizeof(char)*MASK_WIDTH*MASK_WIDTH);
	cudaMemcpyToSymbol(Mt,h_Mt,sizeof(char)*MASK_WIDTH*MASK_WIDTH);



    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    
    cudaDeviceSynchronize();
    
    sobelFilter<<<dimGrid,dimBlock>>>(d_imageSobelX,d_imageSobelY,width,height,d_sobelOutput);
    sobelFilterX<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_imageSobelX);
    sobelFilterY<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_imageSobelY);
    
    
    cudaMemcpy(h_imageOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);
    cudaMemcpy(d_sobelOutput,h_imageSobel,sizeGray,cudaMemcpyDeviceToHost);

    endGPU = clock();

    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    /*start = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    end = clock();*/


    /*namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image CUDA", WINDOW_NORMAL);
    namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);
    imshow(imageName,image);
    imshow("Gray Image CUDA", gray_image);
    imshow("Sobel Image OpenCV",abs_grad_x);
    waitKey(0);*/

    
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    
    
    
    printf("%.10f\n",gpu_time_used);

    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(M);
    cudaFree(d_sobelOutput);
    return 0;
}

