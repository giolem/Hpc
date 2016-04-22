#include <cv.h>
#include <highgui.h>
#include <time.h>


#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;




int main(int argc, char **argv){

    char* imageName = argv[1];
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;
    Mat image;
    image = imread(imageName, 1);

    if(argc !=2 || !image.data){
        printf("No image Data \n");
        return -1;
    }

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;


    dataRawImage = (unsigned char*)malloc(size);

    dataRawImage = image.data;
    h_imageOutput = (unsigned char *)malloc(sizeGray);
    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;


    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);



    imwrite("./Sobel_Image.jpg",gray_image);

    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image CUDA", WINDOW_NORMAL);
    namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image CUDA", gray_image);
    imshow("Sobel Image OpenCV",abs_grad_x);

    waitKey(0);
    return 0;
}
