#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


void load_flat_images(const String &dirname, vector<Mat> &img_list, bool showImages = false)
{
    /* Load PGM images
       and create a vector of flattened images 
    */  
    vector<String> files;
    glob(dirname, files);
    for (size_t i = 0; i < files.size(); ++i)
    {
        Mat img = imread(files[i], IMREAD_GRAYSCALE); // load the images
        if ( img.empty() )            // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        if (showImages)
        {
            imshow("Thermal_Images", img);
            waitKey(0);
        }
        img = img.reshape(0, 1); //flatten images
        img_list.push_back(img);
    }
}

Mat average_face(vector<Mat> &img_list, bool showAvgFace = false)
{
    /* Calculates the mean face */

    cout << "Calculating the average route..." << endl;
    Mat mean;
    Mat mean_mean, stddev_mean;
    for (int col = 0; col < img_list[0].cols; ++col)
    {
        int sum = 0;
        for (int row = 0; row < img_list.size(); ++row)
        {
            sum += (int)img_list[row].at<uchar>(0, col);
        }     
        mean.push_back((double)(sum/img_list.size()));    
    }

    mean = mean.reshape(0, 512); // reshape to size of original image 48x60
    mean.convertTo(mean, CV_8U);
    cout<< "DONE" << endl; 

    imwrite( "./Results_Thermal/AverageRoute.jpg", mean);
    if (showAvgFace)
        {
            imshow("Average Route", mean);
            waitKey(0);
        }

    return mean;
}

static Mat prepareDataForPCA(const vector<Mat> &images)
{
  cout << "Prepare images for PCA ..." << endl;
  
  Mat data(static_cast<int>(images.size()), images[0].rows * images[0].cols, CV_8U);
  
  // Turn an image into one row vector in the data matrix
  for(unsigned int i = 0; i < images.size(); i++)
  {
    // Copy the long vector into one row of the dest
    images[i].copyTo(data.row(i)); 
  }

  cout << "DONE" << endl;
  return data;
}
    
void showQueryImage(const String &dirname, int &index)
    {
        vector<String> files;
        glob(dirname, files);
        Mat img = imread(files[index], IMREAD_UNCHANGED); 
        namedWindow("Query Image", WINDOW_AUTOSIZE);
        imshow("Query Image", img);
        waitKey(0);
    }    

void showRetrievedImage(const String &dirname, int &index)
    {
        vector<String> files;
        glob(dirname, files);
        Mat img = imread(files[index], IMREAD_UNCHANGED); 
        namedWindow("Retrieved Image", WINDOW_AUTOSIZE);
        imshow("Retrieved Image", img);
        waitKey(0);
    } 

int main(){

    vector<Mat> images;
    load_flat_images("./EigenThermal", images, false); //images as rows 

    cout << "Number of images is: " << images.size() << endl; //number of images in the folder, rows
    cout << "Number of pixels for each image is: " << images[0].cols << endl;

    Mat avg_train_face = average_face(images, true);
    Mat PCA_data = prepareDataForPCA(images);
    
    //Calculate PCA of the data matrix
    cout << "Calculating PCA ..." << endl;
    PCA pca(PCA_data, Mat(), PCA::DATA_AS_ROW); 
    cout << "DONE"<< endl;
    cout << (int)PCA_data.at<uchar>(0,0) << endl;

    // eigenvectors
    Mat eigenVectors = pca.eigenvectors;
    cout << eigenVectors.rows << endl;
    cout << eigenVectors.cols << endl;

    Mat eigenValues = pca.eigenvalues;
    cout << eigenValues.rows << endl;
    cout << eigenValues.cols << endl;

    normalize(eigenVectors, eigenVectors, 0, 255, NORM_MINMAX); // normalize eigenvectors for display
    
    // Display the eigenfaces corresponding to the 10 largest eigenvalues   
    cout << "EigenImages corresponding to the 10 highest eigenvalues ..." << endl;
    for(int i = 0; i < 10 ; i++)
    {
        //cout << eigenVectors.row(i) << endl;
        Mat eigenFace = eigenVectors.row(i).reshape(0, 512);
        eigenFace.convertTo(eigenFace, CV_8U);
        imwrite( "./Results_Thermal/eigenFace" + to_string(i) + ".jpg", eigenFace);
        imshow("EigenFaces", eigenFace);
        waitKey(0);
        //Mat eigenImages.push_back(eigenFace);
        
    }
    cout << "DONE"<< endl;

    // Display the eigenfaces corresponding to the 10 smallest eigenvalues   
    cout << "EigenImages corresponding to the 10 smallest eigenvalues ..." << endl; 
    for(int i = eigenVectors.rows - 1; i > eigenVectors.rows - 11; i--)
    {

        Mat eigenFace = eigenVectors.row(i).reshape(0, 512);
        eigenFace.convertTo(eigenFace, CV_8U);
        imshow("EigenImages", eigenFace);
        waitKey(0);
        //Mat eigenImages.push_back(eigenFace);
        
    }
    cout << "DONE"<< endl;
