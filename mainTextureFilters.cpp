#include <dirent.h>
#include "iostream"
#include <fstream>
#include "vector"
#include "set"
#include "stack"
#include <iostream>
#include "stdio.h"
#include "string"
#include "vector"
#include "stack"
#include "queue"
#include "map"
#include "math.h"
#include "time.h"
#include <algorithm>
using namespace std;
#include "opencv2/opencv.hpp"
//#include "opencv2/face.hpp"



using namespace cv;

typedef unsigned int uint;
const float PI = 3.1415;

//response = np.exp(-x ** 2 / (2. * sigma ** 2))
void func1(float *response, float *lengths, float sigma, int size)
{
    for(int i = 0; i < size; i++)
        response[i] = exp(- lengths[i] * lengths[i] / (2 * sigma * sigma));
}

//response = -response * x
void func2(float *response, float *lengths, int size)
{
    for(int i = 0; i < size; i++)
        response[i] = -response[i] * lengths[i];
}

//response = response * (x ** 2 - sigma ** 2)
void func3(float *response, float *lengths, float sigma, int size)
{
    for(int i = 0; i < size; i++)
        response[i] = response[i] * (lengths[i] * lengths[i] - sigma * sigma);
}

// response /= np.abs(response).sum()
void normalize(float *response, int size)
{
    float summ = 0;
    for(int i = 0; i < size; i++)
        summ += std::abs(response[i]);
    for(int i = 0; i < size; i++)
        response[i] /= summ;
}

void make_gaussian_filter(float *response, float *lengths, float sigma, int size, int order=0)
{
    assert(order <= 2);//, "Only orders up to 2 are supported"

    // compute unnormalized Gaussian response
    func1(response, lengths, sigma, size);
    if (order == 1)
        func2(response, lengths, size);
    else if (order == 2)
        func3(response, lengths, sigma, size);

    normalize(response, size);
}

void getX(float *xCoords, Point2f* pts, int size)
{
    for(int i = 0; i < size; i++)
        xCoords[i] = pts[i].x;
}

void getY(float *yCoords, Point2f* pts, int size)
{
    for(int i = 0; i < size; i++)
        yCoords[i] = pts[i].y;
}

void multiplyArrays(float *gx, float *gy, float *response, int size)
{
    for(int i = 0; i < size; i++)
        response[i] = gx[i] * gy[i];
}

void makeFilter(float scale, int phasey, Point2f* pts, float *response, int size)
{
    float xCoords[size];
    float yCoords[size];
    getX(xCoords, pts, size);
    getY(yCoords, pts, size);

    float gx[size];
    float gy[size];
    make_gaussian_filter(gx, xCoords, 3 * scale, size);
    make_gaussian_filter(gy, yCoords, scale, size, phasey);
    multiplyArrays(gx, gy, response, size);
    normalize(response, size);
}

void createPointsArray(Point2f *pointsArray, int radius)
{
    int index = 0;
    for(int x = -radius; x <= radius; x++)
        for(int y = -radius; y <= radius; y++)
        {
            pointsArray[index] = Point2f(x,y);
            index++;
        }
}

void rotatePoints(float s, float c, Point2f *pointsArray, Point2f *rotatedPointsArray, int size)
{
    for(int i = 0; i < size; i++)
    {
        rotatedPointsArray[i].x = c * pointsArray[i].x - s * pointsArray[i].y;
        rotatedPointsArray[i].y = s * pointsArray[i].x - c * pointsArray[i].y;
    }
}

void computeLength(Point2f *pointsArray, float *length, int size)
{
    for(int i = 0; i < size; i++)
        length[i] = sqrt(pointsArray[i].x * pointsArray[i].x + pointsArray[i].y * pointsArray[i].y);
}

void toMat(float *edgeThis, Mat &edgeThisMat, int support)
{
    edgeThisMat = Mat::zeros(support, support, CV_32FC1);
    float* nextPts = (float*)edgeThisMat.data;
    for(int i = 0; i < support * support; i++)
    {
        nextPts[i] = edgeThis[i];
    }
}

void makeRFSfilters(vector<Mat>& edge, vector<Mat >& bar, vector<Mat>& rot,
                    vector<float> &sigmas, int n_orientations=6, int radius=24)
{
    int support = 2 * radius + 1;
    int size = support * support;
    Point2f orgpts[size];
    createPointsArray(orgpts, radius);

    for(uint sigmaIndex = 0; sigmaIndex < sigmas.size(); sigmaIndex++)
        for(int orient = 0; orient < n_orientations; orient++)
        {
            float sigma = sigmas[sigmaIndex];
            //Not 2pi as filters have symmetry
            float angle = PI * orient / n_orientations;
            float c = cos(angle);
            float s = sin(angle);
            Point2f rotpts[size];
            rotatePoints(s, c, orgpts, rotpts, size);
            float edgeThis[size];
            makeFilter(sigma, 1, rotpts, edgeThis, size);
            float barThis[size];
            makeFilter(sigma, 2, rotpts, barThis, size);
            Mat edgeThisMat;
            Mat barThisMat;
            toMat(edgeThis, edgeThisMat, support);
            toMat(barThis, barThisMat, support);

            edge.push_back(edgeThisMat);
            bar.push_back(barThisMat);
        }

    float length[size];
    computeLength(orgpts, length, size);

    float rotThis1[size];
    float rotThis2[size];
    make_gaussian_filter(rotThis1, length, 10, size);
    make_gaussian_filter(rotThis2, length, 10, size, 2);

    Mat rotThis1Mat;
    Mat rotThis2Mat;
    toMat(rotThis1, rotThis1Mat, support);
    toMat(rotThis2, rotThis2Mat, support);
    rot.push_back(rotThis1Mat);
    rot.push_back(rotThis2Mat);
}

//", vector<Mat>& edgeNormalised"
void normaliseFilters(vector<Mat>& edge, vector<Mat>& edgeNormalised)
{
    int size = edge[0].rows * edge[0].cols;
    for(uint i = 0; i < edge.size(); i++)
    {
        Mat edgeNew = Mat::zeros(edge[i].rows, edge[i].cols, CV_8UC1);
        uchar* pNewEdge = (uchar*) edgeNew.data;
        float* pEdge = (float*)edge[i].data;
        double minVal; double maxVal; Point minLoc; Point maxLoc;

        minMaxLoc(edge[i], &minVal, &maxVal, &minLoc, &maxLoc, noArray());
        float multyplayFactor = 125 / max(abs(minVal), abs(maxVal));
        for(int i = 0; i < size; i++)
        {
            *pNewEdge = 125 + *pEdge * multyplayFactor;
            pEdge++;
            pNewEdge++;
        }
        edgeNormalised.push_back(edgeNew);
    }
}

void drawing(vector<Mat>& edge, vector<Mat>& bar, vector<Mat>& rot, int n_sigmas, int n_orientations)
{
    vector<Mat > edgeNormalised, barNormalised, rotNormalised;
    normaliseFilters(edge, edgeNormalised);
    normaliseFilters(bar, barNormalised);
    normaliseFilters(rot, rotNormalised);

    int support = edge[0].cols;
    int width = support * (1.5 * n_orientations +1);
    int height = support * (1.5 * n_sigmas + 1) + support * (1.5 * n_sigmas) + support * 1.5;

    Mat filterShow = Mat::zeros(height, width, CV_8UC1);

    int edgeIndex = 0;
    for(int sigmaIndex = 0; sigmaIndex < n_sigmas; sigmaIndex++)
        for(int orient = 0; orient < n_orientations; orient++)
        {
            //edge
            Mat ref = filterShow(Rect(support * (1.5*orient + 0.5), support * (1.5*sigmaIndex + 0.5), support, support));
            edgeNormalised[edgeIndex].copyTo(ref);

            //bar
            ref = filterShow(Rect(support * (1.5*orient + 0.5), support * (1.5*sigmaIndex + 0.5 + n_sigmas * 1.5), support, support));
            barNormalised[edgeIndex].copyTo(ref);
            edgeIndex++;
        }

    //rot
    Mat ref = filterShow(Rect(support * (1.5*0 + 0.5), support * (0.5 + 2 * n_sigmas * 1.5), support, support));
    rotNormalised[0].copyTo(ref);
    ref = filterShow(Rect(support * (1.5*1 + 0.5), support * (0.5 + 2 * n_sigmas * 1.5), support, support));
    rotNormalised[1].copyTo(ref);

    imshow("filterShow", filterShow);
}

void  apply_filterbank(Mat &img,
                       vector<vector<Mat> > &filterbank,
                       vector<vector<Mat> > &response,
                       int n_sigmas, int n_orientations)
{
    response.resize(3);
    vector<Mat>& edges = filterbank[0];
    vector<Mat>& bar = filterbank[1];
    vector<Mat>& rot = filterbank[2];
    int i = 0;
    for(int sigmaIndex = 0; sigmaIndex < n_sigmas; sigmaIndex++)
    {
        Mat newMat = Mat::zeros(img.rows, img.cols, img.type());
        for(int orient = 0; orient < n_orientations; orient++)
        {
            Mat dst;
            filter2D(img, dst,  -1, edges[i], Point( -1, -1 ), 0, BORDER_DEFAULT );
            newMat = cv::max(dst, newMat);
            i++;
        }
        Mat newMatUchar;
        newMat = cv::abs(newMat);
        newMat.convertTo(newMatUchar, CV_8UC1);
        response[0].push_back(newMatUchar);
    }

    i = 0;
    for(int sigmaIndex = 0; sigmaIndex < n_sigmas; sigmaIndex++)
    {
        Mat newMat = Mat::zeros(img.rows, img.cols, img.type());
        for(int orient = 0; orient < n_orientations; orient++)
        {
            Mat dst;
            filter2D(img, dst,  -1 , bar[i], Point( -1, -1 ), 0, BORDER_DEFAULT );
            newMat = max(dst, newMat);
            i++;
        }
        Mat newMatUchar;
        newMat = cv::abs(newMat);
        newMat.convertTo(newMatUchar, CV_8UC1);
        response[1].push_back(newMatUchar);
    }

    for(uint i = 0; i < 2; i++)
    {
        Mat newMat = Mat::zeros(img.rows, img.cols, img.type());
        Mat dst;
        filter2D(img, dst,  -1 , rot[i], Point( -1, -1 ), 0, BORDER_DEFAULT );
        newMat = max(dst, newMat);
        Mat newMatUchar;
        newMat = cv::abs(newMat);
        newMat.convertTo(newMatUchar, CV_8UC1);
        response[2].push_back(newMatUchar);
    }


}

void normaliseImage(Mat& image, Mat& normalised)
{
    normalised = Mat::zeros(image.rows, image.cols, CV_8UC1);
    uchar* pImage = (uchar*) image.data;
    uchar* pNormalised = (uchar*)normalised.data;
    double minVal; double maxVal; Point minLoc; Point maxLoc;

    minMaxLoc(image, &minVal, &maxVal, &minLoc, &maxLoc, noArray());

    for(int i = 0; i < image.rows * image.cols; i++)
    {
        *pNormalised = (*pImage - minVal)/ (maxVal - minVal) * 255;

        pImage++;
        pNormalised++;
    }
}

/************************* Mine **********************************/
// Aggregate Images
void saveImgs(uint num, Mat image) {
    imwrite("images/imageNo1.png", image);
}

void aggregateImg(uint num, double alpha, Mat &aggImg, Mat input) {
  double beta = 1.0 - alpha;
  if (num == 0) {
    cout << "initialising aggImg" << endl;
    input.copyTo(aggImg);
  } else {
    cout << "Adding new image, num = " << num << endl;
    addWeighted(aggImg, beta , input, alpha, 0.0, aggImg, -1);
  }
}

// Cluster images
Mat applyKmeans(Mat samples){
  int clusterCount = 10, attempts = 5;
  Mat  labels, centers;

  // Apply KMeans
  kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
  return centers;
}

Mat createSamples(Mat input){
    input.convertTo(input, CV_32F);
    Mat samples(input.rows * input.cols, 1, CV_32F);

    // Copy across input image
    for (int y = 0; y < input.rows; y++) {
      for (int x = 0; x < input.cols; x++) {
        samples.at<float>(y, x) = input.at<float>(y, x);
      }
    }
    return applyKmeans(samples);
}
/************************* End **********************************/

void drawingResponce(vector<vector<Mat> > &response, Mat &aggImg, double &alpha, int &counter) {
    int numInRow = response[0].size();
    int support = response[0][0].cols;
    Mat responceShow = Mat::zeros((1.5 * numInRow + 1) * support, (1.5 * numInRow + 1) * support, CV_8UC1);
    rectangle(responceShow, Rect(0,0, responceShow.cols, responceShow.rows), Scalar(125), -1);

    //edges
    for(uint type = 0; type < response.size(); type++)
    {
        for(uint imageIndex = 0; imageIndex < response[type].size(); imageIndex++)
        {
            Mat normalised;
            normaliseImage(response[type][imageIndex], normalised);
            imshow("normalisedImage", normalised);
            waitKey();
            //saveImgs(imageIndex ,normalised);
            aggregateImg(counter, alpha, aggImg, normalised);
            alpha *= 0.5;
            cout << "This is the alpha:" << alpha << "coutner:" << counter << endl;
            normalised.copyTo(responceShow(Rect((1.5*imageIndex + 0.5) * support, (1.5*type + 0.5) * support, support,support)));
            counter++;
            imshow("aggImg", aggImg);
        }
    }

    resize(responceShow, responceShow, Size(responceShow.cols/8, responceShow.rows/8));

    imshow("responceShow", responceShow);
}

/*************************** START Import images from Dir ****************************/
// Check that file in dir is an accepted img type
bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

/*************************** END Import images from Dir ****************************/

int main()
{
    Mat aggImg;

    vector<float> sigmas;
    sigmas.push_back(1);
    sigmas.push_back(2);
    sigmas.push_back(4);
    int n_sigmas = sigmas.size();
    int n_orientations = 6;
    vector<Mat > edge, bar, rot;
    makeRFSfilters(edge, bar, rot, sigmas, n_orientations);

    //plot filters
    drawing(edge, bar, rot, n_sigmas, n_orientations);

    vector<vector<Mat > > filterbank;
    filterbank.push_back(edge);
    filterbank.push_back(bar);
    filterbank.push_back(rot);

    vector<vector<Mat> > response;

    //apply filters to lena
    // Mat img = imread("../../Lena.png", IMREAD_GRAYSCALE);
    // cout << "This is lena size: " << img.size() << endl;
    // cout << "image equalised" << endl;

    // import images from dir
    std::string extTypes[] = {".jpg", ".png", ".bmp"};
    std::string dirNme = "../../../TEST_IMAGES/kth-tips/bread/train/";

    DIR *dir;
    dir = opendir(dirNme.c_str());
    string imgName;
    struct dirent *ent;

    double alpha = 0.5;
    int counter =0;

    if(dir != NULL){
      while ((ent = readdir(dir)) != NULL) {
        imgName = ent->d_name;
        if (hasEnding(imgName, ".png")) {
          cout << "correct extension" << endl;

          // Sort out string Stream
          std::stringstream ss;
          ss << dirNme << imgName;
          std::string imgpath = ss.str();

          Mat img = imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
          equalizeHist(img, img);

          cout << "This is imgpath:" << imgpath << " img.size()" << img.size()  << endl;
          imshow("hello", img);
          waitKey();

          Mat imgFloat;
          img.convertTo(imgFloat, CV_32FC1);

          apply_filterbank(imgFloat, filterbank, response, n_sigmas, n_orientations);
          drawingResponce(response, aggImg, alpha, counter);
          response.clear();
          img.release();
          imgFloat.release();
        } else{
          cout << "incorrect extension" << endl;
        }
      }
    }

    //cluster images
    Mat centers = createSamples(aggImg);

    cout << "These are the clusters: " << centers << endl;
    FileStorage fs("../textonDictionary.xml", FileStorage::WRITE);
    fs << "cluster" << centers;
    fs.release();

    waitKey(0);
}
