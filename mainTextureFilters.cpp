#include <dirent.h>
#include <unistd.h> // For sleep function
#include <iostream>
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
int texDictSize = 40;

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

    // Apply Edge Filters //
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

    // Apply Bar Filters //
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

    // Apply Gaussian and LoG Filters //
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

cout <<"leaving apply filtebank" << endl;
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
  //  cout << "initialising aggImg" << endl;
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
  cout << "apply kmeans, this is size: " << labels.size() << "centres" << centers.size() << endl;
  //cout << "sample is this size: " << samples;

  // Apply KMeans
  kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
  return centers;
}

Mat createSamples(Mat input){
    cout << "Create samples these are the samples before imshow" << input ;

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

void drawingResponce(vector<vector<Mat> > &response, Mat &aggImg, int &counter){
//    int numInRow = response[0].size();
//    int support = response[0][0].cols;
//    Mat responceShow = Mat::zeros((1.5 * numInRow + 1) * support, (1.5 * numInRow + 1) * support, CV_8UC1);
//    rectangle(responceShow, Rect(0,0, responceShow.cols, responceShow.rows), Scalar(125), -1);
    double alpha = 0.5;

    for(uint type = 0; type < response.size(); type++)
    {
      cout << "This is the response size:" << response.size() << endl;
        for(uint imageIndex = 0; imageIndex < response[type].size(); imageIndex++)
        {
          cout << "This is the response[type].size:" << response[type].size() << "Type: " << type << endl;

          //  Mat normalised;
          //  normaliseImage(response[type][imageIndex], normalised);
            // imshow("normalisedImage", normalised);
            //saveImgs(imageIndex ,normalised);

            aggregateImg(counter, alpha, aggImg, response[type][imageIndex]);
            alpha *= 0.5;

            cout << "aggrgate image taking shape: " << endl;



            // cout << "This is the alpha:" << alpha << "coutner:" << counter << endl;
            // normalised.copyTo(responceShow(Rect((1.5*imageIndex + 0.5) * support, (1.5*type + 0.5) * support, support,support)));
            // counter++;
        }
    }
    // resize(responceShow, responceShow, Size(responceShow.cols/8, responceShow.rows/8));
    //
    // imshow("responceShow", responceShow);
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

Mat loadImg(string imgpath){
  Mat img = imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
  equalizeHist(img, img);

  cout << "This is imgpath:" << imgpath << " img.size()" << img.size()  << endl;

  Mat imgFloat;
  img.convertTo(imgFloat, CV_32FC1);
  return imgFloat;
}

void printTexDict(double (&textonDictArray)[40]){
  int classesSize = sizeof(textonDictArray)/sizeof(textonDictArray[0]);
  cout << "This is the full texton Dictionary size:" << classesSize << endl;
  for(int i = 0;i < classesSize;i++){
    cout << i << ":" << textonDictArray[i]<< endl;
  }
}

void createTexDic(vector<vector<Mat> > &filterbank, int n_sigmas, int n_orientations, double (&textonDictArray)[40]){


  // import images from dir
  string extTypes[] = {".jpg", ".png", ".bmp"};
  string classes[] = {"cotton/train/", "wood/train/", "cork/train/", "bread/train/"};
  int classesSize = sizeof(classes)/sizeof(classes[0]);

  int doCount = 0;
  do{

    stringstream dss;
    string dirtmp = "../../../TEST_IMAGES/kth-tips/";
    dss << dirtmp;
    dss << classes[doCount];
    string dirNme = dss.str();

    cout << "this is the dir name: " << dirNme << endl;

    DIR *dir;
    dir = opendir(dirNme.c_str());

    string imgName;
    struct dirent *ent;

    int counter = 0;

    if(dir != NULL){
      Mat aggImg;
      vector<vector<Mat> > response;

      while ((ent = readdir(dir)) != NULL) {
        imgName = ent->d_name;

        if (hasEnding(imgName, ".png")) {
          cout << "correct extension: " << imgName << endl;

          // Sort out string Stream
          std::stringstream ss;
          ss << dirNme << imgName;
          std::string imgpath = ss.str();

          Mat imgFloat = loadImg(imgpath);

          apply_filterbank(imgFloat, filterbank, response, n_sigmas, n_orientations);
        } else{
          cout << "incorrect extension:" << imgName << "LL" << endl;
        }
      }
      drawingResponce(response, aggImg, counter);
      cout << "This is your agg img" << endl;
      imshow("agg image", aggImg);
      cvDestroyWindow("agg image");
      waitKey();

      //cluster images
      Mat centers = createSamples(aggImg);

      cout << "These are the clusters: " << centers << endl;

      // the number of values already in array
      int start = doCount * 10;
      for(int j =0;j<centers.rows;j++){
        textonDictArray[j+start] = centers.at<float>(0, j);
      }
    }

    FileStorage fs("../textonDictionary.xml", FileStorage::WRITE);
    // if(doCount == 0){
    // FileStorage fs("../textonDictionary.xml", FileStorage::WRITE);
    // fs << "wood" << "[";
    // fs << centers;
    // fs << "]";
    // } else {
    // FileStorage fs("../textonDictionary.xml", FileStorage::APPEND);
    // fs << "wood" << "[";
    // fs << centers;
    // fs << "]";
    // }

    fs.release();
    doCount ++;
  }while(doCount < classesSize);
  printTexDict(textonDictArray);

}

void createModels(vector<vector<Mat> > filterbank, int n_sigmas, int n_orientations, double (&texDict)[40]){
  Mat aggImg;
  double alpha =0.5;
  int counter =0;
  vector<vector<Mat> > response;

  string path;
  stringstream ss;
  cout << "please enter the directory(excluding the .png extension) of your file assuming the path:\n../../../TEST_IMAGES/kth-tips/" << endl;
  //cin >> path;
  ss << "../../../TEST_IMAGES/kth-tips/bread/train/" << "52a-scale_2_im_1_col" << ".png";
  cout << "This is your path: \n" << ss.str();

  Mat img = loadImg(ss.str());
  cout << "Mat size.." << img.size();

  apply_filterbank(img, filterbank, response, n_sigmas, n_orientations);
  drawingResponce(response, aggImg, counter);
  Mat centers = createSamples(aggImg);
  cout << "These are the cluster centers" << centers << endl;
  cout << "This is the current texton dictionary no 15: " << endl;
  response.clear();
}

int main()
{

    double textonDictionary[40];

    vector<float> sigmas;
    sigmas.push_back(1);
    sigmas.push_back(2);
    sigmas.push_back(4);
    int n_sigmas = sigmas.size();
    int n_orientations = 6;
    vector<Mat > edge, bar, rot;
    makeRFSfilters(edge, bar, rot, sigmas, n_orientations);

    //plot filters
//    drawing(edge, bar, rot, n_sigmas, n_orientations);

    vector<vector<Mat > > filterbank;
    filterbank.push_back(edge);
    filterbank.push_back(bar);
    filterbank.push_back(rot);


    bool cont;
    do{
      cout << "\nMenu: Please enter the chosen options number \n"<< endl;
      cout << "1: texton dictionary creation" << endl;
      cout << "2: model creation" << endl;
      cout << "3: model testing" << endl;
      cout << "4: exit\n" << endl;

      string tmp;
      cin >> tmp;

      if(tmp == "1"){
        cout << "creating texton dictionary" << endl;
        createTexDic(filterbank, n_sigmas, n_orientations, textonDictionary);
        sort(textonDictionary, textonDictionary+40);
        cout << "printing texton dictionary from main" << endl;
        printTexDict(textonDictionary);
        cont = true;
      } else if(tmp == "2"){
        cout << "Creating models" << endl;
        printTexDict(textonDictionary);
        createModels(filterbank, n_sigmas, n_orientations, textonDictionary);
        cont = true;
      } else if(tmp == "3"){
        cout << "Entering test mode" << endl;
        cont = true;
      } else if(tmp == "4"){
        cout << "exiting" << endl;
        cont = false;
      } else {
        cout << "that input was not recognised." << endl;
        cvDestroyAllWindows();
        cont = true;
      }
    }while(cont);

}
