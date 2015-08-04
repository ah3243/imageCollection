#include <dirent.h>
#include <unistd.h> // For sleep function
#include <iostream>
#include <fstream>
#include "set"
#include "stack"
#include <iostream>
#include "stdio.h"
#include "string"
#include <vector>
#include "queue"
#include "map"
#include "math.h"
#include "time.h"
#include <algorithm>
using namespace std;
#include "opencv2/opencv.hpp"
#include <typeinfo>
#include <sstream>
#include <chrono>  // time measurement
#include <thread>  // time measurement
#include <boost/math/special_functions/round.hpp>

using namespace cv;

typedef unsigned int uint;
const float PI = 3.1415;

typedef vector<float> m1;
typedef vector<m1> m2;
typedef vector<m2> m3;

typedef vector<Mat> mH1;
typedef vector<mH1> mH2;

//response = np.exp(-x ** 2 / (2. * sigma ** 2))
void func1(float *response, float *lengths, float sigma, int size) {
    for(int i = 0; i < size; i++)
        response[i] = exp(- lengths[i] * lengths[i] / (2 * sigma * sigma));
}

//response = -response * x
void func2(float *response, float *lengths, int size) {
    for(int i = 0; i < size; i++)
        response[i] = -response[i] * lengths[i];
}

//response = response * (x ** 2 - sigma ** 2)
void func3(float *response, float *lengths, float sigma, int size) {
    for(int i = 0; i < size; i++)
        response[i] = response[i] * (lengths[i] * lengths[i] - sigma * sigma);
}

// response /= np.abs(response).sum()
void normalize(float *response, int size) {
    float summ = 0;
    for(int i = 0; i < size; i++)
        summ += std::abs(response[i]);
    for(int i = 0; i < size; i++)
        response[i] /= summ;
}

void make_gaussian_filter(float *response, float *lengths, float sigma, int size, int order=0) {
    assert(order <= 2);//, "Only orders up to 2 are supported"

    // compute unnormalized Gaussian response
    func1(response, lengths, sigma, size);
    if (order == 1)
        func2(response, lengths, size);
    else if (order == 2)
        func3(response, lengths, sigma, size);

    normalize(response, size);
}

void getX(float *xCoords, Point2f* pts, int size) {
    for(int i = 0; i < size; i++)
        xCoords[i] = pts[i].x;
}

void getY(float *yCoords, Point2f* pts, int size) {
    for(int i = 0; i < size; i++)
        yCoords[i] = pts[i].y;
}

void multiplyArrays(float *gx, float *gy, float *response, int size) {
    for(int i = 0; i < size; i++)
        response[i] = gx[i] * gy[i];
}

void makeFilter(float scale, int phasey, Point2f* pts, float *response, int size) {
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

void createPointsArray(Point2f *pointsArray, int radius) {
    int index = 0;
    for(int x = -radius; x <= radius; x++)
        for(int y = -radius; y <= radius; y++)
        {
            pointsArray[index] = Point2f(x,y);
            index++;
        }
}

void rotatePoints(float s, float c, Point2f *pointsArray, Point2f *rotatedPointsArray, int size) {
    for(int i = 0; i < size; i++)
    {
        rotatedPointsArray[i].x = c * pointsArray[i].x - s * pointsArray[i].y;
        rotatedPointsArray[i].y = s * pointsArray[i].x - c * pointsArray[i].y;
    }
}

void computeLength(Point2f *pointsArray, float *length, int size) {
    for(int i = 0; i < size; i++)
        length[i] = sqrt(pointsArray[i].x * pointsArray[i].x + pointsArray[i].y * pointsArray[i].y);
}

void toMat(float *edgeThis, Mat &edgeThisMat, int support) {
    edgeThisMat = Mat::zeros(support, support, CV_32FC1);
    float* nextPts = (float*)edgeThisMat.data;
    for(int i = 0; i < support * support; i++)
    {
        nextPts[i] = edgeThis[i];
    }
}

void makeRFSfilters(vector<Mat>& edge, vector<Mat >& bar, vector<Mat>& rot, vector<float> &sigmas, int n_orientations=6, int radius=24) {
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
void normaliseFilters(vector<Mat>& edge, vector<Mat>& edgeNormalised) {
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

void drawing(vector<Mat>& edge, vector<Mat>& bar, vector<Mat>& rot, int n_sigmas, int n_orientations) {
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

void  apply_filterbank(Mat &img, vector<vector<Mat> >filterbank, vector<vector<Mat> > &response, int n_sigmas, int n_orientations) {
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
  cout <<"leaving apply filterbank" << endl;
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

// Save to image dir
void saveImgs(uint num, Mat image) {
    imwrite("images/imageNo1.png", image);
}

// Aggregate Images
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

// Apply kmeans, rtn centers
Mat applyKmeans(Mat samples, int clusterCount){
    int  attempts = 5;
    Mat  labels, centers;
    // Apply KMeans
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
    return centers;
}

// convert 2d Mat to 1Col float mat, Pass to kmeans
Mat createSamples(Mat input, int cluster){
    input.convertTo(input, CV_32F);
    Mat samples(input.rows * input.cols, 1, CV_32F);
    // Copy across input image
    for (int y = 0; y < input.rows; y++) {
      for (int x = 0; x < input.cols; x++) {
        samples.at<float>(y, x) = input.at<float>(y, x);
      }
    }
    return applyKmeans(samples, cluster);
}

// Print Texton Dictionary
void printTexDict(vector<float> textonDict){
  int classesSize = textonDict.size();
  cout << "This is the full texton Dictionary size:" << classesSize << endl;
  for(int i = 0;i < classesSize;i++){
    cout << i << ":" << textonDict.at(i)<< endl;
  }
}

void printModelsInner(vector<float> v, int count){
  if(v.size()!=0){
   cout << "Model:" << count << "\nIt's size is: " << v.size() <<  endl;
      for(int i = 0; i < v.size(); i++){
          cout << v[i] << " ";
      }
      cout << "\n";
    }
}

void printModels(vector <vector<float> > v){
  cout << "Below are the model cluster centers, vector size: " << v.size() << endl;
    for(int b =0;b<v.size();b++)
     printModelsInner(v[b], b);
}

// Check that file in dir is an accepted img type
bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

// Load and Equalise Imgs from path, rtn img float
bool loadImg(Mat& img){
  if(!img.data){
    return false;
  }
  equalizeHist(img, img);

  img.convertTo(img, CV_32FC1);
  return true;
}

// Round to 3dp
  //inner loop
  void roundModelInner(vector<float>& v){
    for(int j=0;j<v.size();j++){
      v[j] = floorf(v[j]*1000)/1000;
    }
  }
  // outerloop
  void roundModel(vector<vector<float> >& v){
    for(int i=0;i<v.size();i++){
      roundModelInner(v[i]);
    }
  }

// converts input 1d Mat to vector
void matToVec(vector<float> &textonDict, Mat centers){
  for(int j = 0;j < centers.rows;j++){
    textonDict.push_back(centers.at<float>(0, j));
  }
}

void drawingResponceInner(vector<Mat>& response, vector<vector<float> > &models, int &counter, int flag, Mat &aggImg, uint type, double& alpha){
  for(uint imageIndex = 0; imageIndex < response.size(); imageIndex++){
    if(flag){
      // Aggregate for Texton Dictionary
      aggregateImg(counter, alpha, aggImg, response[imageIndex]);
      alpha *= 0.5;
    }else {
      cout << "\ndrawing Response model:" << type << " : " << imageIndex << endl;
      // cluster and save to models
      Mat clusters = createSamples(response[imageIndex], 10);
      matToVec(models[counter], clusters);
    }
  }
}

// produce Agg image from responses
void drawingResponce(vector<vector<Mat> > &response, vector<vector<float> > &models, int &counter, int flag, Mat &aggImg){
    double alpha = 0.5;
    for(uint type = 0; type < response.size(); type++)
    {
      drawingResponceInner(response[type], models, counter, flag, aggImg, type, alpha);
    }
}

void testImgModel(vector<vector<Mat> > &response, vector<float> &model){
  int numOfClusters = 10;

  for(int i = 0; i < response.size(); i++){
    for(int j = 0; j < response[i].size(); j++){
      Mat clusters = createSamples(response[i][j], numOfClusters);
      matToVec(model, clusters);
    }
  }
}

// Generate models from training images
void createModels(vector<vector<Mat> >& response, vector<vector<float> >& models, int counter){
  Mat aggImg;

  drawingResponce(response, models,counter, 0, aggImg);
//  roundModel(models);

  waitKey(1000);
  response.clear();
}

// Generate Texton Dictionary from all imgs in sub dirs
void createTexDic(mH2 filterbank, vector<string> classes , m3& models, int n_sigmas, int n_orientations, vector<float>& textonDict, string type){
  string extTypes[] = {".jpg", ".png", ".bmp"};
  int classesSize = classes.size();
  int doCount = 0;

  do{
    string classnme = classes.at(doCount);

    stringstream dss;
    string dirtmp = "../../../TEST_IMAGES/kth-tips/";
    dss << dirtmp;
    dss << classnme;
    dss << "/";
    dss << type;
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

      int imagecounter =0;
      while ((ent = readdir(dir)) != NULL) {
        imgName = ent->d_name;

        if (hasEnding(imgName, ".png")) {
          cout << "correct extension: " << imgName << endl;

          // Sort out string Stream
          std::stringstream ss;
          ss << dirNme << imgName;
          std::string imgpath = ss.str();

          // Load image
          Mat img = imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);

          if(loadImg(img)){
            cout << "This is imgpath:" << imgpath << " img.size()" << img.size()  << endl;
            // Apply and store in response
            apply_filterbank(img, filterbank, response, n_sigmas, n_orientations);

            // If type is test cluster each image
            if(type.compare("test/")==0){
              createModels(response, models[doCount], counter);
              response.clear();
            }
          }

        } else{
          cout << "incorrect extension: " << imgName << endl;
        }
      imagecounter++;
      }

      // If type is train aggregate and generate clusters
      if(type.compare("train/")==0){
        // Aggregate fitler responses from the same classes
        drawingResponce(response, models[doCount], counter, 1,aggImg);
        Mat centers = createSamples(aggImg, 10);
        //cluster aggregated response
        cout << "These are the clusters: " << centers << endl;
        // Store kmeans cluster centers(textons) in vector referenced from main//
        // the number of values already in array
        matToVec(textonDict, centers);
      }
    }

    doCount ++;
  }while(doCount < classesSize);
}

void roundTex(vector<float>& tex){
  for(int i=0;i<tex.size();i++){
      tex[i] = floorf((tex[i])*1000)/1000;
      //tex[i] = round(tex[i]);
  }
}

void removeDups(vector<float>& tex){
    set<float> v;

    unsigned size = tex.size();

    for(unsigned i = 0;i<size;i++)
      v.insert(tex[i]);

    // Clear tex vector
    tex.clear();
    // Insert sorted values
    tex.assign(v.begin(), v.end());
}

int maxHistVal(Mat in){
  int maxVal = 0;

  // Started at 1 to avoid first very high value..
  for(int i = 1; i < in.rows;i++){
    for(int j = 0; j < in.cols;j++){
      if(in.at<float>(i,j)>maxVal){
        maxVal = in.at<float>(i,j);
      }
    }
  }
  return maxVal;
}

// Calculate and return histogram image
Mat showHist(Mat& inputHist, int histBins){
  // Create variables
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histBins );
  int maxVal = maxHistVal(inputHist);

  // Calibrate the maximum histogram value at 80% of window height
  double scaleFactor = ((hist_h*0.8)/maxVal);

  Mat m( hist_h, hist_w, CV_8UC1, Scalar( 0) );
  int countHist = 0;
  cout << "\n\n";
  for( int i = 1; i < histBins; i++ ){
    // Count sum of histogram values
    countHist += inputHist.at<float>(i);
    cout << "This is value: " << i << " value: " << inputHist.at<float>(i) << endl;
    // Draw rectangle representative of hist values in output image
    rectangle( m, Point( bin_w*(i),  hist_h) ,
                    Point( (bin_w*(i))+bin_w,  hist_h - (inputHist.at<float>(i) * scaleFactor)),
                    Scalar( 255, 255, 255),-1, 8);
  }
  cout << "\n\n";
  return m;
}

void createHist(Mat& in, Mat& out, int histSize, const float* histRange, bool uniform){
  bool accumulate = false;

  // Compute the histograms:
  cout << "CreateHist: This is input sze: " << in.size() << " And out size: " << out.size() << endl;

  calcHist( &in, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate );
}

// Takes in Vector<float> and converts to Mat<float>
void textToMat(Mat& tmp, vector<float> texDict){
  for(int i = 0; i < texDict.size();i++){
    tmp.at<float>(i,0) = texDict.at(i);
  }
}

void textonFind(vector<float> txtDict, float& m){
  float distance = 0.0, nearest = 0.0;
  distance = abs(txtDict[0] - m);
  nearest = txtDict[0];

  for(int i = 0; i < txtDict.size(); i++){
    if(abs(txtDict[i] - m) < distance){
      nearest = txtDict[i];
      distance = abs(txtDict[i] - m);
    }
  }
  m = nearest;
}

void textonModel(vector<float> txtDict, vector<float>& models){
      for(int j = 0; j < models.size();j++){
        textonFind(txtDict, models[j]);
      }
}

// Create bins for each textonDictionary Value
void binLimits(vector<float> texDict, float* bins, int size){
  cout << "inside binLimits" << endl;
  bins[0] = 0;

  for(int i = 1;i <= size;i++){
      bins[i] = (texDict[i-1] + 0.001);
      cout << "texDict: " << i << " "<< texDict[i-1] << " becomes: " << bins[i] << endl;
  }
  bins[size+1] = 255;
}

void savetxtDict(vector<float> dict, float* binArray, int binNum){
  FileStorage fs("txtDict.xml", FileStorage::WRITE);

  // Save texton Dictionary
  cout << "saving texton dictionary" << endl;
  fs << "TextonDictionary" << "[";
    for(int i =0;i<dict.size();i++)
      fs << dict[i];
  fs << "]";

  // Save Corresponding Bin Array
  cout << "saving Bin limits" << endl;
  fs << "binArray" << "[";
    for(int j=0;j<binNum;j++){
      fs << binArray[j];
    }
  fs << "]";
  fs.release();
}

void loadTex(vector<float>& out){
  FileStorage fs("txtDict.xml", FileStorage::READ);

  FileNode n = fs["TextonDictionary"];
  if(n.type() != FileNode::SEQ){
    cout << "incorrect filetype: " << n.type() << endl;
    return;
  }

  FileNodeIterator it = n.begin(), it_end = n.end();
  int cnt =0;
  for(;it != it_end;++it){
    out.push_back((float)*it);
    cnt++;
  }
  cout << "finished reading Textons..\n\n";
  fs.release();
}

void loadBins(float* out){
  FileStorage fs("txtDict.xml", FileStorage::READ);

  FileNode n = fs["binArray"];
  if(n.type() != FileNode::SEQ){
    cout << "incorrect filetype: " << n.type() << endl;
    return;
  }

  FileNodeIterator it = n.begin(), it_end = n.end();
  int cnt =0;
  for(;it != it_end;++it){
    out[cnt] = (float)*it;
    cnt++;
  }
  cout << "finished reading Bins..\n\n";
  fs.release();
}

void loadHist(mH2& hist){
  FileStorage fs("test123.xml", FileStorage::READ);
  FileNode n = fs["ModelHistograms"];

  // Loop through Classes
  for(int i=0;i<n.size();i++){
    stringstream ss;
    ss << "Class_";
    ss << i;
    string a = ss.str();

    FileNode n1 = n[a];

    cout << "\nThis is: " << a <<  " Entering loop." << endl;

    // Loop through Each classes Models
    for(int j = 0; j < n1.size(); j++){
      stringstream ss1;
      ss1 << "Model_";
      ss1 << j;
      string b = ss1.str();

      FileNode n2 = n1[b];
      cout << "This is: " << b <<  " Entering loop." << endl;

      // Save stored Mat to mask
      FileNodeIterator it = n2.begin(), it_end = n2.end();
      for(;it != it_end;++it){
        Mat mask;
        (*it) >> hist[i][j];
      }
    }
  }
  fs.release();
}

void saveHist(mH2 hist){
  cout << "saving histogram." << endl;
  FileStorage fs("test123.xml", FileStorage::WRITE);

  int size = hist.size();
  fs << "ModelHistograms" << "{";

  for(int i=0;i<size;i++){
    stringstream ss;
    string a, b;
    a = "Class_";
    ss << i;
    b = ss.str();
    a +=b;

    fs << a << "{";

    for(int j =0;j<hist[i].size();j++){
      stringstream ss1;
      string c;
      ss1 << "Model_" << j;
      c = ss1.str();
      fs << c << "[";
      fs << hist[i][j];
      fs << "]";
    }
    fs << "}";
  }
  fs << "}";
  fs.release();
}


void makeTexDictionary(mH2 filterbank,  vector<string> classes, int n_sigmas, int n_orientations, string type){
  vector<float> texDict;
  m3 models(4, m2(8, m1(0)));
  // Create texton dict and store
  createTexDic(filterbank, classes, models, n_sigmas, n_orientations, texDict, type);

  // Sort texton Dict and round to 2dp
  sort(texDict.begin(), texDict.end());
  roundTex(texDict);
  removeDups(texDict);

  int texDictSize = texDict.size();
  float binArray[texDictSize];
  binLimits(texDict, binArray, texDictSize);

  // Store in local dir, texDict+2 to account for starting 0 and finishing 255
  savetxtDict(texDict, binArray, texDictSize+2);
}

void displayTexDict(vector<float> texDict){
  // Convert array to Mat
  int listLen = texDict.size();
  Mat tmp(listLen,1,CV_32FC1);
  textToMat(tmp, texDict);

  const string windowname1 = "texton Distribution";

  // Display Texton Histogram
  int histSize = 20;
  float range[] = { 0, 255 };
  bool uniform = true;
  Mat histImage;
  createHist(tmp, histImage, histSize ,range, uniform);
  Mat histImg = showHist(histImage, histSize);

  namedWindow(windowname1, CV_WINDOW_AUTOSIZE);
  imshow(windowname1,histImg);
  waitKey(0);
  destroyWindow(windowname1);
  cout << "Leaving DisplayTexDict. " << endl;
}

void generateModels(mH2 filterbank, vector<float> textonDictionary, const float* binArray, vector<string> classes, int n_sigmas, int n_orientations, string type){
  m3 models(4, m2(8, m1(0)));
  mH2 modelHist(10, mH1(10, Mat::zeros(80,1,CV_32FC1)));

  // Return clusters(in models) from filter responses to images in test dirs
  createTexDic(filterbank, classes, models, n_sigmas, n_orientations, textonDictionary, type);

  // loop through different classes
  for(int a = 0; a < models.size(); a++){
    // loop through different models
    for(int b = 0; b < models[a].size() && models[a][0].size() != 0; b++){
      if(models[a][b].size()!=0){
        cout << "starting this loop: " << a << " mini loop number: " << b << endl;

        textonModel(textonDictionary, models[a][b]);

        // Convert array to Mat
        Mat tmp = Mat::zeros(80, 1, CV_32FC1);
        textToMat(tmp, models[a][b]);

        // Generate texton histogram and return Mat image and display
        bool uniform = false;

        createHist(tmp, modelHist[a][b],textonDictionary.size(), binArray, uniform);
//            Mat histImg = showHist(modelHist[a][b], texDictSize);
      }
    }
  }

    saveHist(modelHist);
}

int main()
{
    // Start the window thread(essential for deleting windows)
    cvStartWindowThread();

    // dirs holding texton and model generating images
    string textonclasses[] = {"cotton", "wood", "cork", "bread"};
    vector<string> classes (textonclasses, textonclasses + sizeof(textonclasses)/sizeof(textonclasses[0]));

    int height = 10;
    int width = 10;
    int modelH = 4, modelW = 8, modelD =0;
    bool modelsGenerated = false;

    // Declare resources
    mH2 filterbank;
    const string type[] = {"train/", "test/", "novel/"};

    vector<float> sigmas;
    sigmas.push_back(1);
    sigmas.push_back(2);
    sigmas.push_back(4);
    int n_sigmas = sigmas.size();
    int n_orientations = 6;

    vector<Mat > edge, bar, rot;
    makeRFSfilters(edge, bar, rot, sigmas, n_orientations);

    // Store created filters in fitlerbank 2d vector<Mat>
    filterbank.push_back(edge);
    filterbank.push_back(bar);
    filterbank.push_back(rot);

    // --------------------- Texton Dictionary Creation ---------------------- //
    // If statement to reduce const histRange scope, allowing it to be redeclared
    if(true){

      cout << "creating texton dictionary" << endl;



      // Check for saved Dictionary xml file, generate and save new one if not found
      ifstream savedDict("txtDict.xml");
      if(!savedDict){
        cout << "\n\n\n\nCAlculating Texton Dicationary..\n\n\n";
        makeTexDictionary(filterbank, classes, n_sigmas, n_orientations, type[0]);
      }

      vector<float> textonDictionary;
      loadTex(textonDictionary);

      ifstream savedMod("test123.xml");
      if(!savedMod){
        cout << "\n\n\nCAlculating models...\n\n";
        float binArray[textonDictionary.size()];
        loadBins(binArray);
        generateModels(filterbank, textonDictionary, binArray, classes, n_sigmas, n_orientations, type[0]);
      }

      printTexDict(textonDictionary);
      displayTexDict(textonDictionary);
    }

    bool cont = false;
    do{

      cout << "\nMenu: Please enter the chosen options number \n"<< endl;
      cout << "1: ReCalculate Texton Dictionary" << endl;
      cout << "2: ReCalculate Models" << endl;
      cout << "3: model testing" << endl;
      cout << "4: exit\n" << endl;

      string tmp;
      cin >> tmp;
      if(tmp == "1"){
          cout << "\n\n-------------------Regenerating TextonDictionary and Bins--------------------\n\n";
          cout << "ReCalculating Texton Dictionary and saving\n\n";
          vector<float> textonDictionary;
          makeTexDictionary(filterbank, classes, n_sigmas, n_orientations, type[0]);
          cont = true;
      }
      else if(tmp == "2"){
        // Measure start time
        auto t1 = std::chrono::high_resolution_clock::now();
        cout << "\n\n-------------------Regenerating models--------------------\n\n";

        // Check for saved Dictionary xml file, generate and save new one if not found
        ifstream savedDict("txtDict.xml");
        if(!savedDict)
          makeTexDictionary(filterbank, classes, n_sigmas, n_orientations, type[1]);

        vector<float> textonDictionary;
        loadTex(textonDictionary);
        float binArray[textonDictionary.size()];
        loadBins(binArray);

        generateModels(filterbank, textonDictionary, binArray, classes, n_sigmas, n_orientations, type[0]);

        // Measure time efficiency
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "f() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";

        cont = true;
        modelsGenerated = true;
      }
      else if(tmp == "3"){
        cout << "\n\n-------------------Testing Image Against Models--------------------\n\n";
        // Measure start time
        auto t3 = std::chrono::high_resolution_clock::now();

        vector<float> textonDictionary;
        loadTex(textonDictionary);
        float binArray[textonDictionary.size()];
        loadBins(binArray);

        m3 models(4, m2(8, m1(0)));
        mH2 modelHist(10, mH1(10, Mat::zeros(80,1,CV_32FC1)));

        loadHist(modelHist);

        vector<float> testModel;



          Mat inputImg =  imread("../../../TEST_IMAGES/testImage/52a-scale_2_im_8_col.png", CV_LOAD_IMAGE_GRAYSCALE);
          if(!inputImg.data){
            cout << "unable to load image.." << endl;
            return -1;
          }

          vector<vector<Mat> > response;
          apply_filterbank(inputImg, filterbank, response, n_sigmas, n_orientations);
          testImgModel(response, testModel);

          cout << "Before conversion: \n\n";
          printModelsInner(testModel, 0);
          textonModel(textonDictionary, testModel);
          cout << "After conversion: \n\n";
          printModelsInner(testModel, 0);

          // Get total length of single column matrix
          int listLen = testModel.size();

          // Convert array to Mat
          Mat tmp(listLen, 1, CV_32FC1);
          textToMat(tmp, testModel);
          bool uniform = false;
          cout << "going into create Hist" << endl;
          Mat novelHist;
          createHist(tmp, novelHist, textonDictionary.size(), binArray, uniform);

          cout << "going into matchModel" << endl;

          double distance = 0.0;
          int match = 0;

          for(int i = 0; i < modelHist.size(); i++){
            cout << "here.. i: " << i << endl;
            for(int j = 0; j< modelHist[i].size(); j++){
              cout << "Inner here.. i: " << i << " j: " << j << endl;
              cout << "modelHIst is : " << modelHist[i][j].at<float>(10,10) << endl;
//              cout << "the size: " <<  modelHist[i][j].size() << " and the novelHist: " << novelHist.size() << endl;

              double tmpt = compareHist(modelHist[3][0], novelHist, CV_COMP_CHISQR);
              cout << "This is tmp: " << tmpt << endl;
              if(tmpt<distance){
                distance = tmpt;
                match = i;
                cout << "found a better match, the new distance is: " << distance << endl;
              }
                int intDistance = (double) tmpt;
                cout << "This is the int distance: " << intDistance << endl;
            }
          }
          if(match == -1){
            cout << "A match wasn't able to be found.. " << endl;
          }else {
            cout << "\n\nThis: " << classes.at(match) << " is the closest match." << endl;
          }

          int value = (double) distance;
         cout << "This is the final value: " << value << endl;

//          double distance = matchModel(novelHist, modelHist, height, width);

          Mat histImg = showHist(novelHist, textonDictionary.size());
          namedWindow("testImage", CV_WINDOW_AUTOSIZE);
          imshow("testImage", histImg);

          // Measure time efficiency
          auto t4 = std::chrono::high_resolution_clock::now();
          std::cout << "f() took "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
                    << " milliseconds\n";

          waitKey(2000);
          cvDestroyAllWindows();
          cont = true;
      }
      else if(tmp == "4"){
        cout << "exiting" << endl;
        cont = false;
      } else {
        cout << "that input was not recognised." << endl;
        cvDestroyAllWindows();
        cont = true;
      }
    }while(cont);
}
