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
#include <chrono>  // time measurement
#include <thread>  // time measurement
#include <boost/math/special_functions/round.hpp>

using namespace cv;

typedef unsigned int uint;
const float PI = 3.1415;

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

void  apply_filterbank(Mat &img, vector<vector<Mat> > &filterbank, vector<vector<Mat> > &response, int n_sigmas, int n_orientations) {
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
   cout << "Model:" << count << "\nIt's size is: " << v.size() <<  endl;
      for(int i = 0; i < v.size(); i++){
          cout << v[i] << " ";
      }
      cout << "\n";
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
  roundModel(models);

  waitKey(1000);
  response.clear();
}

// Generate Texton Dictionary from all imgs in sub dirs
void createTexDic(vector<vector<Mat> >& filterbank, vector<string> classes , vector<vector<vector<float> > >& models, int n_sigmas, int n_orientations, vector<float>& textonDict, string type){

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

    // FileStorage fs("../textonDictionary.xml", FileStorage::WRITE);
    // fs << "wood" << "[";
    // fs << centers;
    // fs << "]";
    // fs.release();

    doCount ++;
  }while(doCount < classesSize);
  // Print Current referenced texton dictionary
//  printTexDict(textonDict);
}

void roundTex(vector<float>& tex){
  for(int i=0;i<tex.size();i++){
      tex[i] = floorf((tex[i])*1000)/1000;
      //tex[i] = round(tex[i]);
  }
}

void removeDups(vector<float>& tex){
    set<float> v;

    // Assigning the size prevents repeatedly calculating it
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
  for( int i = 1; i < histBins; i++ ){
    // Count sum of histogram values
    countHist += inputHist.at<float>(i);

    // Draw rectangle representative of hist values in output image
    rectangle( m, Point( bin_w*(i),  hist_h) ,
                    Point( (bin_w*(i))+bin_w,  hist_h - (inputHist.at<float>(i) * scaleFactor)),
                    Scalar( 255, 255, 255),-1, 8);
  }
  return m;
}

Mat createHist(Mat& in, int histSize, const float* histRange, bool uniform){
  bool accumulate = false;

  int h = 600, w = 450;
  int total = (h*w)*100;
  Mat output = Mat::zeros(h,w,CV_32FC1);

  // Compute the histograms:
  calcHist( &in, 1, 0, Mat(), output, 1, &histSize, &histRange, uniform, accumulate );
  return output;
}

// Takes in Vector<float> and converts to Mat<float>
void textToMat(Mat& tmp, vector<float> texDict){
  for(int i = 0; i < texDict.size();i++){
    tmp.at<float>(i) = texDict.at(i);
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

  for(int i = 1;i < size;i++){
      bins[i] = (texDict[i-1] + 0.001);
      cout << "texDict: " << i << " "<< texDict[i-1] << " becomes: " << bins[i] << endl;
  }
  bins[size] = 255;
}

void allocate2dMat(vector<vector<Mat> >& in, int height, int width){
  for(int i = 0;i < height;i++){
    in.push_back(vector<Mat>());
    for(int j = 0;j < width;j++){
      in[i].push_back(Mat(400,600,CV_8UC1));
    }
  }
}

void clear2dvect(vector<vector<vector<float> > >& in, int height, int width){
  for(int i = 0;i < in.size();i++){
    for(int j = 0;j < in[i].size();j++){
      in[i][j].clear();
    }
  }
  cout << "done clearing float Vectors.." << endl;
}

void clear2dMat(vector<vector<Mat> >& in, int height, int width){
  for(int i = 0;i < height;i++){
    in[i].clear();
  }
  cout << "done clearing Mats.." << endl;
}

// double matchModel(Mat novel, vector<vector<Mat> > models, int height, int width){
//   double distance = 0.0;
//   cout << "inside matchModel " << endl;
//   // for(int i = 0; i < models.size(); i++){
//   //   for(int j = 0; !models[i][j].empty(); j++){
//       // cout << "Model[i]:" << i << " Size: " << models[i].size() << endl;
//       // cout << "Model[i][j]: " << j << " size: " << models[i][j].size() << endl;
//       cout << "going to compare.. models[3][0]" << models[3][0] << endl;
//
//       //double tmp = compareHist(models[3][0], novel, CV_COMP_CHISQR);
// //      cout << "This is tmp: " << tmp << endl;
//   //  }
//   //}
//   cout << "leaving match Model.. this is distance: " << distance << endl;
//   return distance;
// }

int main()
{
    // Start the window thread(essential for deleting windows)
    cvStartWindowThread();

    int height = 10;
    int width = 10;

    // Declare resources
    vector<vector<vector<float> > > models(4,vector<vector<float> >(8,vector<float>(0)));
    vector<vector<Mat> > modelHist;
    vector<vector<Mat > > filterbank;
    vector<float> textonDictionary;
    const string type[] = {"train/", "test/", "novel/"};

    // dirs holding texton and model generating images
    string textonclasses[] = {"cotton", "wood", "cork", "bread"};
    vector<string> classes (textonclasses, textonclasses + sizeof(textonclasses)/sizeof(textonclasses[0]));

    allocate2dMat(modelHist, height, width);

    vector<float> sigmas;
    sigmas.push_back(1);
    sigmas.push_back(2);
    sigmas.push_back(4);
    int n_sigmas = sigmas.size();
    int n_orientations = 6;
    vector<Mat > edge, bar, rot;
    makeRFSfilters(edge, bar, rot, sigmas, n_orientations);

    // // plot filters
    // drawing(edge, bar, rot, n_sigmas, n_orientations);

    filterbank.push_back(edge);
    filterbank.push_back(bar);
    filterbank.push_back(rot);

    // --------------------- Texton Dictionary Creation ---------------------- //
    // If statement to reduce const histRange scope, allowing it to be redeclared
    if(true){

      cout << "creating texton dictionary" << endl;

      textonDictionary.clear();
      cout << "texton Dictionary cleared. " << endl;

      // Create texton dict and store
      createTexDic(filterbank, classes, models, n_sigmas, n_orientations, textonDictionary, type[0]);

      // Sort texton Dict and round to 2dp
      sort(textonDictionary.begin(), textonDictionary.end());
      roundTex(textonDictionary);
      removeDups(textonDictionary);

      // Print texton dict to console
      cout << "printing texton dictionary from main" << endl;
    //  printTexDict(textonDictionary);

      // Convert array to Mat
      Mat tmp(80, 80, CV_32FC1);
      textToMat(tmp, textonDictionary);
      const string windowname1 = "texton Distribution";

      // Generate texton histogram and return Mat image and display
      int histSize = 20;
      float range[] = { 0, 255 };
      bool uniform = true;
      Mat histImage = createHist(tmp, histSize ,range, uniform);
      Mat histImg = showHist(histImage, histSize);

      namedWindow(windowname1, CV_WINDOW_AUTOSIZE);
      imshow(windowname1,histImg);
      waitKey(0);
      destroyWindow(windowname1);
    }
    int texDictSize = textonDictionary.size();
    float binArray[texDictSize];
    binLimits(textonDictionary, binArray, texDictSize);

    bool cont;
    do{

      cout << "\nMenu: Please enter the chosen options number \n"<< endl;
      cout << "1: texton dictionary creation" << endl;
      cout << "2: model creation" << endl;
      cout << "3: model testing" << endl;
      cout << "4: exit\n" << endl;

      string tmp;
      cin >> tmp;

      // --------------------- Model Creation ---------------------- //
      if(tmp == "2"){

        // If textonDictionary is empty redirect to main
        if(!textonDictionary.empty()){
          cout << "Creating models" << endl;

          // Measure start time
          auto t1 = std::chrono::high_resolution_clock::now();

          cout << "entering clear vectors" << endl;
          clear2dvect(models, 10, 10);
          clear2dMat(modelHist, 10, 10);

          // Return clusters(in models) from filter responses to images in test dirs
          createTexDic(filterbank, classes, models, n_sigmas, n_orientations, textonDictionary, type[1]);

          // Loop through the different model classes
          cout << "This is model.size(): " << models.size() << " this is the model[0].size():" << models[0].size() << " This is the models[0][0].size(): " << models[0][0].size() << endl;

          // loop through different classes
          for(int a = 0; a < models.size(); a++){

            // loop through different models
            for(int b = 0; b < models[a].size() && models[a][0].size() != 0; b++){
  //          cout << "This is the size before if: " << models[a][0].size() << endl;
              if(models[a][b].size()!=0){
                cout << "starting this loop: " << a << " mini loop number: " << b << endl;
                textonModel(textonDictionary, models[a][b]);
                printModels(models[a]);

                // Convert array to Mat
                Mat tmp(120, 120, CV_32FC1);
                textToMat(tmp, models[a][b]);
                const string windowname2 = "models...";

                // Generate texton histogram and return Mat image and display
                bool uniform = false;

                modelHist[a][b] = createHist(tmp, texDictSize, binArray, uniform);
                Mat histImg = showHist(modelHist[a][b], texDictSize);

                cout << "outside of createHist going into imshow()\n\n";
                namedWindow(windowname2, CV_WINDOW_AUTOSIZE);
                imshow(windowname2,histImg);
                waitKey();
                destroyWindow(windowname2);
              }
            }
          }

          // Measure time efficiency
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "f() took "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                    << " milliseconds\n";
          cont = true;
        }
        else{
          cout << "\nYou must create the texton library before model creation." << endl;
          waitKey(1000);
        }
      }

      // --------------------- Test Novel Image ---------------------- //
      else if(tmp == "3"){
        cout << "Entering test mode" << endl;
        vector<float> testModel;

        // If textonDictionary is empty redirect to main
        if(!textonDictionary.empty()){
          // Measure start time
          auto t3 = std::chrono::high_resolution_clock::now();

          vector<vector<Mat> > response;

          Mat inputImg =  imread("../../../TEST_IMAGES/testImage/52a-scale_2_im_8_col.png", CV_LOAD_IMAGE_GRAYSCALE);
          if(!inputImg.data){
            cout << "unable to load image.." << endl;
            return -1;
          }

          namedWindow("testing", CV_WINDOW_AUTOSIZE);
          imshow("testing", inputImg);

          apply_filterbank(inputImg, filterbank, response, n_sigmas, n_orientations);
          testImgModel(response, testModel);

          cout << "Before conversion: \n\n";
          printModelsInner(testModel, 0);
          textonModel(textonDictionary, testModel);
          cout << "After conversion: \n\n";
          printModelsInner(testModel, 0);

          // Convert array to Mat
          Mat tmp(120, 120, CV_32FC1);
          textToMat(tmp, testModel);
          bool uniform = false;
          cout << "going into create Hist" << endl;
          Mat novelHist = createHist(tmp, texDictSize, binArray, uniform);

          cout << "going into matchModel" << endl;
//          double tmpdouble = compareHist(modelHist[3][0], novelHist, CV_COMP_CHISQR);

          double distance = 0.0;
          cout << "inside matchModel " << endl;
          for(int i = 0; i < modelHist.size(); i++){
            for(int j = 0; !modelHist[i][j].empty(); j++){
              double tmpt = compareHist(modelHist[i][j], novelHist, CV_COMP_CHISQR);
              cout << "This is tmp: " << tmpt << endl;
              if(tmpt<distance){
                distance = tmpt;
                cout << "found a larger one: " << distance << endl;
              }
                int intDistance = (double) tmpt;
                cout << "This is the int distance: " << intDistance << endl;
            }
          }
          //int value = (double) distance;
//          cout << "This is the final value: " << value << endl;

//          double distance = matchModel(novelHist, modelHist, height, width);

          Mat histImg = showHist(novelHist, texDictSize);
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
        else{
          cout << "\nYou must create the texton library before model creation." << endl;
          waitKey(1000);
        }

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
