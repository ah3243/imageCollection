#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream> // For videoStream
#include <iostream>
#include <cstdio> // for printf()
#include <dirent.h> // For accessing filesystem
#include <boost/filesystem.hpp>
#include <sstream>
// #include <stdlib.h> // for exit() function


using namespace boost::filesystem;
using namespace std;
using namespace cv;

void menuPrint(){
  cout << "\n\n---------------------------------\n";
  cout << "Please enter:\n\n";
  cout << "'1' for storing a Texton image" << endl;
  cout << "'2' for deleting all current stored Texton images" << endl;
  cout << "'3' for storing a Model image" << endl;
  cout << "'4' for deleting all current stored Model images" << endl;
  cout << "'5' for Novel Image Capture" << endl;
  cout << "'9' to Close the program." << endl;
  cout << "----------------------------------\n\n";
}

void listDir(const char *inPath, vector<Mat>& dirFiles){
  DIR *pdir = NULL;
  cout << "inpath : " << inPath << endl;
  pdir = opendir(inPath);
  // Check that dir was initialised correctly
  if(pdir == NULL){
    cout << "ERROR! unable to open directory, exiting." << endl;
    exit(1);
  }
  struct dirent *pent = NULL;

  // Continue as long as there are still values in the dir list
  while (pent = readdir(pdir)){
    if(pdir==NULL){
      cout << "ERROR! dir was not initialised correctly, exiting." << endl;
      exit(3);
    }
    stringstream ss;
    ss << inPath;
    ss << pent->d_name;
    string a = ss.str();
    Mat tmp = imread(a, CV_BGR2GRAY);
    dirFiles.push_back(tmp);
    if(!tmp.data){
      cout << "unable to read image.." << endl;
    }
  }
  closedir(pdir);
  cout << "finished Reading Successfully.." << endl;
}

// Generate dirs for models, textons and novel images if the dont exist
void generateDirs(){
  vector<path> p;
  p.push_back("./textons");
  p.push_back("./models");
  p.push_back("./novel");

  for(int i=0;i<3;i++){
    if(!exists(p[i]))
      boost::filesystem::create_directory(p[i]);
  }
}

void  cropImage(Mat img, Mat& out){
    int h, w, size;
    h = ((img.rows-20)/2);
    w = ((img.cols-20)/2);
    size = 20;

    Mat cropped = img(Rect(w,h,20,20));

    out = cropped.clone();
    namedWindow("SavedImg", CV_WINDOW_AUTOSIZE);
    imshow("SavedImg", cropped);
}

void saveImage(string path, int num, Mat& img){
  string type = ".png";
  stringstream ss;
  ss << path << num << type;
  string a = ss.str();
  cout << "This is the name: " << a << endl;

  Mat out;
  cropImage(img, out);

  cout << "here.." << endl;
  imwrite(a, out);
}

void clearDir(string a){
  path p(a);
  remove_all(p);
  generateDirs();
}

int main(int argc, char** argv){
  string basePath = "../../../TEST_IMAGES/kth-tips/";
  vector<string> extPath = {"bread/", "cotton/", "cork/"};
  typedef vector<Mat> mV;
  mV txtons(100, Mat(200,200, CV_32FC1));
  mV bread (10,Mat(200,200,CV_32FC1));
  mV cotton (10, Mat(200,200,CV_32FC1));
  mV cork (10, Mat(200,200,CV_32FC1));

  int cnt = 0;
  for(int i=0;i<4;i++){
    stringstream ss;
    ss << basePath;
    ss << extPath[cnt];
    ss << "train/";
    string b = ss.str();
    const char* a = b.c_str();
    cout << "this is the path.. " << a << endl;

    switch(i){
      case 0:
      cout << "number: " << i << endl;
      listDir("../../../TEST_IMAGES/kth-tips/textons/",txtons);
      cnt--;
      break;

      case 1:
      cout << "number: " << i << endl;
      listDir(a,bread);
      break;

      case 2:
      cout << "number: " << i << endl;
      listDir(a,cotton);
      break;

      case 3:
      cout << "number: " << i << endl;
      listDir(a,cork);
      break;
    }
    cnt ++;
  }
  cout << "This is the size of textons: " << txtons[0].size() << ", bread: " << bread.size() << ", cotton: " << cotton.size() << ", cork: " << cork.size() << endl;

  // The most current image suffix for model and texton images
  int mod = 0, tex = 0;
  // Model and texton paths
  string model, texton, novel;
  model = "./models/image";
  texton = "./textons/tex";
  novel = "./novel/current";

  VideoCapture stream(0);
  if(!stream.isOpened()){
    cout << "Video stream unable to be opened exiting.." << endl;
    return -1;
  }

  namedWindow("VideoStream", CV_WINDOW_AUTOSIZE);

  menuPrint();
  generateDirs();

  while(true){
    stringstream ss;

    Mat inputTmp;
    stream.read(inputTmp);
    int h, w, size;
    h = ((inputTmp.rows-200)/2);
    w = ((inputTmp.cols-200)/2);
    size = 200;

    rectangle(inputTmp, Point(w,h), Point(w+200, h+200),Scalar(0,0,255), 2, 8);
    imshow("VideoStream", inputTmp);

    char c = waitKey(30);

    if(c == '1'){
      cout << "capturing textons images" << endl;
      Mat savedImage = inputTmp.clone();
      saveImage(texton, tex, savedImage);
      menuPrint();
      tex++;
    }else if(c == '2'){
      cout << "clearing old textons" << endl;
      clearDir("./textons");
      menuPrint();
      tex = 0;
    }else if(c == '3'){
      cout << "capturing modelImages" << endl;
      Mat savedImage = inputTmp.clone();
      saveImage(model, mod, savedImage);
      menuPrint();
      mod++;
    }else if(c == '4'){
      cout << "clearing old models" << endl;
      clearDir("./models");
      menuPrint();
      mod = 0;
    }else if(c == '5'){
      cout << "collecting novel image"<< endl;
      clearDir("./novel");
      Mat savedImage = inputTmp.clone();
      saveImage(novel, 0, savedImage);
      menuPrint();
    }else if(c == '9'){
      cout << "quitting.. " << endl;
      break;
    }else if((int)c != -1){
      cout << "That input was not recognised." << endl;
      menuPrint();
    }
  }

  return 0;
}
