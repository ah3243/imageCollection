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

void errorFunc(string input){
  cerr << "\n\nERROR!: " << input << "\nExiting.\n\n";
  exit(-1);
}

void warnFunc(string input){
  cerr << "\nWARNING!: " << input << endl;
};


void menuPrint(){
  cout << "\n\n---------------------------------\n";
  cout << "Please enter:\n\n";
  cout << "'1' for Storing a Texton image" << endl;
  cout << "'2' for Removing all stored Texton images" << endl;

  cout << "'3' for Add to a Class" << endl;
  cout << "'4' for Removing a stored Class" << endl;
  cout << "'5' for Removing all current stored Classes" << endl;

  cout << "'6' for Single centre region Novel Image Capture" << endl;
  cout << "'7' for Plus config Novel Image Capture" << endl;
  cout << "'8' to Close the program." << endl;
  cout << "----------------------------------\n\n";
}

void getClasses(const char *inPath, map<string, vector<Mat> >& classes){
  DIR *pdir = NULL;
  cout << "inpath : " << inPath << endl;
  pdir = opendir(inPath);
  // Check that dir was initialised correctly
  if(pdir == NULL){
    errorFunc("Unable to open directory.");
  }
  struct dirent *pent = NULL;

  // Continue as long as there are still values in the dir list
  while (pent = readdir(pdir)){
    if(pdir==NULL){
      errorFunc("Dir was not initialised correctly.");
    }

    // Extract and save img filename without extension
    stringstream ss;
    ss << pent->d_name;
    string fileNme =  ss.str();

    // If not file then continue iteration
    string dot[] = {".", ".."};
    if(fileNme.compare(dot[0]) == 0 || fileNme.compare(dot[1]) == 0){
      continue;
    }
    // Get root Class name
    string cls;
    int lastIdx = fileNme.find_last_of(".");
    int classmk = fileNme.find_last_of("_");
    if(classmk>0){
      cls = fileNme.substr(0, classmk);
    } else{
      cls = fileNme.substr(0, lastIdx);
    }

    ss.str("");
    ss << inPath;
    ss << pent->d_name;
    string a = ss.str();
    Mat tmp = imread(a, CV_BGR2GRAY);
    if(tmp.data){
      classes[cls].push_back(tmp);
    }else{
      warnFunc("Unable to read image.");
    }
  }
  closedir(pdir);
  cout << "finished Reading Classes.." << endl;
}

void getTexImgs(const char *inPath, vector<Mat>& textDict){
  DIR *pdir = NULL;
  cout << "inpath : " << inPath << endl;
  pdir = opendir(inPath);
  // Check that dir was initialised correctly
  if(pdir == NULL){
    errorFunc("Unable to open directory.");
  }
  struct dirent *pent = NULL;

  // Continue as long as there are still values in the dir list
  while (pent = readdir(pdir)){
    if(pdir==NULL){
      errorFunc("Dir was not initialised correctly.");
    }
    stringstream ss;
    ss << inPath;
    ss << pent->d_name;
    string a = ss.str();
    Mat tmp = imread(a, CV_BGR2GRAY);
    if(tmp.data){
      textDict.push_back(tmp);
    }else{
      warnFunc("Unable to read image.");
    }
  }
  closedir(pdir);
  cout << "finished Reading TextonImgs.." << endl;
}

// Generate dirs for models, textons and novel images if the dont exist
void generateDirs(){
  vector<path> p;
  p.push_back("../../../TEST_IMAGES/CapturedImgs/textons");
  p.push_back("../../../TEST_IMAGES/CapturedImgs/classes");
  p.push_back("../../../TEST_IMAGES/CapturedImgs/novel");

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

void saveImage(string path, string cls , int num, Mat& img){
  string type = ".png";
  stringstream ss;
  ss << path << cls << num << type;
  string a = ss.str();
  cout << "This is the name: " << a << endl;

  Mat out;
  cropImage(img, out);
  imwrite(a, out);
}

void clearDir(string a){
  path p(a);
  remove_all(p);
  generateDirs();
}

int clearClass(string cls){
  namespace fs = boost::filesystem;
  path classes("../../../TEST_IMAGES/CapturedImgs/classes/");
  directory_iterator end_iter;

  if(fs::exists(classes) && fs::is_directory(classes)){
    for(directory_iterator it(classes);it != end_iter;++it){
      string tmp = it-> path().string();
      string delim[]= {"/","_", "."};
      string name;
      int start = tmp.find_last_of(delim[0])+1;
      int end = tmp.find_last_of(delim[1]);
      cout << "end: " << end;
      if(end>0){
        cout << "insidde..";

        name =  tmp.substr(start,end-start);
      }else{
        end = tmp.find_last_of(delim[2]);
        name = tmp.substr(start,end-start);
      }
      cout << "These are the names: " << name << endl;
      if(name.compare(cls)==0){
        fs::remove(tmp);
      }
    }
    cout << "\n";
  }
}

int main(int argc, char** argv){
  string basePath = "../../../TEST_IMAGES/CapturedImgs/";
  map<string, vector<Mat> > classes;
  vector<Mat> txtons;

  generateDirs();

  // Import texton dictionary
  getTexImgs("../../../TEST_IMAGES/CapturedImgs/textons/", txtons);

  // Import models
  getClasses("../../../TEST_IMAGES/CapturedImgs/classes/",classes);

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
      cout << "Storing a Texton image" << endl;
      Mat savedImage = inputTmp.clone();
      saveImage(texton, "", tex, savedImage);
      menuPrint();
      tex++;
    }else if(c == '2'){
      cout << "Removing all Stored Textons" << endl;
      clearDir("./textons");
      menuPrint();
      tex = 0;
    }else if(c == '3'){
      cout << "Adding to a Class" << endl;
      cout << "\nPlease input the name of the class." << endl;
      string cls;
      cin >> cls;

      Mat savedImage = inputTmp.clone();

      saveImage(model, cls , mod, savedImage);
      menuPrint();
    }else if(c == '4'){
      cout << "Removing a Stored Class" << endl;
      cout << "\nWhich class would you like to remove:" << endl;
      string cls;
      cin >> cls;
      clearClass(cls);
      menuPrint();
      mod = 0;
    }else if(c == '5'){
      cout << "Removing all stored Classes"<< endl;
      clearDir("./classes");
      menuPrint();
      mod = 0;
    }else if(c == '6'){
      cout << "collecting novel image"<< endl;
      clearDir("./novel");
      Mat savedImage = inputTmp.clone();
      saveImage(novel, "", 0, savedImage);
      menuPrint();
    }else if(c == '8'){
      cout << "quitting.. " << endl;
      break;
    }else if((int)c != -1){
      cout << "That input was not recognised." << endl;
      menuPrint();
    }
  }

  return 0;
}
