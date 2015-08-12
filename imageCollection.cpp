#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream> // For videoStream
#include <iostream>
#include <cstdio> // for printf()
#include <dirent.h> // For accessing filesystem
#include <boost/filesystem.hpp>
#include <sstream>
#include <boost/algorithm/string.hpp> // for to_lower function in main
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
  cout << "Please enter the keyword for you option:\n\n";
  cout << "0. Texton" << endl;
  cout << "1. Class" << endl;
  cout << "2. Novel" << endl;
  cout << "3. Quit" << endl;
  cout << "----------------------------------\n\n";
}

void extractClsNme(string &nme){
  string tmp = nme;
  string delim[]= {"/","_", "."};
  int start = tmp.find_last_of(delim[0])+1;
  int end = tmp.find_last_of(delim[1]);
  if(end>0){
    nme =  tmp.substr(start,end-start);
  }else{
    end = tmp.find_last_of(delim[2]);
    nme = tmp.substr(start,end-start);
  }
}
void extractFullNme(string &nme){
  string tmp = nme;
  string delim[]= {"/", "."};
  int start = tmp.find_last_of(delim[0])+1;
  int end = tmp.find_last_of(delim[1]);
  nme = tmp.substr(start,end-start);
}

int getSuffix(string p){
  int suf;
  string tmp;
  suf = p.find_last_of("_");
  if(suf>0){
    tmp = p.substr(suf+1, p.length());
    return atoi(tmp.c_str());
  }else{
    return 0;
  }
}

int getHighestSuffix(path p, string cls){
  if(!exists(p) || !is_directory(p)){
    cout << "The entered path was not valid. Returning." << endl;
    return -1;
  }
  string fullNme, clsRoot;
  int highVal = 0, tmpVal;
  directory_iterator itr_end;
  for(directory_iterator itr(p);itr != itr_end;++itr){
    fullNme = itr -> path().string();
    clsRoot = fullNme;
    extractFullNme(fullNme);
    extractClsNme(clsRoot);
    if(clsRoot.compare(cls) == 0){
      tmpVal = getSuffix(fullNme);
      if(highVal < tmpVal){
        highVal = tmpVal;
      }
    }
  }
  return highVal;
}

void getUniqueClassNme(path p, vector<string>& classes){
  if(!exists(p) || !is_directory(p)){
    cout << "the entered path was not valid. Returning." << endl;
    exit(-1);
  }

  string nme;
  directory_iterator itr_end;
  for(directory_iterator itr(p); itr != itr_end; ++itr){
    int unique = 0;
    nme = itr -> path().string();
    extractClsNme(nme);

    for(int i=0;i<classes.size();i++){
      if(nme.compare(classes[i])==0){
        unique = 1;
      }
    }
    if(unique == 0 ){
      classes.push_back(nme);
    }
  }
}

void printClasses(vector<string> s){
  cout << "\n\nBelow are your current Classes:\n";
  for(int i;i<s.size();i++){
    cout << i << ": " << s[i] << endl;
  }
  cout << "\n";
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
    h = ((img.rows-200)/2);
    w = ((img.cols-200)/2);
    size = 200;

    Mat cropped = img(Rect(w,h,200,200));

    out = cropped.clone();
}

void saveImage(string path, string cls , int num, vector<Mat>& img){
  string type = ".png", delim = "_", dir;
  if(!exists(path) || !is_directory(path)){
    cout << "Path was not valid. Returning. " << path << endl;
    return;
  }
  for(int i=0;i<img.size();i++){
    stringstream ss;
    string a = path;
    ss << cls << delim << num+i << type;
    cout << "This is the num: " << num << endl;
    a += ss.str();
    cout << "This is the name: " << a << endl;
    cout << "this is the size(): " << img[i].size() << endl;
    cout << "and the number: " << img.size() << endl;
    imwrite(a, img[i]);
  }
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
      string name = tmp;
      extractClsNme(name);

      // remove if it matches target, if no target the remove all
      if(cls.size() == 0 || name.compare(cls)==0){
        fs::remove(tmp);
      }
    }
    cout << "\n";
  }
}

void printgetImageMenu(){
  cout << "\nGet Image Menu:\nPlease enter the number of your chosen option\n";
  cout << "0 Capture Image\n";
  cout << "1 Save and Quit\n";
  cout << "2 Discard and Quit\n";
  cout << "\n";
}

void getImages(vector<Mat>& matArr){
  printgetImageMenu();
  vector<Mat> local;

  VideoCapture stream(0);
  if(!stream.isOpened()){
    cout << "Video stream unable to be opened exiting.." << endl;
    exit(0);
  }
  namedWindow("VideoStream",CV_WINDOW_AUTOSIZE);
  Mat out;
  while(true){
    Mat inputTmp;
    stream.read(inputTmp);
    int h, w, size;
    h = ((inputTmp.rows-200)/2);
    w = ((inputTmp.cols-200)/2);
    size = 200;

    imshow("VideoStream", inputTmp);
    rectangle(inputTmp, Point(w,h), Point(w+200, h+200),Scalar(0,0,255), 2, 8);
    char c = waitKey(50);
    switch (c) {
      case '0':
        cout << "Capturing Image\n";
        out = Scalar(0);
        cropImage(inputTmp, out);
        local.push_back(out);
        break;
      case '1':
        cout << "Saving and Returning\n";
        for(int i=0;i<local.size();i++)
          matArr.push_back(local[i]);
    //      cvDestroyAllWindows();
        return;
      case '2':
        cout << "Discarding and Returning\n";
        return;
    }
  }
  cvDestroyWindow("VideoStream");
}

void retnFileNmes(path p, string name, vector<string>& matches){
  if(exists(p)){
    if(is_regular_file(p)){
      cout << "This is a single file not directory. Returning.\n";
    }else if(is_directory(p)){
      int c;
      directory_iterator dir_end;
      for(directory_iterator itr(p);itr != dir_end;++itr){
        string tmp = itr -> path().string();
        string cls = tmp;
        extractClsNme(cls);
        extractFullNme(tmp);
        // Add to return vector if it matches class or there was no class input
        if(name.size()==0 || cls.compare(name)==0){
          matches.push_back(tmp);
        }
      }
    }
  }
}

void printNovelImgMenu(){
  cout << "\nNovel Image Menu:\nPlease enter the number of your chosen option\n";
  cout << "0 List number of Novel images\n";
  cout << "1 Collect new Novel Images\n";
  cout << "2 Clear all Novel Images\n";
  cout << "q Return to Main Menu\n";
}

void novelImgHandler(){
  cout << "\n........Entering textonHandler........\n\n";
   cvStartWindowThread(); // Start the window thread(essential for deleting windows)
  string novel = "../../../TEST_IMAGES/CapturedImgs/novel/";
  vector<Mat> imgArr;
  vector<string> fileNmes;
  printNovelImgMenu();
  while(true){
    char c;
    cin >> c;
    switch (c) {
      case '0':
        cout << "Listing Number of Novel images\n";
        retnFileNmes(novel, "", fileNmes);
        cout << "There are currently: " << fileNmes.size() << " in the dir." << endl;
        printNovelImgMenu();
        break;
      case '1':
        cout << "Starting Novel image collection\n";
        imgArr.clear();
        getImages(imgArr);
        saveImage(novel, "novelImg", 0, imgArr);
        printNovelImgMenu();
        break;
      case '2':
        cout << "Clearing All Novel images\n";
        clearDir(novel);
        printNovelImgMenu();
        break;
      case 'q':
        cout << "\nExiting to main\n";
        return;
    }
  }
}

void printTextonMenu(){
  cout << "\nTexton Menu:\nPlease enter the number of your chosen option\n";
  cout << "0 List number of Textons\n";
  cout << "1 Add new texton images\n";
  cout << "2 Clear all stored Textons\n";
  cout << "q Return to Main Menu\n";
  cout << "\n";
}

void textonHandler(){
  cout << "\n........Entering textonHandler........\n\n";
  printTextonMenu();
  while(true){
    char c = waitKey();
    switch(c){
      case 1:
        cout << "Storing a Texton image" << endl;
//        Mat savedImage = inputTmp.clone();
//        saveImage(texton, "", tex, savedImage);
        printTextonMenu();
      case 2:
        cout << "Removing all Stored Textons" << endl;
        clearDir("./textons");
        printTextonMenu();
      case 'q':
        cout << "\nExiting to main\n";
        return;
    }
  }
}

void printClassMenu(){
  cout << "\nClass Menu:\nPlease enter the number of your chosen option\n";
  cout << "0 List all classes\n";
  cout << "1 Add a Class\n";
  cout << "2 Append to a Class\n";
  cout << "3 Remove a Class\n";
  cout << "4 Remove all Stored Classes\n";
  cout << "q Return to Main Menu\n";
  cout << "\n";
}

void classHandler(){
  cout << "\n........Entering classHandler........\n\n";

  string clsPath = "../../../TEST_IMAGES/CapturedImgs/classes/";
  string cls;
  int highVal;
  vector<Mat> imgArr;
  vector<string> clssNmes;
  printClassMenu();
  while(true){
    char c;
    cin >> c;
    switch (c) {
      case '0':
        cout << "Listing Classes" << endl;
        getUniqueClassNme(clsPath, clssNmes);
        printClasses(clssNmes);
        cout << "THis is the total number of class names: " << clssNmes.size() << endl;
        printClassMenu();
        break;
      case '1':
        cout << "Adding a Class" << endl;
        cout << "\nPlease input the name of the class." << endl;
        cls = "";
        cin >> cls;
        getImages(imgArr);
        cout << "This is the arrSize: " << imgArr.size() << endl;
        saveImage(clsPath, cls, 0, imgArr);
        imgArr.clear();
        printClassMenu();
        break;
      case '2':
        cout << "Appending to a Class" << endl;
        cout << "\nPlease input the name of the class." << endl;
        cls = "";
        cin >> cls;
        highVal =  getHighestSuffix(clsPath, cls);
        getImages(imgArr);
        saveImage(clsPath, cls, highVal, imgArr);
        imgArr.clear();
        printClassMenu();
        break;
      case '3':
        cout << "Removing a Class" << endl;
        cout << "\nWhich class would you like to remove:" << endl;
        cls = "";
        cin >> cls;
        clearClass(cls);
        printClassMenu();
        break;
      case '4':
        cout << "Removing all Classes" << endl;
        clearClass("");
        printClassMenu();
        break;
      case 'q':
          cout << "\nExiting to main\n";
        return;
    }
  }
}

int main(int argc, char** argv){
  generateDirs();

  string capture;
  while(true){
    menuPrint();

    cin >> capture;
    boost::algorithm::to_lower(capture);
    cin.ignore(); // only collect a single word

    if(capture.compare("texton")==0){
      cout << "trex\n";
    }else if(capture.compare("class")==0){
      cout << "class\n";
      classHandler();
    }else if(capture.compare("novel")==0){
      cout << "novelImgs";
      novelImgHandler();
    }else if(capture.compare("quit")==0){
      cout << "quitting\n";
      return 0;
    }else{
      cout << "Your input of: " << capture << " was not recognised.\n" << endl;
    }
  }
  return 0;
}
