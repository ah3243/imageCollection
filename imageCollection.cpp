#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h> // for printf()
#include <stdlib.h> // for exit() function
#include <fstream> // For videoStream
#include <dirent.h> // For accessing filesystem

using namespace std;
using namespace cv;

void menuPrint(){
  cout << "\n\n---------------------------------\n";
  cout << "Please enter:\n\n";
  cout << "'t' for Texton image storage." << endl;
  cout << "'q' to close the program." << endl;
  cout << "----------------------------------\n\n";
}

int main(){

  DIR *pdir = NULL;

  pdir = opendir("../");
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
    cout << pent->d_name << endl;
  }

  closedir(pdir);
  cout << "finished Reading Successfully.." << endl;

  VideoCapture stream(0);
  if(!stream.isOpened()){
    cout << "Video stream unable to be opened exiting.." << endl;
    return -1;
  }
  namedWindow("VideoStream", CV_WINDOW_AUTOSIZE);

  menuPrint();

  namedWindow("savedImage", CV_WINDOW_AUTOSIZE);
  Mat savedImage;
  savedImage =  Mat::zeros(640,480,CV_8UC3);

  while(true){
    Mat inputTmp;
    stream.read(inputTmp);
    imshow("VideoStream", inputTmp);
    char c = waitKey(30);

    if(c == 't'){
      cout << "capturing textons images" << endl;
      savedImage = inputTmp.clone();
      imshow("savedImage", savedImage);
      menuPrint();
    }else if(c == 'q'){
      cout << "quitting.. " << endl;
      break;
    }
  }

  return 0;
}
