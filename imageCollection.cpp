#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>

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

  VideoCapture stream(0);
  if(!stream.isOpened()){
    cout << "Video stream unable to be opened exiting.." << endl;
    return -1;
  }
  namedWindow("VideoStream", CV_WINDOW_AUTOSIZE);

  menuPrint();

  while(true){
    Mat inputTmp;
    stream.read(inputTmp);
    imshow("VideoStream", inputTmp);
    char c = waitKey(30);

    if(c == 'q'){
      cout << "quitting.. " << endl;
      break;
    }else if(c == 't'){
      cout << "capturing textons images" << endl;
      menuPrint();
    }

  }


  return 1;
}
