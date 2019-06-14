#include <iostream>
#include "traffic_sign_classifier.h"
#include <string>
#include <fstream>
using namespace std;
const char* names[9] = {"20", "30", "50", "60", "70", "80", "100", "120", "unknown"};

int main(int argc, char** argv)
{
    fstream file;
    file.open(argv[1], ios::in);
    if(!file){
        std::cout << "no image file" << std::endl;
        file.close();
        return 1;
    }
    file.close();

    file.open(argv[2], ios::in);
    if(!file){
        std::cout << "no tensorrt path" << std::endl;
        file.close();
        return 1;
    }
    file.close();
    cv::Mat img = cv::imread( argv[1], cv::IMREAD_UNCHANGED );
    TrafficSignClassifier net(argv[2]);
    float ms = net.inference(img);
    cout<<"inference time: "<<ms<<"  pred:"<<names[net.result]<<endl;
    return 0;
}