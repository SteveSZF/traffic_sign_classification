#ifndef __TRAFFIC_SIGN_CLASSIFIER_H__
#define __TRAFFIC_SIGN_CLASSIFIER_H__

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <time.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core.hpp>

#include <opencv2/core/opengl.hpp>

#include <opencv2/highgui.hpp>
//#include <opencv2/cudawarping.hpp>
//#include <opencv2/cudacodec.hpp>
#include <npp.h>

using namespace nvinfer1;
using namespace plugin;

#define MAX_IM (300*300)

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

class PreProcessor
{
public:
    PreProcessor(int out_w, int out_h)
    :out_w(out_w), out_h(out_h)
    {
        CHECK(cudaMalloc(&gpu_buf_in, MAX_IM*3));
        CHECK(cudaMalloc(&gpu_buf_8u_c, out_w*out_h*3));
        CHECK(cudaMalloc(&gpu_buf_32f_c, out_w*out_h*3*sizeof(float)));
        //CHECK(cudaMalloc(&gpu_buf_32f_p, out_w*out_h*3*sizeof(float)));
    }
    PreProcessor(){}
    ~PreProcessor()
    {
        CHECK(cudaFree(gpu_buf_32f_c));
        CHECK(cudaFree(gpu_buf_8u_c));
        CHECK(cudaFree(gpu_buf_in));
    }
    void init(int out_w, int out_h)
    {
        this->out_w = out_w;
        this->out_h = out_h;
        CHECK(cudaMalloc(&gpu_buf_in, MAX_IM*3));
        CHECK(cudaMalloc(&gpu_buf_8u_c, out_w*out_h*3));
        CHECK(cudaMalloc(&gpu_buf_32f_c, out_w*out_h*3*sizeof(float)));
    }
    void process(cv::Mat& im, void* gpu_buf)
    {
        in_w = im.cols;
        in_h = im.rows;
        NppiSize srcSize = {in_w, in_h};
        NppiSize dstSize = {out_w, out_h};
        NppiRect srcROI = {0, 0, in_w, in_h};
        NppiRect dstROI = {0, 0, out_w, out_h};
        Npp32f* r_plane = (Npp32f*)(gpu_buf);
        Npp32f* g_plane = (Npp32f*)(gpu_buf + out_w*out_h*sizeof(float) );
        Npp32f* b_plane = (Npp32f*)(gpu_buf + out_w*out_h*2*sizeof(float) );
        Npp32f* rgb_planes[3] = {r_plane, g_plane, b_plane};
        cudaMemset(gpu_buf_in, 0, MAX_IM*3);
        cudaMemcpy(gpu_buf_in, im.data, in_w*in_h*3*sizeof(uchar), cudaMemcpyHostToDevice);
        nppiSwapChannels_8u_C3IR((Npp8u*)gpu_buf_in, in_w*3, srcSize, Order);
        nppiResize_8u_C3R((Npp8u*)gpu_buf_in, in_w*3*sizeof(uchar), srcSize, srcROI, 
                            (Npp8u*)gpu_buf_8u_c, out_w*3*sizeof(uchar), dstSize, dstROI, NPPI_INTER_LINEAR);
        nppiConvert_8u32f_C3R((Npp8u*)gpu_buf_8u_c, out_w*3*sizeof(uchar), (Npp32f*)gpu_buf_32f_c, out_w*3*sizeof(float), dstSize);
        nppiMulC_32f_C3IR(m_scale, (Npp32f*)gpu_buf_32f_c, out_w*3*sizeof(float), dstSize);
        nppiAddC_32f_C3IR(a_scale, (Npp32f*)gpu_buf_32f_c, out_w*3*sizeof(float), dstSize);
        nppiCopy_32f_C3P3R((Npp32f*)gpu_buf_32f_c, out_w*3*sizeof(float), rgb_planes, out_w*sizeof(float), dstSize);
    }
private:
    int out_w;
    int out_h;
    int in_w;
    int in_h;
    Npp32f m_scale[3] = {0.0078431, 0.0078431, 0.0078431};
    Npp32f a_scale[3] = {-1.0, -1.0, -1.0};
    int Order[3] = {2,1,0};
    void* gpu_buf_in;
    void* gpu_buf_8u_c;
    void* gpu_buf_32f_c;
};

class TrafficSignClassifier
{
public:
    TrafficSignClassifier(const char* modelname);
    ~TrafficSignClassifier();
    float inference(cv::Mat& im);
    //public output space
    float* pred_cls;
    int width;
    int height;
    void* buffers[2];
    PreProcessor processor;
    int result;
private:
    int batchsize;
    int cls_num;
    Dims3 inputDims;
    Dims3 clsDims;
    size_t inputSize;
    size_t clsSize;
    IRuntime* infer;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
};

#endif