#include "traffic_sign_classifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

static Logger gLogger;

TrafficSignClassifier::TrafficSignClassifier(const char* modelname)
{
    batchsize = 1;
    //read the tensorrt engine file
    unsigned char* buf;
    FILE* fp = fopen(modelname, "rb");
    fseek(fp, 0L, SEEK_END); 
    int size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    buf = (unsigned char*)malloc(size);
    fread((void*)buf, 1, size, fp);
    fclose(fp);
    // create tensorrt engine 
    infer = createInferRuntime(gLogger);
    assert(infer != nullptr);
    engine = infer->deserializeCudaEngine(buf, size, nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    std::cout<<"Create TensorRT Engine Done!"<<std::endl;
    //init the gpu memory
    assert(engine->getNbBindings() == 2);
    inputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(0));
    clsDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(1));
    inputSize = batchsize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float);
    clsSize = batchsize * clsDims.d[0] * sizeof(float);
    std::cout<<"Input Dim: "<<batchsize<<"x"<<inputDims.d[0]<<"x"<<inputDims.d[1]<<"x"<<inputDims.d[2]<<std::endl;
    std::cout<<"Out cls Dim: "<<batchsize<<"x"<<clsDims.d[0]<<std::endl;
    this->width = inputDims.d[2];
    this->height = inputDims.d[1];
    cls_num = clsDims.d[0];
    processor.init(this->width, this->height);
    CHECK(cudaMalloc(&buffers[0], inputSize));
    CHECK(cudaMalloc(&buffers[1], clsSize));
    pred_cls = (float*)malloc(clsSize);
    free(buf);
    CHECK(cudaStreamCreate(&stream));
}

TrafficSignClassifier::~TrafficSignClassifier()
{
    context->destroy();
    engine->destroy();
    infer->destroy();
    //free gpu memory
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    free(pred_cls);
    cudaStreamDestroy(stream);
}

float TrafficSignClassifier::inference(cv::Mat& im)
{
    float ms{0.0f};
    //CHECK(cudaMemcpy(buffers[0], data, inputSize, cudaMemcpyHostToDevice));
    processor.process(im, buffers[0]);

    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    cudaEventRecord(start, stream);
    context->enqueue(batchsize, buffers, stream, nullptr);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaMemcpy(pred_cls, buffers[1], clsSize, cudaMemcpyDeviceToHost));

    float max = -1e6;
    int idx = 0;
    for(int i=0; i<cls_num; i++)
    {
        if(pred_cls[i] > max){
            max = pred_cls[i];
            idx = i;
        }
    }
    result = idx;
    return ms;
}