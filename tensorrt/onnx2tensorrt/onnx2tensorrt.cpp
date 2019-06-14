#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"

#include "common.h"

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;
static int gUseDLACore = 2;

//static int CLS_NUM = 7;

static const int CAL_BATCH_SIZE = 16;
static const int FIRST_CAL_BATCH = 500, NB_CAL_BATCHES = 2000;                // calibrate over images 0-600
static const int FIRST_CAL_SCORE_BATCH = 500, NB_CAL_SCORE_BATCHES = 200; // score over images 500-5000
static DataType DATA_TYPE = DataType::kFLOAT;

//const char* gNetworkName{nullptr};


bool onnxToTRTModel(const std::string& modelFile,
                    const char* trtPath,
                    int& maxBatchSize, 
                    DataType dataType,
                    IInt8Calibrator* calibrator,
                    nvinfer1::IHostMemory*& trtModelStream)
{
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    // parse the onnx model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(modelFile.c_str(), verbosity))
    {
        std::string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    //check platform and datatype
    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8()))
    {
        std::cout<<"Current Device does not support INT8 inference!"<<std::endl;
        return false;
    }
    if(dataType == DataType::kHALF && !builder->platformHasFastFp16())
    {
        std::cout<<"Current Device does not support FP16 inference!"<<std::endl;
        return false;
    }

    // Build the engine
    builder->setMaxWorkspaceSize(1l << 30);
    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);
    builder->setDebugSync(true);
    builder->setInt8Mode(dataType == DataType::kINT8);
    builder->setFp16Mode(dataType == DataType::kHALF);
    builder->setInt8Calibrator(calibrator);
    if (gUseDLACore >= 0)
    {
        samplesCommon::enableDLA(builder, gUseDLACore);
        if (maxBatchSize > builder->getMaxDLABatchSize())
        {
            std::cerr << "Requested batch size " << maxBatchSize << " is greater than the max DLA batch size of "
                      << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
            maxBatchSize = builder->getMaxDLABatchSize();
        }
    }

    builder->setMaxBatchSize(maxBatchSize);
    std::cout<<"000"<<std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // serialize the engine, then close everything down
    
    std::cout<<"111"<<std::endl;
    trtModelStream = engine->serialize();
    std::cout<<"222"<<std::endl;
    FILE* fp = fopen(trtPath, "wb");
    fwrite(trtModelStream->data(), 1, trtModelStream->size(), fp);
    fclose(fp);

 // we don't need the network any more, and we can destroy the parser
    parser->destroy();
    engine->destroy();
    network->destroy();
    builder->destroy();
    std::cout<<"Create TensorRT model finished!"<<std::endl;
    
    return true;
}

float doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[3];
    float ms{0.0f};

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = 0;
    int outputIndex_cls = 1;
    // create GPU buffers and a stream
    Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(inputIndex));
    Dims3 outputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(outputIndex_cls));

    size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float);
    size_t outputSize = batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex_cls], outputSize));

    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    cudaEventRecord(start, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaMemcpy(output, buffers[outputIndex_cls], outputSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex_cls]));
    CHECK(cudaStreamDestroy(stream));
    return ms;
}


float scoreModel(std::string modelname, const char* trtPath, int batchSize, int firstBatch, int nbScoreBatches, DataType datatype, IInt8Calibrator* calibrator, bool quiet = false)
{
    IHostMemory* trtModelStream{nullptr};
    bool valid = onnxToTRTModel(modelname, trtPath, batchSize, datatype, calibrator, trtModelStream);

    if (!valid)
    {
        std::cout << "Engine could not be created at this precision" << std::endl;
        return 0;
    }

    assert(trtModelStream != nullptr);

    // Create engine and deserialize model.
    std::cout<<"000"<<std::endl;
    IRuntime* infer = createInferRuntime(gLogger);
    assert(infer != nullptr);
    if (gUseDLACore >= 0)
    {
        infer->setDLACore(gUseDLACore);
    }
    ICudaEngine* engine = infer->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    std::cout<<"111"<<std::endl;
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    context->destroy();
    engine->destroy();
    infer->destroy();
    return 0;
}

static void printUsage()
{
    std::cout << std::endl;
    std::cout << "Usage: ./sample_int8 <network name> <optional params>" << std::endl;
    std::cout << std::endl;
    std::cout << "Optional params" << std::endl;
    std::cout << "  batch=N            Set batch size (default = 100)" << std::endl;
    std::cout << "  start=N            Set the first batch to be scored (default = 100). All batches before this batch will be used for calibration." << std::endl;
    std::cout << "  score=N            Set the number of batches to be scored (default = 400)" << std::endl;
    std::cout << "  search             Search for best calibration. Can only be used with legacy calibration algorithm" << std::endl;
    std::cout << "  legacy             Use legacy calibration algorithm" << std::endl;
    std::cout << "  useDLACore=N       Enable execution on DLA for all layers that support dla. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 2 || !strncmp(argv[1], "help", 4) || !strncmp(argv[1], "--help", 6) || !strncmp(argv[1], "--h", 3))
    {
        printUsage();
        exit(0);
    }
    fstream file;
    file.open(argv[1], ios::in);
    if(!file){
        std::cout << "no onnx file" << std::endl;
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
    
    std::string modelName(argv[1]);

    char trtPath[100];
    strcpy(trtPath, argv[2]);
    strcat(trtPath, "/model_best.trt");

    int batchSize = CAL_BATCH_SIZE;
    int firstScoreBatch = FIRST_CAL_SCORE_BATCH;
    int nbScoreBatches = NB_CAL_SCORE_BATCHES;
    bool search = false;
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION;

    for (int i = 3; i < argc; i++)
    {
        if (!strncmp(argv[i], "batch=", 6))
            batchSize = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "start=", 6))
            firstScoreBatch = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "score=", 6))
            nbScoreBatches = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "search", 6))
            search = true;
        else if (!strncmp(argv[i], "legacy", 6))
            calibrationAlgo = CalibrationAlgoType::kLEGACY_CALIBRATION;
        else if (!strncmp(argv[i], "useDLACore=", 11))
            gUseDLACore = stoi(argv[i] + 11);
        else
        {
            std::cout << "Unrecognized argument " << argv[i] << std::endl;
            exit(0);
        }
    }

    if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
    {
        search = false;
    }

    if (batchSize > 128)
    {
        std::cout << "Please provide batch size <= 128" << std::endl;
        exit(0);
    }

    std::cout.precision(6);
    int dla{gUseDLACore};

    // Set gUseDLACore to -1 here since FP16 mode is not enabled.
    
    if (gUseDLACore >= 0)
    {
        std::cout << "\nDLA requested. Disabling for FP32 run since its not supported." << std::endl;
        gUseDLACore = -1;
    }
    std::cout << "\nFP32 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    scoreModel(modelName, trtPath, batchSize, firstScoreBatch, nbScoreBatches, DATA_TYPE, nullptr);
    
    // Set gUseDLACore correctly to enable DLA if requested.
    //gUseDLACore = dla;
    //std::cout << "\nFP16 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    //scoreModel(modelName, trtPath, batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, nullptr);


    return 0;
}
