# Traffic Sign Classification
本项目用于交通标志分类，并使用TensorRT推理引擎进行部署

## 运行环境
OS:Ubuntu16.04 
GPU:1080Ti 
CUDA:9.0 
cudnn:7.0 
PyTorch:1.0.1
TensorRT:5.0.2.6
## 数据集
CCTSDB数据集、GTSRB数据集  
## 网络结构选择
| 网络结构 | resnet50 | resnet18 | alexnet | vgg11 | squeezenet1.1 |  mobilenetv2 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 单次预测时间 | 11.4ms | 4.84ms | 3.17ms | 2.34ms | 4.33ms | 10.27ms | 
| 测试准确率 | 99.1% | 98.7% | 83.8% | 97.8% | 71.6%  | 95.5% | 

所以最终选择vgg11作为分类网络，在TensorRT推理引擎上单次预测时间为0.91456ms  
## 运行步骤
### 训练+测试
$ python3 main.py
### 生成onnx文件：
$ python3 generate_onnx.py
### 生成trt文件：
$ cd TensorRT/onnx2tensorrt  
$ make  
$ ./onnx2tensorrt.bin onnx路径 trt文件存储路径  
### 使用trt文件预测：
$ cd TensorRT/runtensorrt  
$ make  
$ ./traffic_sign_classifier.bin 图片路径 trt文件路径  
