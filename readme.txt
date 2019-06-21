

训练+测试：
python3 main.py
测试：
注释main.py中的main()函数，运行python3 main.py
生成onnx文件：
python3 generate_onnx.py
生成trt文件：
cd TensorRT/onnx2tensorrt
make
./onnx2tensorrt.bin onnx路径 trt文件存储路径
使用trt文件预测：
cd TensorRT/runtensorrt
make
./traffic_sign_classifier.bin 图片路径 trt文件路径


准确率速度对比
Pytorch:
resnet50 99.1% 11.4ms
resnet18 98.7% 4.84ms
alexnet 83.8% 3.17ms
vgg11 fold1 98.90% 2.14ms fold2 97.745% 2.368ms
squeezenet1.1 fold0 71.6% 4.33ms
mobilenetv2 95.54% 10.27ms

Tensorrt:
vgg11 0.91456ms


