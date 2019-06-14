

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


