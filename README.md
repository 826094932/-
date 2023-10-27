## 实践流程
### 1.配置环境
&emsp;本次实践所使用的编程环境：
Python 3.8(ubuntu20.04) 
PyTorch 1.110
Cuda 11.3

  &emsp;克隆YOLO5的仓库，并下载其中的包，配置其中的程序运行环境。

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt 

&emsp;本次实践所需要的Python库具体如下：
#YOLOv5 requirements
#Usage: pip install -r requirements.txt

#Base ------------------------------------------------------------------------
gitpython>=3.1.30
matplotlib>=3.3
numpy>=1.22.2
opencv-python>=4.1.1
Pillow>=7.1.2
psutil  # system resources
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
thop>=0.1.1  # FLOPs computation
torch>=1.8.0  # see https://pytorch.org/get-started/locally (recommended)
torchvision>=0.9.0
tqdm>=4.64.0
ultralytics>=8.0.147
#protobuf<=3.20.1  # https://github.com/ultralytics/yolov5/issues/8012

#Logging ---------------------------------------------------------------------
#tensorboard>=2.4.1
#clearml>=1.2.0
#comet

#Plotting --------------------------------------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

#Export ----------------------------------------------------------------------
#coremltools>=6.0  # CoreML export
#onnx>=1.10.0  # ONNX export
#onnx-simplifier>=0.4.1  # ONNX simplifier
#nvidia-pyindex  # TensorRT export
#nvidia-tensorrt  # TensorRT export
#scikit-learn<=1.1.2  # CoreML quantization
#tensorflow>=2.4.0  # TF exports (-cpu, -aarch64, -macos)
#tensorflowjs>=3.9.0  # TF.js export
#openvino-dev>=2023.0  # OpenVINO export

#Deploy ----------------------------------------------------------------------
setuptools>=65.5.1 # Snyk vulnerability fix
#tritonclient[all]~=2.24.0

#Extras ----------------------------------------------------------------------
#ipython  # interactive notebook
#mss  # screenshots
#albumentations>=1.0.3
#pycocotools>=2.0.6  # COCO mAP

### 2、数据处理和划分
&emsp;  本次实践的数据集，主要来自于学生自己拍的关于论文、书籍、报纸和杂志的图片，并使用Labellmg对图片的文本版面内容进行标注，分为：Text、Figure、Table、Equation四个部分，用VOC格式进行标注。（YOLO5训练需要将VOC标注得到的.xml文件转换为.txt文本格式）
对数据集进行划分，分别为训练集（train）和验证集（val），其中训练集的图片数量为6352张，验证集的图片数量为500张，保存在image文件夹中，每张图片对应的标注文件txt保存在labels中，且每张图片对应的标注文件与图片名称相同。
数据集层级如下：
![img_1.png](img_1.png)

### 3、配置模型训练所需要的参数
&emsp;创建yaml文件，yaml文件中需要包括：整个数据的路径（PATH）、训练集的路径（TRAIN）、验证集的路径（VAL）和模型训练所需要划分的类别NAME，NAME包括Text、Figure、Table、Equation四个类别。
![img_2.png](img_2.png)
### 注意事项：
&emsp;Name中的0，1，2，3需要按照txt文件中标注对应的具体序号进行编写，不然会出现文本版面识别相反，在一开始的训练中，我将Table和Equation的标签号写反，导致训练出来的模型识别的Table和Equation相反。

### 4、设置需要的模型训练参数，进行模型训练。
&emsp;在终端中调用以下命令行，进行模型训练

python train.py --data data/data.yaml --cfg models/yolov5m.yaml --weights ./weights/yolov5m.pt --epochs 30
–data 创建 yaml文件

--weights下载预训练模型，本次实践选择使用的是yolov5m.pt的预训练模型。

### 注意事项：
&emsp;  迭代次数需要根据服务器的具体情况改变，epochs的数量越大，服务器所需要的GPU越大，如果设置参数过大，会导致超过最大GPU导致报错，且迭代次数影响这模型最后的精度，要根据自己的数据集和服务器选择合适的迭代次数。


### 5、对模型进行简单测试
&emsp;在终端中调用以下命令行，进行模型预测

&emsp;python detect.py --weights 模型路径 --source 检测图片路径

&emsp;之后，会在runs\detect\exp文件夹中生成和测试图片名字名相同的检测后图片。


### 6、将Pt格式的模型转变为onnx格式，便于后续进行部署
&emsp;Yolo5自带转变模型格式的文件export.py,在终端中调用

&emsp;python export.py --weights xxx.pt --include onnx engine

&emsp;系统会将路径中的Pt格式的模型转变为onnx格式。# -
文本版面检测实践
