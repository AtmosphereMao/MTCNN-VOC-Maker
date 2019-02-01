# MTCNN-VOC-Maker
Using MTCNN to create VOC format dataset of your face

**主要用途**
  该项目的主要用途是制作自己脸部的VOC格式的数据集，并应用于人脸识别当中。
  
**环境**
tensorflow,opencv,numpy

**使用教程** 
  打开文件
  ```
    $ python MTCNN_FACE.py --save_path="你要保存的目录" --dataset_name="你的数据名字"
  ```
  加载完后，会出现在Video的窗口
  按 C 截取图片，按 Q 退出程序
  （注意每次截取4张以上，才有完整的标记数据文件）
 
