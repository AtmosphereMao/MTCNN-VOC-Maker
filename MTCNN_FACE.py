import tensorflow as tf
import detect_face
import cv2
import numpy as np
import os
import random

tf.app.flags.DEFINE_string(
    'save_path', None, 'data保存路径.')
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'dataset名字.')

FLAGS = tf.app.flags.FLAGS

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# 摄像头
def main(videoNum):
    if not os.path.exists(FLAGS.save_path):
        os.mkdir(FLAGS.save_path)
    try:
        os.mkdir(FLAGS.save_path + "/"+FLAGS.dataset_name)
        os.mkdir(FLAGS.save_path + "/"+FLAGS.dataset_name+"/Annotations")
        os.mkdir(FLAGS.save_path + "/"+FLAGS.dataset_name+"/ImageSets")
        os.mkdir(FLAGS.save_path + "/" + FLAGS.dataset_name + "/JPEGImages")
    except:
        pass
    cap = cv2.VideoCapture(videoNum)
    count = 0
    while cap.isOpened():
        ok, img =cap.read()
        if not ok:
            break
        imgC = np.copy(img) # 未处理的照片（后面用于存储）
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]  # 人脸数目
        # print('找到人脸数目为：{}'.format(nrof_faces))

        # print(bounding_boxes)

        crop_faces = []
        position = []
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)
            # print(face_position[0:4])
            # | 1 当前的目标图片 | 2 左上角的起始点 | 3 右下角的结束点 | 4 颜色(R,G,B) | 5 是否填充 -1 则需要 >0则是线条的宽度
            cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
            crop = img[face_position[1]:face_position[3],
                   face_position[0]:face_position[2], ]
            for i in range(4):
                position.append(face_position[i])


            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
            # print(crop.shape)
            crop_faces.append(crop)


        cv2.imshow("Video", img)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('c'):
            if nrof_faces == 0:
                print("未检测到人脸")
                continue
            cv2.imwrite(FLAGS.save_path + "/" + FLAGS.dataset_name + "/JPEGImages/"+str(count)+".jpg", imgC)
            imgCInfo = list(imgC.shape)
            with open(FLAGS.save_path + "/" + FLAGS.dataset_name + "/Annotations/"+str(count)+".xml", "w") as f:
                f.write('''<annotation>
    <folder>%s</folder>
    <filename>%s</filename>
    <source>
        <database>%s</database>
    </source>
    <size>
        <width>%i</width>
        <height>%i</height>
        <depth>%i</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Me</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%i</xmin>
            <ymin>%i</ymin>
            <xmax>%i</xmax>
            <ymax>%i</ymax>
        </bndbox>
    </object>
</annotation>    
                                ''' % (
                    str(FLAGS.dataset_name),
                    str(count)+".jpg",
                    FLAGS.dataset_name + " Database",
                    imgCInfo[0],
                    imgCInfo[1],
                    imgCInfo[2],
                    position[0],
                    position[1],
                    position[2],
                    position[3]
                ))
            position = []
            count += 1
        if c & 0xFF == ord('q'):
            break
    if count > 2:
        # 数目比例：
        # trainval
        # 约整个数据集的50 %
        # test
        # 约整个数据集的50 %
        # train
        # 约trainval的50 %
        # val
        # 约trainval的50 %
        # num = np.random.randint(1, count+1, count)
        num = random.sample(range(0, count+1), count)
        with open(FLAGS.save_path + "/"+FLAGS.dataset_name+"/ImageSets/test.txt", 'w') as f:
            for i in num[0 : int(count/2)]:
                f.writelines(str(i)+"\n")
        with open(FLAGS.save_path + "/" + FLAGS.dataset_name + "/ImageSets/trainval.txt", 'w') as f:
            for i in num[int(count/2): count]:
                f.writelines(str(i)+"\n")
        with open(FLAGS.save_path + "/" + FLAGS.dataset_name + "/ImageSets/val.txt", 'w') as f:
            for i in num[int(count/2): int(count/2) + int(count/4)]:
                f.writelines(str(i)+"\n")
        with open(FLAGS.save_path + "/" + FLAGS.dataset_name + "/ImageSets/train.txt", 'w') as f:
            for i in num[int(count/2) + int(count/4) : count]:
                f.writelines(str(i)+"\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if not FLAGS.save_path:
        raise ValueError('Set up --save_path')
    if not FLAGS.dataset_name:
        raise ValueError('Set up --dataset_name')
    main(0)

