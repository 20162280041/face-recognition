import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import align.detect_face


'''
利用align文件里的detect_face模块里的预训练好的mtcnn模型进行人脸检测
faceImagesGray文件夹保存灰剪裁的人脸灰度图片，保存方式同task1

'''


# print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def detection(image):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # detect with RGB image
    h, w = image.shape[:2]
    bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        # print("can't detect face in the frame")
        return None, None
    # print("num %d faces detected" % len(bounding_boxes))
    bgr = image
    bbox = []
    for i in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[i, 0:4])   # float
        margin = np.abs(np.minimum(det[0] - det[2], det[1] - det[3])) / 10
        bb = np.zeros(4, dtype=np.int32)
        # x1, y1, x2, y2
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, w)
        bb[3] = np.minimum(det[3] + margin / 2, h)
        cv.rectangle(bgr, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2, 8, 0)
        bbox.append(bb)
    return bgr, bbox


def mkdir(path):
    path = path.strip()
    path = path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        print(path + ' 创建成功')
        os.makedirs(path)
        return True
    else:
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":
    path = '../result/faceImages'  # 存放原始图片的10个文件夹的路径

    folders = os.listdir(path)
    for folder in folders:
        path_1 = path + '/' + folder
        filenames = os.listdir(path_1)
        final_path = '../result/faceImagesGray/' + folder
        mkdir(final_path)
        for filename in filenames:
            img = cv.imread(path_1 + '/' + filename)
            bgr, bbox = detection(img)
            if not (bbox is None):
                for i, bb in enumerate(bbox):
                    crop = img[bb[1]: bb[3], bb[0]: bb[2]]
                    Gray_crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                    cv.imwrite(final_path + '/' + filename[:-4] + '_' + str(i) + '.jpg', Gray_crop)  # 保存人脸灰度图


