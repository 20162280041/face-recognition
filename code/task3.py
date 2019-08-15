import numpy as np
import cv2 as cv
import os


'''
整理图片数据和标签
将图片大小形状统一为:[1, 64, 64, 3]，并以8:2的比例将图片分为训练集和测试集
'''

if __name__ == "__main__":

    path = '../result/faceImagesGray'  # 存放人脸灰度图的路径
    tr_data = []
    te_data = []
    tr_labels = []
    te_labels = []

    folders = os.listdir(path)
    for folder in folders:
        path_1 = path + '/' + folder
        pic_names = os.listdir(path_1)
        data = []
        labels = []
        for pic_name in pic_names:
            arr_img = cv.imread(path_1 + '/' + pic_name)
            resize_img = cv.resize(arr_img, (64, 64))
            img_new = np.float32(np.reshape(resize_img, [1, 64, 64, 3]))
            data.append(img_new)
            labels.append(folder)
        Sum = len(data)
        part = int(Sum * 0.8) + 1
        part = int(part / 100) * 100 + 70  # 为了保证训练集整百个， 且训练：测试约为 8:2
        tr_data.extend(data[0: part])  # 训练图片数据存放在列表tr_data
        te_data.extend(data[part:])
        tr_labels.extend(labels[0: part])  # 训练图片标签存放在列表tr_labels, 与训练图片数据一一对应
        te_labels.extend(labels[part:])

    data = {"tr_data": tr_data, "te_data": te_data}
    labels = {"tr_labels": tr_labels, "te_labels": te_labels}
    np.save('../result/data.npy', data)
    np.save('../result/labels.npy', labels)
