# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:06:45 2019

@author: windows 10
"""

import cv2 as cv
import os

'''
拍摄10个人的人脸图片，每人各600张。
各人图片存放在以其姓名拼音命名的文件夹里。
在本文件所在路径下创建一个命名为faceImages的文件夹，存放以上10个文件夹。
'''


def video_demo(path):
    # 0是代表摄像头编号，只有一个的话默认为0
    capture = cv.VideoCapture(0)
    i = 0
    while (True):
        # 调用摄像机
        ref, frame = capture.read()
        # 输出图像,第一个为窗口名字
        cv.imshow('frame', frame)
        # 10ms显示图像
        cv.waitKey(10)
        # 储存照片
        cv.imwrite(path + str(i) + '.jpg', frame)
        i = i + 1
        if i == 600:
            capture.release()
            cv.destroyAllWindows()
            break


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
    

if __name__ == '__main__':
    path = '..\\result\\faceImages\\'  # 存放即将收集的图片的路径
    name = input("请输入照片被采集者的名字拼音全拼：")
    path = path + name + '\\'
    mkdir(path)
    video_demo(path)

