import tensorflow as tf
import numpy as np
import cv2 as cv
from task2 import detection


'''
调用模型对实时抓拍的人脸图片进行识别
'''


# 获取人脸数据的标签，根据字母排序
labels = np.load('../result/labels.npy')
labels = labels.item()
labels = list(set(labels["te_labels"]))
labels.sort()

while(True):
    capture = cv.VideoCapture(0)
    ref, frame = capture.read()
    bgr, bboxs = detection(frame)  # 检测人脸
    if bboxs is None:
        print("未检测到人脸")
    else:
        bbox = bboxs[0]
        crop = frame[bbox[1]: bbox[3], bbox[0]: bbox[2]]  # 提取人脸
        Gray_crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)  # 灰度化

        # 调整图片形状、大小为：[1, 64, 64, 3]
        img = cv.resize(Gray_crop, (64, 64))
        img = img[np.newaxis, :, :, np.newaxis]
        img2 = img.repeat([3], axis=-1)

        with tf.Session() as sess:
            # 加载模型
            saver = tf.train.import_meta_graph('../result/model/train_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('../result/model/'))
            graph = tf.get_default_graph()
            img_new = graph.get_tensor_by_name('img:0')
            y_new = graph.get_tensor_by_name('y:0')
            # 预测
            y_pre = sess.run(y_new, feed_dict={img_new: img2})

            if np.max(y_pre) > 0.5:
                index = np.argmax(y_pre)
                label_pre = labels[index]
                print(label_pre)
                text = label_pre + '  ' + str(np.max(y_pre))[:4]
                bgr = cv.putText(bgr, text, (bbox[0]-10, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv.imshow('know', bgr)
            else:
                print("unknow")
                bgr = cv.putText(bgr, "unknow", (bbox[0]-10, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv.imshow('unknow', bgr)
                cv.waitKey(10)
    c = cv.waitKey(10)
    if c == 27:  # 按esc键退出
        break
capture.release()
cv.destroyAllWindows()
