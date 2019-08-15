import tensorflow as tf
import numpy as np
import random


'''
训练基于卷积神经网络的人脸识别模型，测试模型精度，保存模型
'''


def load_data():
    labels = np.load('../result/labels.npy')
    data = np.load('../result/data.npy')
    labels = labels.item()
    data = data.item()
    tr_data = data["tr_data"]
    te_data = data["te_data"]
    tr_labels = labels["tr_labels"]
    te_labels = labels["te_labels"]

    seed = 888
    random.seed(seed)
    tr = list(zip(tr_data, tr_labels))
    random.shuffle(tr)
    tr_data[:], tr_labels[:] = zip(*tr)

    set_labels = list(set(te_labels))
    set_labels.sort()
    labels = {set_labels[0]: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
              set_labels[1]: np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
              set_labels[2]: np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
              set_labels[3]: np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
              set_labels[4]: np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.float32),
              set_labels[5]: np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32),
              set_labels[6]: np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32),
              set_labels[7]: np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=np.float32),
              set_labels[8]: np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float32),
              set_labels[9]: np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)}
    return tr_data, te_data, tr_labels, te_labels, labels


if __name__ == "__main__":
    tf.reset_default_graph()
    tr_data, te_data, tr_labels, te_labels, labels = load_data()

    img = tf.placeholder(tf.float32, [1, 64, 64, 3], name='img')
    label = tf.placeholder(tf.float32, [1, 10])

    seed = 890
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96)

    w1 = tf.Variable(tf.random_normal([5, 5, 3, 6], stddev=0.01, seed=seed), name='w1')
    b1 = tf.Variable(tf.zeros([6]), name='b1')
    conv1 = tf.nn.conv2d(img, w1, strides=[1, 1, 1, 1], padding='VALID') + b1   # VALID/SAME
    R_conv1= tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    w2 = tf.Variable(tf.random_normal([5, 5, 6, 10], stddev=0.01, seed=seed), name='w2')
    b2 = tf.Variable(tf.zeros([10]), name='b2')
    conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='VALID') + b2
    R_conv2= tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    x = tf.reshape(pool2, (1, -1))
    n = int(x.shape[1])
    m = tf.constant(200)
    w3 = tf.Variable(tf.random_normal([n, m], stddev=0.01, seed=seed), name='w3')
    b3 = tf.Variable(tf.zeros([1, m]), name='b3')
    full1 = tf.nn.sigmoid(tf.matmul(x, w3) + b3)

    w4 = tf.Variable(tf.random_normal([m, 10], stddev=0.01, seed=seed), name='w4')
    b4 = tf.Variable(tf.zeros([1, 10]), name='b4')
    y = tf.nn.softmax(tf.matmul(full1, w4) + b4, name='y')

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cross_entropy, global_step=global_step)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print("In the training...")
        sess.run(init)
        for k in range(10):
            print("epoch %d" % k)
            for i in range(len(tr_data)):
                tr_img, tr_label = tr_data[i], labels[tr_labels[i]]
                sess.run(train, feed_dict={img: tr_img, label: tr_label})
        saver.save(sess, '../result/model/train_model')

        Sum = 0
        print("In the test...")
        for i in range(len(te_data)):
            te_img, te_label = te_data[i], labels[te_labels[i]]
            label_argmax = int(sess.run(tf.argmax(te_label, axis=1)))
            y_pre_argmax = int(sess.run(tf.argmax(y, axis=1), feed_dict={img: te_img}))
            Sum += (label_argmax == y_pre_argmax)
        print("模型测试集精度：", Sum / len(te_data))


