#coding:utf-8
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
 
#数据集地址
path='E:/f/flower_photos/'
#模型保存地址
model_path='E:/f/flowers/model.ckpt'
 
#将所有的图片resize成100*100
w=100
h=100
c=3
def main():
 
#读取图片
    def read_img(path):
        cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] #如果x在path+x目录中则将其赋值给cate
        imgs=[]
        labels=[]
        for idx,folder in enumerate(cate):      #列出cate列表的数据下标和数据
            for im in glob.glob(folder+'/*.jpg'):   #匹配所有文件路径列表，即获得该目录下所有jpg文件并赋值给im
                print('reading the images:%s'%(im))
                img=io.imread(im) #io.imread 读出的图片格式是uint8，value是numpy array 类型
                img=transform.resize(img,(w,h)) #mode='constant' 改变图片大小为100*100
                imgs.append(img)
                labels.append(idx)
        print(labels)
        return np.asarray(imgs,np.float32),np.asarray(labels,np.int32) #返回图片的矩阵值以及下标值
    data,label=read_img(path)
 
 
#打乱顺序
    num_example=data.shape[0]  #读取一维矩阵的长度
    arr=np.arange(num_example)#一个参数 默认起点0，步长为1 输出：[0,1,2...,num_example]
    np.random.shuffle(arr) #现场修改序列，改变自身内容。（类似洗牌，打乱顺序）
    data=data[arr]
    label=label[arr]

 
#将所有数据分为训练集和验证集
    ratio=0.8
    s=np.int(num_example*ratio)
    x_train=data[:s] #前80%作为训练图片
    y_train=label[:s] #前80%作为训练标签
    x_val=data[s:] #后20%作为测试图片
    y_val=label[s:] #前20%作为测试标签
#-----------------构建网络----------------------
#占位符
    x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
    y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

    def inference(input_tensor, train, regularizer):
        #第一个参数input_tensor：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]
        #这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
        #注意这是一个4维的Tensor，要求类型为float32和float64其中之一
        with tf.variable_scope('layer1-conv1'): 
        #采用tf.variable_scope()进行变量管理，因为神经网络变量太多了，这样就不用担心命名很容易重复了
            conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
            #卷积核的尺寸
            #使用尺寸为5，步长为3，深度为32卷积核，tf.get_variable(变量名，形状，initializer：创建变量的初始化器)
            #stddev是标准差，tf.truncated_normal_initializer(stddev=0.1)
            #这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
            conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
            #将偏置项初始为0 initializer=tf.constant_initializer(0.0)
            conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') 
            #padding='SAME'使用全0填充，strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
            #计算激活函数，将大于0的数保持不变，小于0的数置为0
            #一般情况下，使用ReLU会比较好
            #1、使用 ReLU，就要注意设置 learning rate，不要让网络训练过程中出现很多 “dead” 神经元；
            #2、如果“dead”无法解决，可以尝试 Leaky ReLU、PReLU 、RReLU等Relu变体来替代ReLU；
            #3、不建议使用 sigmoid，如果一定要使用，也可以用 tanh来替代。

            #卷积层的计算过程为卷积层內积加上偏置项conv1_biases的值
            #relu激励函数它的特点是收敛快，求梯度简单，但较脆弱
			
        with tf.name_scope("layer2-pool1"):
            pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
            #第一个参数value：需要池化的输入，一般池化层接在卷积层后面，
            #所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
            #第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]
            #因为我们不想在batch和channels上做池化，所以这两个维度设为了1
            #第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
            #第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'

        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
 
        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
        with tf.variable_scope("layer5-conv3"):
            conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
 
        with tf.name_scope("layer6-pool3"):
            pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
        with tf.variable_scope("layer7-conv4"):
            conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
 
        with tf.name_scope("layer8-pool4"):
            pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            nodes = 6*6*128
            reshaped = tf.reshape(pool4,[-1,nodes]) #-1代表自动计算此维度
 
        with tf.variable_scope('layer9-fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
            fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
 
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            #tf.matmul为两个矩阵相乘
            if train: fc1 = tf.nn.dropout(fc1, 0.5)
            #防止过拟合，第一个参数为输入，第二个参数为神经元被选中的概率
 
        with tf.variable_scope('layer10-fc2'):
            fc2_weights = tf.get_variable("weight", [1024, 512],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
 
            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
            if train: fc2 = tf.nn.dropout(fc2, 0.5)
 
        with tf.variable_scope('layer11-fc3'):
            fc3_weights = tf.get_variable("weight", [512, 7],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
            fc3_biases = tf.get_variable("bias", [7], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc2, fc3_weights) + fc3_biases
 
        return logit
 
#---------------------------网络结束---------------------------
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    logits = inference(x,False,regularizer)
 
#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
    b = tf.constant(value=1,dtype=tf.float32)
    logits_eval = tf.multiply(logits,b,name='logits_eval') 

    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
    print(correct_prediction)
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#定义一个函数，按批次取数据
    def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size+1):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]
 

#训练和测试数据，可将n_epoch设置更大一些
    n_epoch=10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    batch_size=64
    saver=tf.train.Saver()
    sess=tf.Session()  
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        start_time = time.time()
 
    #training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
        print("   train acc: %f" % (np.sum(train_acc)/ n_batch))

    #validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
            print(ac)
            val_loss += err; val_acc += ac; n_batch += 1
        print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
        print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
    saver.save(sess,model_path)
    sess.close()

if __name__ == '__main__':
    main()
