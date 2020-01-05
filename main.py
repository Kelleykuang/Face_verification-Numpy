import numpy as np
import pandas as pd
from skimage import io, transform, measure
import scipy.stats
import glob
import re
import pickle
import time
import logging
import os

image_width = 128
image_height = 128

def traindata_loader(path1, path2, path3):
    imlist1 = glob.glob(path1)
    imlist2 = glob.glob(path2)
    imlist3= glob.glob(path3)
    print(len(imlist1), len(imlist2), len(imlist3))
    rawImageArray = np.zeros((len(imlist1)+len(imlist2)+len(imlist3), image_height, image_width), dtype=np.double)
    people_names = []
    for i in range(len(imlist1)):
        image = imlist1[i]
        #print(image)
        raw_image = io.imread(image, as_gray=True)
        rawImageArray[i] = transform.resize(raw_image, (image_width, image_height))
        people_name = re.search(r'(?<=\\)[a-zA-Z_-]+', image[20:]) 
        people_names.append(people_name.group(0)[:-1])  
    for i in range(len(imlist1), len(imlist1)+len(imlist2)):
        image = imlist2[i-len(imlist1)]
        #print(image)
        raw_image = io.imread(image, as_gray=True)
        rawImageArray[i] = transform.resize(raw_image, (image_width, image_height))
        people_name = re.search(r'(?<=\\)[a-zA-Z_-]+', image[20:]) 
        people_names.append(people_name.group(0)[:-1])  
    for i in range(len(imlist1)+len(imlist2), len(imlist1)+len(imlist2)+len(imlist3)):
        image = imlist3[i-len(imlist1)-len(imlist2)]
        #print(image)
        raw_image = io.imread(image, as_gray=True)
        rawImageArray[i] = transform.resize(raw_image, (image_width, image_height))
        people_name = re.search(r'(?<=\\)[a-zA-Z_-]+', image[20:]) 
        people_names.append(people_name.group(0)[:-1])  
    data_label = pd.get_dummies(people_names).to_numpy()
    return rawImageArray[len(imlist1):], data_label[len(imlist1):]

def conv(data_in, filter, filter_bias):
    '''
      ## convolve with stride=1 and padding=0
      (l, w, h), (fl, fw, h, n), (n) -> (l-fl+1, w-fw+1, n)
    '''
    assert len(data_in.shape) == 3 and len(filter.shape) == 4 and len(filter_bias.shape) == 1
    assert data_in.shape[2] == filter.shape[2]    # must share the same height
    assert filter.shape[3] == filter_bias.shape[0]  # each conv kernel has a bias
    input_len, input_width, input_height = data_in.shape
    filter_len, filter_width, filter_height, filter_num = filter.shape
    feature_len = input_len - filter_len + 1
    feature_width = input_width - filter_width + 1
    feature_height = filter_num
    img2col = np.lib.stride_tricks.as_strided(data_in, shape=(feature_len, feature_width, filter_len, filter_width, filter_height), strides=(8*input_height*input_width, 8*input_height, 8*input_height*input_width, 8*input_height, 8))
    img2col = np.reshape(img2col, (feature_len, feature_width, filter_len*filter_width*filter_height))
    filter = np.reshape(filter, (filter_len*filter_width*filter_height, filter_num))
    output = np.add(np.matmul(img2col, filter), filter_bias)
    assert output.shape == (feature_len, feature_width, feature_height)
    return output

def mfm(data_in):
    '''
    ## max-feature-map 2/1
    (l, w, h) -> (l, w, h/2)  
    '''
    assert len(data_in.shape) == 3
    assert data_in.shape[2]%2 == 0
    input_len, input_width, input_height = data_in.shape
    split = np.zeros((2,input_len, input_width, input_height//2), dtype=np.double)
    split[0] = data_in[:,:,:input_height//2]
    split[1] = data_in[:,:,input_height//2:]
    output = np.amax(split, axis=0)
    repmax = np.zeros(data_in.shape, dtype=np.double)
    repmax[:,:,:input_height//2] = repmax[:,:,input_height//2:] = output
    location = (repmax == data_in)
    assert output.shape == (input_len, input_width, input_height//2)
    assert location.shape == data_in.shape
    return output, location

def pool(data_in):
    '''
    ## max-pooling with 2*2 filter size and stride=2
    (l, w, h) -> (l/2, w/2, h)
    '''

    assert len(data_in.shape) == 3
    assert data_in.shape[0]%2 == 0 and data_in.shape[1]%2 == 0 
    output = measure.block_reduce(data_in, (2,2,1), func=np.max)
    repmax = np.repeat(np.repeat(output, 2, axis=0), 2, axis=1)
    location = (repmax == data_in)
    assert output.shape == (data_in.shape[0]//2, data_in.shape[1]//2, data_in.shape[2])
    assert location.shape == data_in.shape
    return output, location

def padding(data_in, pad_size):
    '''
    ## pad the first 2 dimension of data_in
     (l, w, h), n -> (n+l+n, n+w+n, h)
    '''
    assert len(data_in.shape) == 2 or len(data_in.shape) == 3
    assert pad_size > 0
    if len(data_in.shape) == 2:
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size] = data_in
    else:
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2, data_in.shape[2]), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size,:] = data_in
    return output

def fc(data_in, weights, bias):
    '''
    ## fully-connected layer
    ndarray, (w, node_num), (node_num) -> (node_num)
    '''
    data = data_in.flatten()
    assert data.shape[0] == weights.shape[0]
    assert weights.shape[1] == bias.shape[0]
    weight_num, node_num = weights.shape
    output = np.matmul(data, weights) + bias
    assert output.shape == (node_num,)
    return output

def mfm_fc(data_in):
    '''
    ## max-feature-map for fully-connected layer, 2/1
    (node_num) -> (node_num/2)
    '''
    assert len(data_in.shape) == 1
    assert data_in.shape[0]%2 == 0
    node_num = data_in.shape[0]
    split = np.zeros((2, node_num//2), dtype=np.double)
    split[0] = data_in[:node_num//2]
    split[1] = data_in[node_num//2:]
    output = np.amax(split, axis=0)
    repmax = np.zeros(data_in.shape, dtype=np.double)
    repmax[:node_num//2] = repmax[node_num//2:] = output
    location = (repmax == data_in)
    assert output.shape == (node_num//2,)
    assert location.shape == data_in.shape
    return output, location

def softmax(data_in):
    '''
    ## softmax layer
    (3095) -> (3095)
    '''
    m = np.amax(data_in)
    data_in -=m
    # print(data_in)
    # return data_in - np.log()
    # print(data_in)
    e = np.exp(data_in)
    s = np.sum(e)
    # print(e)
    # print(s)
    output = e/s
    return output 

def cross_entropy(data_in,label_vec):
    '''
    ## cross entropy as loss function. 
    (3095) -> 1
    '''
    l = np.log(data_in)
    return -np.dot(l, label_vec)

class LightCNN_9(object):
    def __init__(self, path=None):
        if path != None:    
            file = open(path, 'rb')
            data = file.read()
            file.close()
            self.__dict__ = pickle.loads(data)
        else:
            self.conv1_kernel = np.random.randn(5, 5, 1, 96)*np.sqrt(1/(5*5*1))
            self.conv1_bias = np.zeros((96), dtype=np.double)
            self.conv2a_kernel = np.random.randn(1, 1, 48, 96)*np.sqrt(1/(1*1*48))
            self.conv2a_bias = np.zeros((96), dtype=np.double)
            self.conv2_kernel = np.random.randn(3, 3, 48, 192)*np.sqrt(1/(3*3*48))
            self.conv2_bias = np.zeros((192), dtype=np.double)
            self.conv3a_kernel = np.random.randn(1, 1, 96, 192)*np.sqrt(1/(1*1*96))
            self.conv3a_bias = np.zeros((192), dtype=np.double)
            self.conv3_kernel = np.random.randn(3, 3, 96, 384)*np.sqrt(1/(3*3*96))
            self.conv3_bias = np.zeros((384), dtype=np.double)
            self.conv4a_kernel = np.random.randn(1, 1, 192, 384)*np.sqrt(1/(1*1*192))
            self.conv4a_bias = np.zeros((384), dtype=np.double)
            self.conv4_kernel = np.random.randn(3, 3, 192, 256)*np.sqrt(1/(3*3*192))
            self.conv4_bias = np.zeros((256), dtype=np.double)
            self.conv5a_kernel = np.random.randn(1, 1, 128, 256)*np.sqrt(1/(1*1*128))
            self.conv5a_bias = np.zeros((256), dtype=np.double)
            self.conv5_kernel = np.random.randn(3, 3, 128, 256)*np.sqrt(1/(3*3*128))
            self.conv5_bias =np.zeros((256), dtype=np.double)
            self.fc_weights = np.random.randn(8*8*128, 512)*np.sqrt(2/(8*8*128+3095))
            self.fc_bias = np.zeros((512), dtype=np.double)
            self.fcout_weights = np.random.randn(256, 3095)*np.sqrt(2/(256+3095))
            self.fcout_bias = np.zeros((3095), dtype=np.double)

            self.conv_kernel = [self.conv1_kernel,self.conv2a_kernel,self.conv2_kernel,self.conv3a_kernel, \
                                self.conv3_kernel,self.conv4a_kernel,self.conv4_kernel]  
            self.conv_bias = [self.conv1_bias,self.conv2a_bias,self.conv2_bias,self.conv3a_bias,\
                            self.conv3_bias,self.conv4a_bias,self.conv4_bias]
            self.fc_w = [self.fc_weights,self.fcout_weights]
            self.fc_b = [self.fc_bias,self.fcout_bias]

        return
    
    def forward(self, data):
        # forward
        pad1 = padding(data, 2)
        conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
        conv_input[:,:,0] = pad1

        conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
        mfm1, mfm1_location = mfm(conv1)

        pool1, pool1_location = pool(mfm1)

        conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
        mfm2a, mfm2a_location = mfm(conv2a)
        # TODO
        conv2 = conv(padding( mfm2a + pool1,1), self.conv2_kernel, self.conv2_bias)
        mfm2, mfm2_location = mfm(conv2)

        pool2, pool2_location = pool(mfm2)

        conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
        mfm3a, mfm3a_location = mfm(conv3a)

        conv3 = conv(padding(mfm3a + pool2, 1), self.conv3_kernel, self.conv3_bias)
        mfm3, mfm3_location = mfm(conv3)

        pool3, pool3_location = pool(mfm3)

        conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
        mfm4a, mfm4a_location = mfm(conv4a)

        conv4 = conv(padding(mfm4a + pool3, 1), self.conv4_kernel, self.conv4_bias)
        mfm4, mfm4_location = mfm(conv4)

        # conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
        # mfm5a, mfm5a_location = mfm(conv5a)

        # mfm5a = mfm5a + mfm4

        # conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
        # mfm5, mfm5_location = mfm(conv5)
            
        pool4, pool4_location = pool(mfm4)

        fc1 = fc(pool4, self.fc_weights, self.fc_bias)
        mfm_fc1, mfm_fc1_location = mfm_fc(fc1)

        fc2 = fc(mfm_fc1, self.fcout_weights, self.fcout_bias)

        softmax_output = softmax(fc2)
        # prediction = np.argmax(softmax)

        return softmax_output

    def train(self, data, label, epoch, min_batch_size, eta):

        def update_batch(batch_data,batch_label,batch_size, eta):

            # for i in min_batch_size:
            total_conv_w = []
            total_conv_b = []
            total_fc_w = []
            total_fc_b = []
            total_loss = 0.0

            # time1 = time.time()
            for data, label in zip(batch_data, batch_label):

                g_conv_w, g_conv_b, g_fc_w, g_fc_b, loss = backprob(data,label)

                total_loss += loss
                if len(total_conv_w) == 0:
                    total_conv_w = g_conv_w.copy()
                    total_conv_b = g_conv_b.copy()
                    total_fc_w = g_fc_w.copy()
                    total_fc_b = g_fc_b.copy()
                else:
                    total_conv_w = [ w1 + dw for w1, dw in zip(total_conv_w, g_conv_w)]
                    total_conv_b = [ b1 + db for b1, db in zip(total_conv_b, g_conv_b)]
                    total_fc_w = [ w1 + dw for w1, dw in zip(total_fc_w, g_fc_w)]
                    total_fc_b = [ b1 + db for b1, db in zip(total_fc_b, g_fc_b)]

                
            for w, g_w in zip(self.conv_kernel,total_conv_w):
                w -= eta* g_w / batch_size
            for b, g_b in zip(self.conv_bias,total_conv_b):
                b -= eta* g_b / batch_size

            for w, g_w in zip(self.fc_w,total_fc_w):
                w - eta* g_w / batch_size
            for b, g_b in zip(self.fc_b,total_fc_b):
                b - eta* g_b[:,0] / batch_size

            total_loss /= batch_size
            # time2 = time.time()
            print("====================")
            print("loss:",total_loss)
            # print("batch_time:",time2 - time1)
            return total_loss

        def SGD(train_image, train_label, test_image, test_label, epochs, batch_size, eta):
            '''Stochastic gradiend descent'''
            
            batch_num = 0

            for j in range(epochs):
                time1 = time.time()
                batch_total_loss = 0.0
                batch_data = [train_image[k:k+batch_size] for k in range(0, len(train_image), batch_size)]
                batch_label = [train_label[k:k+batch_size] for k in range(0, len(train_label), batch_size)]
                
                for mini_batch_image, mini_batch_label in zip(batch_data, batch_label):
                    batch_num += 1
                    
                    batch_total_loss += update_batch(mini_batch_image, mini_batch_label, batch_size, eta)
                    
                    # if batch_num % 10 == 0:
                        # print("after {0} training batch: accuracy is {1}/{2}".format(batch_num, self.evaluate(train_image[0:1000], train_label[0:1000]), len(train_image[0:1000])))
                # 一个epoch结束
                print(f"=============epoch{j}: average loss={batch_total_loss / batch_num}==========")
                self.save()
                batch_num = 0
                time2 = time.time()
                print("epoch time:", time2-time1)

                    # print("\rEpoch{0}:{1}/{2}".format(j+1, batch_num*mini_batch_size, len(train_image)), end=' ')
                
                # print("After epoch{0}: accuracy is {1}/{2}".format(j+1, self.evaluate(test_image, test_label), len(test_image)))

        def get_derivative_softmax(fc2_output, label_vec):
            '''
            fc2_output: (3095,1)
            return (3095,1)
            '''
            return fc2_output - label_vec

        def get_derivative_fcout(input_vec, fc_weights, fc_bias, bp_gradient):
            '''
            y = Wx + b
            bp_gradient: 从后续layer传来的梯度 (n, 1)
            fc2: input_vec: (256,) bp_gradient: (3095,), fc_weights:(256,3095)
            '''
            dw = np.matmul(input_vec[:,None],bp_gradient[:,None].transpose())
            assert dw.shape == fc_weights.shape
            db = bp_gradient
            db = db[:,None]
            assert db.shape == fc_bias[:,None].shape
            dx = np.matmul(fc_weights, bp_gradient[:,None])
            return dw,db,dx
        
        def get_derivate_mfm_fc1(location, bp_gradient):
            '''
            input_vec: (512,)
            bp_gradient: (256,1)
            '''
            tmp = np.vstack((bp_gradient,bp_gradient))
            assert tmp.shape[0] == 512
            output = tmp * location[:,None].astype(int) 
            assert output.shape == (512,1)
            return output

        def get_derivative_fc(input_vec, fc_weights, fc_bias, bp_gradient):
            '''
            y = Wx + b
            bp_gradient: 从后续layer传来的梯度 (n, 1)
            fc1: input_vec: (8*8*128,) bp_gradient: (512,1), fc_weights:(8*8*128,512)
            '''
            dw = np.matmul(input_vec[:,None],bp_gradient.transpose())
            assert dw.shape == fc_weights.shape
            db = bp_gradient
            assert db.shape == fc_bias[:,None].shape
            dx = np.matmul(fc_weights, bp_gradient)
            assert dx.shape == (8*8*128,1)
            return dw,db,dx
            
        def rot180(conv_filters):
            '''
            conv_filters: (fl, fw, h, n) -> (fl, fw, h, n)
            '''
            rot180_filters = np.flip(np.flip(conv_filters, 0), 1)
            return rot180_filters
        
        def get_derivative_conv(input_img, filter, filter_bias, bp_gradient, conv_output):
            # 对w求导的原理是：将bp_gradient的每一层[:,:,i]作为新的filter 对input_img的每一层[:,:,j]进行卷积
            filter_size = filter.shape[0]
            input_img_size= input_img.shape[0]
            input_img_channal = input_img.shape[2] 
            output_channal = conv_output.shape[-1]

            dw = np.zeros(filter.shape)            
            bp_gradient_filter_size = bp_gradient.shape[0]
            feature_size = input_img_size - bp_gradient_filter_size + 1
            # input_img2col = np.zeros((feature_size, feature_size, input_img_channal,bp_gradient_filter_size*bp_gradient_filter_size))

            # for i in range(feature_size):
            #     for j in range(feature_size):
            #         tmp = input_img[i:i+ bp_gradient_filter_size,j:j+ bp_gradient_filter_size,:]
            #         input_img2col[i,j,:,:] = tmp.reshape((bp_gradient_filter_size*bp_gradient_filter_size, input_img_channal)).T
            input_img2col = np.lib.stride_tricks.as_strided(input_img, shape=(feature_size, feature_size, input_img_channal, bp_gradient_filter_size, bp_gradient_filter_size), strides=(8*input_img_channal*input_img_size, 8*input_img_channal, 8, 8*input_img_channal*input_img_size, 8*input_img_channal))
            input_img2col = np.reshape(input_img2col, (feature_size, feature_size, input_img_channal,bp_gradient_filter_size*bp_gradient_filter_size))
            bp_gradient_filter = bp_gradient.reshape(( bp_gradient_filter_size* bp_gradient_filter_size, bp_gradient.shape[-1]))
            dw = np.matmul(input_img2col,bp_gradient_filter)
            
            assert dw.shape == filter.shape

            tmp_bias = np.zeros(input_img.shape[-1])
            rot_filter = rot180(filter).swapaxes(-2,-1)
            if filter.shape[0]!= 1:
                dx = conv(padding(bp_gradient,filter.shape[0]-1), rot_filter, tmp_bias)[1:-1,1:-1,:]
            else:
                dx = conv(bp_gradient, rot_filter, tmp_bias)

            db = np.sum(np.sum(bp_gradient,axis = 1),axis=0)[:None]
            assert db.shape == filter_bias.shape
            return dw,db,dx      

        def get_derivative_conv2pool(input_img, filter, filter_bias, bp_gradient):
            # 对w求导的原理是：将bp_gradient的每一层[:,:,i]作为新的filter 对input_img的每一层[:,:,j]进行卷积

            tmp_bias = np.zeros(input_img.shape[-1])
            rot_filter = rot180(filter).swapaxes(-2,-1)
            if filter.shape[0]!= 1:
                dx = conv(padding(bp_gradient,filter.shape[0]-1), rot_filter, tmp_bias)[1:-1,1:-1,:]
            else:
                dx = conv(bp_gradient, rot_filter, tmp_bias)

            return dx      

        def get_derivative_conv1(input_img, filter, filter_bias, bp_gradient, conv_output):
            '''
            第一层卷积计算反向传播梯度时所用的函数
            '''
            filter_size = filter.shape[0]
            input_img_size= input_img.shape[0]
            input_img_channal = 1 # input image 只有1个channal
            output_channal = conv_output.shape[-1]

            dw = np.zeros(filter.shape)            
            bp_gradient_filter_size = bp_gradient.shape[0]
            feature_size = input_img_size - bp_gradient_filter_size + 1
            # input_img2col = np.zeros((feature_size, feature_size, 1, bp_gradient_filter_size*bp_gradient_filter_size))
            # for i in range(feature_size):
            #     for j in range(feature_size):
            #         input_img2col[i,j,0,:] = input_img[i:i+ bp_gradient_filter_size,j:j+ bp_gradient_filter_size].flatten()
            input_img2col = np.lib.stride_tricks.as_strided(input_img, shape=(feature_size, feature_size, 1, bp_gradient_filter_size, bp_gradient_filter_size), strides=(8*input_img_channal*input_img_size, 8*input_img_channal, 8, 8*input_img_channal*input_img_size, 8*input_img_channal))
            input_img2col = np.reshape(input_img2col, (feature_size, feature_size, 1,bp_gradient_filter_size*bp_gradient_filter_size))
            bp_gradient_filter = bp_gradient.reshape(( bp_gradient_filter_size* bp_gradient_filter_size, bp_gradient.shape[-1]))
            dw = np.matmul(input_img2col,bp_gradient_filter)

            assert dw.shape == filter.shape
            db = np.sum(np.sum(bp_gradient,axis = 1),axis=0)[:None]
            assert db.shape == filter_bias.shape
            return dw,db  

        def get_derivative_pool(location, bp_gradient,pool_output):
            '''
            bp_gradient: (8*8*128,1)
            '''
            # print(bp_gradient.shape)
            bp_gradient = bp_gradient.reshape(pool_output.shape)
            bp_gradient = bp_gradient.repeat(2,axis=0).repeat(2,axis=1)
            # print(location.shape)
            output = bp_gradient * location
            # assert output.shape == 
            # print(output)
            return output
            # output = bp_gradient

        def get_derivative_mfm(location, bp_gradient):
            '''
            location
            '''
            # print(location.shape)
            tmp = np.concatenate((bp_gradient,bp_gradient),axis=-1)
            # print(tmp.shape)
            output = tmp * location.astype(int) 
            # print(output.shape)
            return output

        def backprob(data, label):

            # forward
            time_for1 = time.time()
            pad1 = padding(data, 2)
            conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
            conv_input[:,:,0] = pad1

            conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
            mfm1, mfm1_location = mfm(conv1)

            pool1, pool1_location = pool(mfm1)

            conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
            mfm2a, mfm2a_location = mfm(conv2a)
            # TODO
            conv2 = conv(padding( mfm2a + pool1,1), self.conv2_kernel, self.conv2_bias)
            mfm2, mfm2_location = mfm(conv2)

            pool2, pool2_location = pool(mfm2)

            conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
            mfm3a, mfm3a_location = mfm(conv3a)

            conv3 = conv(padding(mfm3a + pool2, 1), self.conv3_kernel, self.conv3_bias)
            mfm3, mfm3_location = mfm(conv3)

            pool3, pool3_location = pool(mfm3)

            conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
            mfm4a, mfm4a_location = mfm(conv4a)

            conv4 = conv(padding(mfm4a + pool3, 1), self.conv4_kernel, self.conv4_bias)
            mfm4, mfm4_location = mfm(conv4)

            # conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
            # mfm5a, mfm5a_location = mfm(conv5a)

            # mfm5a = mfm5a + mfm4

            # conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
            # mfm5, mfm5_location = mfm(conv5)
            
            pool4, pool4_location = pool(mfm4)

            fc1 = fc(pool4, self.fc_weights, self.fc_bias)
            mfm_fc1, mfm_fc1_location = mfm_fc(fc1)

            fc2 = fc(mfm_fc1, self.fcout_weights, self.fcout_bias)

            softmax_output = softmax(fc2)
            loss = cross_entropy(softmax_output,label)
            # print("loss:", loss)
            time_for2 = time.time()
            #print("forward_time:"+str(time_for2-time_for1))
            # ====================================================== backpropogation =============================================
            # time1 = time.time()
            time_back1 = time.time()
            g_softmax = get_derivative_softmax(softmax_output,label)

            g_fc2_w, g_fc2_b, g_fc2_x = get_derivative_fcout(mfm_fc1,self.fcout_weights,self.fcout_bias,g_softmax)

            g_mfm_fc1 = get_derivate_mfm_fc1(mfm_fc1_location, g_fc2_x)
            g_fc1_w, g_fc1_b, g_fc1_x = get_derivative_fc(pool4.flatten(), self.fc_weights, self.fc_bias, g_mfm_fc1)

            g_pool4 = get_derivative_pool(pool4_location, g_fc1_x, pool4)
            
            # g_mfm5 = get_derivative_mfm( mfm5_location, g_pool4)
            # g_conv5_w, g_conv5_b, g_conv5_x = get_derivative_conv(padding(mfm5a,1),self.conv5_kernel, self.conv5_bias, g_mfm5,conv5)

            # # self.conv5_kernel -= 0.0001*g_conv5_w
            # # self.conv5_bias -= 0.0001*g_conv5_b
            
            # g_mfm5a = get_derivative_mfm( mfm5a_location, g_conv5_x)
            # g_conv5a_w, g_conv5a_b, g_conv5a_x = get_derivative_conv(mfm4,self.conv5a_kernel, self.conv5a_bias, g_mfm5a,conv5a)
            # # self.conv5a_kernel -= 0.0001*g_conv5a_w
            # self.conv5a_bias -= 0.0001*g_conv5a_b

            # # # # #
            time_conv4_begin = time.time()
            g_mfm4 = get_derivative_mfm( mfm4_location, g_pool4)
            g_conv4_w, g_conv4_b, g_conv4_x = get_derivative_conv(padding(mfm4a+pool3,1),self.conv4_kernel, self.conv4_bias, g_mfm4,conv4)
            g_conv4_x_po = get_derivative_conv2pool(padding(pool3,1),self.conv4_kernel, self.conv4_bias, g_mfm4)

            # g_conv4_w = np.add(g_conv4_w, g_conv4_w_po)
            # g_conv4_b = np.add(g_conv4_b, g_conv4_b_po)

            g_mfm4a = get_derivative_mfm( mfm4a_location, g_conv4_x)
            g_conv4a_w, g_conv4a_b, g_conv4a_x = get_derivative_conv(pool3,self.conv4a_kernel, self.conv4a_bias, g_mfm4a,conv4a)
            time_conv4_end = time.time()

            g_pool3 = get_derivative_pool(pool3_location, g_conv4a_x + g_conv4_x_po, pool3)
            
            time_conv3_begin = time.time()
            # # #
            g_mfm3 = get_derivative_mfm( mfm3_location, g_pool3)
            g_conv3_w, g_conv3_b, g_conv3_x = get_derivative_conv(padding(mfm3a+pool2,1),self.conv3_kernel, self.conv3_bias, g_mfm3,conv3)
            g_conv3_x_po = get_derivative_conv2pool(padding(pool2,1),self.conv3_kernel, self.conv3_bias, g_mfm3)

            g_mfm3a = get_derivative_mfm( mfm3a_location, g_conv3_x)
            g_conv3a_w, g_conv3a_b, g_conv3a_x = get_derivative_conv(pool2,self.conv3a_kernel, self.conv3a_bias, g_mfm3a,conv3a)
            time_conv3_end = time.time()

            g_pool2 = get_derivative_pool(pool2_location, g_conv3a_x+ g_conv3_x_po,pool2)

            # # #
            time_conv2_begin = time.time()
            g_mfm2 = get_derivative_mfm( mfm2_location, g_pool2)
            g_conv2_w, g_conv2_b, g_conv2_x = get_derivative_conv(padding(mfm2a+pool1,1),self.conv2_kernel, self.conv2_bias, g_mfm2,conv2)
            g_conv2_x_po = get_derivative_conv2pool(padding(pool1,1),self.conv2_kernel, self.conv2_bias, g_mfm2)

            g_mfm2a = get_derivative_mfm( mfm2a_location, g_conv2_x)
            g_conv2a_w, g_conv2a_b, g_conv2a_x = get_derivative_conv(pool1,self.conv2a_kernel, self.conv2a_bias, g_mfm2a,conv2a)
            time_conv2_end = time.time()

            g_pool1 = get_derivative_pool(pool1_location, g_conv2a_x + g_conv2_x_po ,pool1)

            time_conv1_begin = time.time()
            g_mfm1 = get_derivative_mfm( mfm1_location, g_pool1)
            g_conv1_w, g_conv1_b= get_derivative_conv1(padding(data, 2),self.conv1_kernel, self.conv1_bias, g_mfm1,conv1)
            time_back2 = time.time()
            # print("conv4:",time_conv4_end - time_conv4_begin)
            # print("conv3:",time_conv3_end - time_conv3_begin)
            # print("conv2:",time_conv2_end - time_conv2_begin)
            # print("conv1:",time_back2 - time_conv1_begin)

            # print("back:", time_back2 - time_back1)
            
            #print("backward_time:"+str(time_back2-time_back1))
            g_conv_w = [ g_conv1_w,g_conv2a_w, g_conv2_w, g_conv3a_w, g_conv3_w, g_conv4a_w, g_conv4_w]
            g_conv_b = [ g_conv1_b,g_conv2a_b, g_conv2_b, g_conv3a_b, g_conv3_b, g_conv4a_b, g_conv4_b]
            
            g_fc_w = [g_fc1_w, g_fc2_w]
            g_fc_b = [g_fc1_b, g_fc2_b]

            return g_conv_w, g_conv_b, g_fc_w, g_fc_b, loss
        
        # for i in range(500):
        #     eta = 0.0001
        #     total_conv_w, total_conv_b, total_fc_w,total_fc_b,loss = backprob(data,label)
        #     print(loss)
        #     for w, g_w in zip(self.conv_kernel,total_conv_w):
        #         w -= eta* g_w 
        #     for b, g_b in zip(self.conv_bias,total_conv_b):
        #         b -= eta* g_b

        #     for w, g_w in zip(self.fc_w,total_fc_w):
        #         w - eta* g_w 
        #     for b, g_b in zip(self.fc_b,total_fc_b):
        #         b - eta* g_b[:,0] 
            # update_batch(data,label,1,0.0001)
        SGD(data, label, None, None, epoch, min_batch_size, eta)

    def test(self, path_match, path_mismatch):
        dirs = os.listdir(path_match)
        FP = 0
        FN = 0
        TP = 0
        TN = 0
        sim_thr = 0.0001
        dir1 = os.listdir(path_match)
        for dir in dir1:
            files = os.listdir(os.path.join(path_match,dir))
            #print(files)
            image1 = io.imread(os.path.join(path_match,dir,files[0]), as_gray=True)
            image1 = transform.resize(image1, (image_width, image_height))
            predict1 = self.forward(image1)
            #print(predict1)
            image2 = io.imread(os.path.join(path_match,dir,files[1]), as_gray=True)
            image2 = transform.resize(image2, (image_width, image_height))
            predict2 = self.forward(image2)
            #print(predict2)
            similarity = np.inner(predict1,predict2) / (np.linalg.norm(predict1) * np.linalg.norm(predict2))
            #similarity = np.corrcoef(predict1, predict2)[0,1]
            print(similarity)
            if similarity >= sim_thr:
                TP += 1
            else:
                FN += 1
        dir2 = os.listdir(path_mismatch)
        print("mis:")
        for dir in dir2[:len(dir1)]:
            files = os.listdir(os.path.join(path_mismatch,dir))
            #print(files)
            image1 = io.imread(os.path.join(path_mismatch,dir,files[0]), as_gray=True)
            image1 = transform.resize(image1, (image_width, image_height))
            predict1 = self.forward(image1)
            #print(predict1)
            image2 = io.imread(os.path.join(path_mismatch,dir,files[1]), as_gray=True)
            image2 = transform.resize(image2, (image_width, image_height))
            predict2 = self.forward(image2)
            # print(predict2)
            # similarity = np.corrcoef(predict1, predict2)[0,1]
            # print(similarity)
            similarity = np.inner(predict1,predict2) / (np.linalg.norm(predict1) * np.linalg.norm(predict2))
            print(similarity)
            if similarity >= sim_thr:
                FP += 1
            else:
                TN += 1
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN) 
        F1 = 2 / (1/precision + 1/recall)
        print("precision:", precision)
        print("recall:", recall)
        print("F1:", F1)
        # result = []
        # for image1, image2 in data1, data2:

        #     predict1 = self.forward(image1)
        #     predict2 = self.forward(image2)
        #     cos_similarity = np.dot(predict1, predict2) / (np.dot(predict1, predict1) * np.dot(predict2, predict2))
        #     if cos_similarity**2 >= 0.5:
        #         result.append(1)
        #     else:
        #         result.append(0)

        # return result

    
    def save(self):
        file = open('LightCNN9_model.bin', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return

if __name__ == "__main__":
    # path = 'D:/USTC_ML/大作业/Face_verification-Numpy/LFW_dataset/*/*/*.jpg'
    # path = './LFW/*/*/*.jpg'
    path1 = './train_image1/*/*/*.jpg'
    path2 = './train_image2/*/*/*.jpg'
    path3 = './train_image3/*/*/*.jpg'
    time1= time.time()
    #train_data, train_label = traindata_loader(path1, path2, path3)
    time2= time.time()

    print(f"Data loading finished.{time2 - time1}")
    model = LightCNN_9('LightCNN9_model.bin')
    # print(train_label.shape)
    # a = np.zeros((train_label.shape[0],3095),dtype=int)
    # a[:,:train_label.shape[1]] = train_label

    # model.train(train_data,a,4,64,0.0005)
    path_match = './test_dataset/match_pairs'
    path_mismatch = './test_dataset/mismatch_pairs'
    model.test(path_match, path_mismatch)

    