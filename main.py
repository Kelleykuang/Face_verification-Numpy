import numpy as np
from skimage import io, transform
import glob
import pickle
import time
import logging
import os

from forward_layers import *
from backward_layers import *
from utils import traindata_loader

image_width = 128
image_height = 128


class LightCNN_9(object):
    def __init__(self, path=None):
        if path != None:    # Load existing model
            file = open(path, 'rb')
            data = file.read()
            file.close()
            self.__dict__ = pickle.loads(data)
        else:   # Initialize
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
    def save(self):
        file = open('LightCNN9_model.bin', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return
    def forward(self, data):
        '''
        Forward propagation and output the feature of image
        (image_width, image_height) -> (256)
        ''' 
        pad1 = padding(data, 2)
        conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
        conv_input[:,:,0] = pad1

        conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
        mfm1, mfm1_location = mfm(conv1)

        pool1, pool1_location = pool(mfm1)

        conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
        mfm2a, mfm2a_location = mfm(conv2a)
        conv2 = conv(padding( mfm2a,1), self.conv2_kernel, self.conv2_bias)
        mfm2, mfm2_location = mfm(conv2)

        pool2, pool2_location = pool(mfm2)

        conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
        mfm3a, mfm3a_location = mfm(conv3a)

        conv3 = conv(padding(mfm3a, 1), self.conv3_kernel, self.conv3_bias)
        mfm3, mfm3_location = mfm(conv3)

        pool3, pool3_location = pool(mfm3)

        conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
        mfm4a, mfm4a_location = mfm(conv4a)

        conv4 = conv(padding(mfm4a, 1), self.conv4_kernel, self.conv4_bias)
        mfm4, mfm4_location = mfm(conv4)

        conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
        mfm5a, mfm5a_location = mfm(conv5a)

        conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
        mfm5, mfm5_location = mfm(conv5)
            
        pool4, pool4_location = pool(mfm5)

        fc1 = fc(pool4, self.fc_weights, self.fc_bias)
        mfm_fc1, mfm_fc1_location = mfm_fc(fc1)

        return mfm_fc1

    def train(self, data, label, epoch, min_batch_size, eta):
        '''
        Train the model with backpropagation
        '''
        
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
                    
                print(f"=============epoch{j}: average loss={batch_total_loss / batch_num}==========")
                self.save()
                batch_num = 0
                time2 = time.time()
                print("epoch time:", time2-time1)

        def update_batch(batch_data,batch_label,batch_size, eta):
            total_conv_w = []
            total_conv_b = []
            total_fc_w = []
            total_fc_b = []
            total_loss = 0.0
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
                w -= eta* g_w / batch_size
            for b, g_b in zip(self.fc_b,total_fc_b):
                b -= eta* g_b[:,0] / batch_size

            total_loss /= batch_size
            print("====================")
            print("loss:",total_loss)
            return total_loss
                
        def backprob(data, label):
            
            # Forward propagation

            pad1 = padding(data, 2)
            conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
            conv_input[:,:,0] = pad1

            conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
            mfm1, mfm1_location = mfm(conv1)

            pool1, pool1_location = pool(mfm1)

            conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
            mfm2a, mfm2a_location = mfm(conv2a)

            conv2 = conv(padding( mfm2a,1), self.conv2_kernel, self.conv2_bias)
            mfm2, mfm2_location = mfm(conv2)

            pool2, pool2_location = pool(mfm2)

            conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
            mfm3a, mfm3a_location = mfm(conv3a)

            conv3 = conv(padding(mfm3a, 1), self.conv3_kernel, self.conv3_bias)
            mfm3, mfm3_location = mfm(conv3)

            pool3, pool3_location = pool(mfm3)

            conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
            mfm4a, mfm4a_location = mfm(conv4a)

            conv4 = conv(padding(mfm4a, 1), self.conv4_kernel, self.conv4_bias)
            mfm4, mfm4_location = mfm(conv4)

            conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
            mfm5a, mfm5a_location = mfm(conv5a)

            conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
            mfm5, mfm5_location = mfm(conv5)
                
            pool4, pool4_location = pool(mfm5)

            fc1 = fc(pool4, self.fc_weights, self.fc_bias)
            mfm_fc1, mfm_fc1_location = mfm_fc(fc1)

            fc2 = fc(mfm_fc1, self.fcout_weights, self.fcout_bias)

            softmax_output = softmax(fc2)
            loss = cross_entropy(softmax_output,label)
            
            g_softmax = get_derivative_softmax(softmax_output,label)

            # Backward propagation
        
            g_fc2_w, g_fc2_b, g_fc2_x = get_derivative_fcout(mfm_fc1,self.fcout_weights,self.fcout_bias,g_softmax)

            g_mfm_fc1 = get_derivate_mfm_fc1(mfm_fc1_location, g_fc2_x)
            g_fc1_w, g_fc1_b, g_fc1_x = get_derivative_fc(pool4.flatten(), self.fc_weights, self.fc_bias, g_mfm_fc1)

            g_pool4 = get_derivative_pool(pool4_location, g_fc1_x, pool4)
            
            g_mfm5 = get_derivative_mfm( mfm5_location, g_pool4)
            g_conv5_w, g_conv5_b, g_conv5_x = get_derivative_conv(padding(mfm5a,1),self.conv5_kernel, self.conv5_bias, g_mfm5,conv5)
            
            g_mfm5a = get_derivative_mfm( mfm5a_location, g_conv5_x)
            g_conv5a_w, g_conv5a_b, g_conv5a_x = get_derivative_conv(mfm4,self.conv5a_kernel, self.conv5a_bias, g_mfm5a,conv5a)

            g_mfm4 = get_derivative_mfm( mfm4_location, g_conv5a_x)
            g_conv4_w, g_conv4_b, g_conv4_x = get_derivative_conv(padding(mfm4a,1),self.conv4_kernel, self.conv4_bias, g_mfm4,conv4)

            g_mfm4a = get_derivative_mfm( mfm4a_location, g_conv4_x)
            g_conv4a_w, g_conv4a_b, g_conv4a_x = get_derivative_conv(pool3,self.conv4a_kernel, self.conv4a_bias, g_mfm4a,conv4a)
        
            g_pool3 = get_derivative_pool(pool3_location, g_conv4a_x,pool3)
            
            g_mfm3 = get_derivative_mfm( mfm3_location, g_pool3)
            g_conv3_w, g_conv3_b, g_conv3_x = get_derivative_conv(padding(mfm3a,1),self.conv3_kernel, self.conv3_bias, g_mfm3,conv3)
            
            g_mfm3a = get_derivative_mfm( mfm3a_location, g_conv3_x)
            g_conv3a_w, g_conv3a_b, g_conv3a_x = get_derivative_conv(pool2,self.conv3a_kernel, self.conv3a_bias, g_mfm3a,conv3a)
            
            g_pool2 = get_derivative_pool(pool2_location, g_conv3a_x,pool2)

            g_mfm2 = get_derivative_mfm( mfm2_location, g_pool2)
            g_conv2_w, g_conv2_b, g_conv2_x = get_derivative_conv(padding(mfm2a,1),self.conv2_kernel, self.conv2_bias, g_mfm2,conv2)
            g_mfm2a = get_derivative_mfm( mfm2a_location, g_conv2_x)
            g_conv2a_w, g_conv2a_b, g_conv2a_x = get_derivative_conv(pool1,self.conv2a_kernel, self.conv2a_bias, g_mfm2a,conv2a)
        
            g_pool1 = get_derivative_pool(pool1_location, g_conv2a_x,pool1)

            g_mfm1 = get_derivative_mfm( mfm1_location, g_pool1)
            g_conv1_w, g_conv1_b= get_derivative_conv1(padding(data, 2),self.conv1_kernel, self.conv1_bias, g_mfm1,conv1)
            
            g_conv_w = [ g_conv1_w,g_conv2a_w, g_conv2_w, g_conv3a_w, g_conv3_w, g_conv4a_w, g_conv4_w, g_conv5a_w, g_conv5_w ]
            g_conv_b = [ g_conv1_b,g_conv2a_b, g_conv2_b, g_conv3a_b, g_conv3_b, g_conv4a_b, g_conv4_b, g_conv5a_b, g_conv5_b ]
            
            g_fc_w = [g_fc1_w, g_fc2_w]
            g_fc_b = [g_fc1_b, g_fc2_b]

            return g_conv_w, g_conv_b, g_fc_w, g_fc_b, loss
        
        SGD(data, label, None, None, epoch, min_batch_size, eta)
        
    def test(self, path_match, path_mismatch):
        FP = 0
        FN = 0
        TP = 0
        TN = 0
        sim_thr = 0.078
        allsim1 = 0
        allsim2 = 0

        dir1 = os.listdir(path_match)
        for dir in dir1: # Test match pairs
            files = os.listdir(os.path.join(path_match,dir))
            image1 = io.imread(os.path.join(path_match,dir,files[0]), as_gray=True)
            image1 = transform.resize(image1[25:-25][25:-25], (image_width, image_height))
            image1 = (image1-np.amin(image1))/(np.amax(image1)-np.amin(image1))
            predict1 = self.forward(image1)
            image2 = io.imread(os.path.join(path_match,dir,files[1]), as_gray=True)
            image2 = transform.resize(image2[25:-25][25:-25], (image_width, image_height))
            image2 = (image2-np.amin(image2))/(np.amax(image2)-np.amin(image2))
            predict2 = self.forward(image2)            
            similarity = np.corrcoef(predict1, predict2)[0,1]
            allsim1 += similarity
            if similarity >= sim_thr:
                TP += 1
            else:
                FN += 1

        dir2 = os.listdir(path_mismatch)
        for dir in dir2[:len(dir1)]: # Test mismatch pairs
            files = os.listdir(os.path.join(path_mismatch,dir))
            image1 = io.imread(os.path.join(path_mismatch,dir,files[0]), as_gray=True)
            image1 = transform.resize(image1[25:-25][25:-25], (image_width, image_height))
            image1 = (image1-np.amin(image1))/(np.amax(image1)-np.amin(image1))
            predict1 = self.forward(image1)
            image2 = io.imread(os.path.join(path_mismatch,dir,files[1]), as_gray=True)
            image2 = transform.resize(image2[25:-25][25:-25], (image_width, image_height))
            image2 = (image2-np.amin(image2))/(np.amax(image2)-np.amin(image2))
            predict2 = self.forward(image2)
            similarity = np.corrcoef(predict1, predict2)[0,1]
            allsim2 += similarity
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
        print("sim1:",allsim1/len(dir1),"sim2:",allsim2/len(dir1),"delta:",allsim1/len(dir1)-allsim2/len(dir1))
    
    def TA_test(self, path):
        dirs = os.listdir(path)
        sim_thr = -0.0926
        result = open("result.txt","w")
        for dir in dirs:
            files = os.listdir(os.path.join(path,dir))
            image1 = io.imread(os.path.join(path,dir,files[0]), as_gray=True)
            image1 = transform.resize(image1[25:-25][25:-25], (image_width, image_height))
            image1 = (image1-np.amin(image1))/(np.amax(image1)-np.amin(image1))
            predict1 = self.forward(image1)
            image2 = io.imread(os.path.join(path,dir,files[1]), as_gray=True)
            image2 = transform.resize(image2[25:-25][25:-25], (image_width, image_height))
            image2 = (image2-np.amin(image2))/(np.amax(image2)-np.amin(image2))
            predict2 = self.forward(image2)
            similarity = np.corrcoef(predict1, predict2)[0,1]
            if similarity >= sim_thr:
                result.write(str(1)+'\n')
            else:
                result.write(str(0)+'\n')
        result.close()
        return

if __name__ == "__main__":

    # Data loading
    path = './train_image/*/*/*.jpg'
    time1= time.time()
    train_data, train_label = traindata_loader(path)
    time2= time.time()
    print(f"Data loading finished.{time2 - time1}")

    # Training
    model = LightCNN_9()
    model.train(train_data, train_label, epoch=20, min_batch_size=64, eta=0.0005)
    
    # Testing
    path_match = './test_dataset/match_pairs'
    path_mismatch = './test_dataset/mismatch_pairs'
    model.test(path_match, path_mismatch)
    
    # model.TA_test(path='./test_dataset/')

    