# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:59:37 2019

@author: 61995
"""

import pandas
import tensorflow as tf

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fista import *
import time

start_time = time.time()
data = scio.loadmat('Video_train.mat')
M = data['M']
AA = np.zeros([24,32,32,600])
for j in range(24):
    for i in range(600):
        #M[j,i][M[j,i]<0.01]=0
        AA[j,:,:,i] = M[j,i]
lamda1 = 0.3
lamda2 = 0.3
beta1 = 0.5
beta2= 0.5
n1 = 0
n2 = 0
gamx1 = 1
gamx2 = 1
gamu1 =1
gamu2 =1
sigma = 1e-15
d_x1 = 150
d_x2 =60
d_u1 = 40
d_u2 = 3
d_input1 = 144
batch_size=4
y = tf.placeholder(tf.float32, shape=(4,20,20,1)) #image

v = tf.placeholder(tf.float32, shape=(4,2,2,600,d_input1))
x1_r = tf.placeholder(tf.float32, shape=(4,2,2,d_x1))#x1 record
x1_up = tf.placeholder(tf.float32, shape=(4,2,2,d_x1))



X1_r = tf.placeholder(tf.float32, shape=(d_x1,2*2*600*batch_size))
X1 = tf.placeholder(tf.float32, shape=(d_x1,2*2*600*batch_size))
U1 = tf.placeholder(tf.float32, shape=(d_u1,600*batch_size))
inputimg = tf.placeholder(tf.float32, shape=(d_input1,2*2*600*4))
Xp = tf.placeholder(tf.float32, shape=(d_x1,600*4))

X2_r = tf.placeholder(tf.float32, shape=(d_x2,2*2*600*batch_size))
X2 = tf.placeholder(tf.float32, shape=(d_x2,2*2*600*batch_size))
U2 = tf.placeholder(tf.float32, shape=(d_u2,600*batch_size))
inputimg2 = tf.placeholder(tf.float32, shape=(d_u1,2*2*600*4))
Xp2 = tf.placeholder(tf.float32, shape=(d_x2,600*4))




x2_r = tf.placeholder(tf.float32, shape=(4,2,2,60))#x2 record
cinput = tf.placeholder(tf.float32, shape=(d_x1,d_input1))
binput = tf.placeholder(tf.float32, shape=(d_u1,d_x1))
c2input = tf.placeholder(tf.float32, shape=(d_x2,d_u1))
b2input = tf.placeholder(tf.float32, shape=(d_u2,d_x2))
#u2_p= tf.placeholder(tf.float32, shape=(4,1,1,3))#u_prediction_t-1
u1_p_r= tf.placeholder(tf.float32, shape=(4,1,1,40))
with tf.variable_scope("varibles", reuse=tf.AUTO_REUSE) as vr:
    x1 = tf.get_variable('x1', shape = [4,2,2,d_x1],  initializer=tf.zeros_initializer)
    u1 = tf.get_variable('u1', shape = [4,1,1,d_u1], initializer=tf.zeros_initializer)
    x2 = tf.get_variable('x2', shape = [4,2,2,60], initializer=tf.zeros_initializer)
    u2 = tf.get_variable('u2', shape = [4,1,1,3], initializer=tf.zeros_initializer)
    C_2 = tf.get_variable('C2', shape = [60,40], initializer=tf.initializers.random_uniform(-0.5,0.5))
    A_2 = tf.get_variable('A2', shape = [60,60], initializer=tf.initializers.random_uniform(0,0.1))
    B_2 = tf.get_variable('B2', shape = [3,60], initializer=tf.initializers.random_uniform())
    v_r = [v for v in tf.all_variables() if v.name.startswith(vr.name)]
    
    
    

    yy =tf.extract_image_patches(y,ksizes=[1,12,12,1],strides=[1,8,8,1],rates =[1,1,1,1],padding="VALID")
    yyt =tf.transpose(yy,[3,1,2,0])
with tf.variable_scope("parameters_1", reuse=tf.AUTO_REUSE) as enc:
    
    x1c1=tf.layers.dense(
        inputs=x1,
        units=144,
        use_bias=False,
        kernel_initializer = tf.initializers.random_uniform(-0.5,0.5),
        name='C1',
        reuse=None
    )#(1,4,4,144)) 
    x1a1 = tf.layers.dense(
        inputs=x1_r,
        units=d_x1,
        use_bias=False,
        kernel_initializer = tf.initializers.random_uniform(0,0.1),
        name='A1',
        reuse=None
    )#(1,4,4,150))
    x1a1t =tf.transpose(x1a1,[3,1,2,0])
    
#    x1pool = 4*tf.layers.average_pooling2d(
#        tf.abs(x1),
#        pool_size=2,
#        strides=2,
#        padding='valid',
#        data_format='channels_last',
#        name=None
#    )#(1, 2, 2, 150)
    x1pool = tf.abs(x1[:,0,0,:])+tf.abs(x1[:,0,1,:])+tf.abs(x1[:,1,0,:])+tf.abs(x1[:,1,1,:])
    x1pool = tf.expand_dims(x1pool,axis=1)
    x1pool = tf.expand_dims(x1pool,axis=1)
   
    u1b1=tf.layers.dense(
        inputs=u1,
        units=d_x1,
        use_bias=False,
        kernel_initializer = tf.initializers.random_uniform(),
        name = 'B1'
    )#(1, 2, 2, 150)
    gama1 = (gamu1*(1+tf.exp(-u1b1))/2)
    gama1t = tf.transpose(gama1,[3,1,2,0])
    lossbathch1 = tf.reduce_sum((yy-x1c1)**2,axis = -1) 
    lossbathch1 = tf.reduce_sum(lossbathch1,axis = -1) 
    lossbathch1 = tf.reduce_sum(lossbathch1,axis = -1)
    logit11 = (tf.losses.mean_squared_error(yy,x1c1))*2*2*144#+tf.reduce_sum((gama1)*tf.sqrt(x1pool**2 + sigma))#+gamx1*tf.reduce_sum(tf.sqrt(x1**2 + sigma))#tf.reduce_sum(tf.log(x1**2+1))#gamx1*tf.reduce_sum(tf.abs(x1))
    logit12 = lamda1*tf.reduce_sum(tf.sqrt((x1a1-x1)**2+sigma))#lamda1*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x1, labels=x1_r))#lamda1*tf.reduce_sum(tf.sqrt((x1_r-x1)**2+sigma))#lamda1*tf.reduce_sum(tf.sqrt((x1_r-x1)**2+sigma))#+gamx1*tf.losses.huber_loss(tf.zeros(tf.shape(x1)),x1)#lamda1*tf.losses.absolute_difference(x1_r,x1)+gamx1*tf.reduce_sum(tf.abs(x1))#lamda1*tf.reduce_sum(tf.losses.huber_loss(x1_r,x1))+ gamx1*tf.losses.huber_loss(tf.zeros(tf.shape(x1)),x1)
    logit13 = tf.reduce_sum(gama1 *x1pool)#+ beta1 * tf.reduce_sum(tf.sqrt(u1**2+sigma))#tf.reduce_sum(tf.log(u1**2+1))#beta1 * tf.reduce_sum(tf.abs(u1))#beta1 * tf.losses.huber_loss(tf.zeros(tf.shape(u1)),u1)#(tf.losses.mean_squared_error(x1pool,u1b1))#tf.reduce_sum(gama1*x1pool)+ beta1 * tf.losses.huber_loss(tf.zeros(tf.shape(u1)),u1)
    x1_grad =tf.gradients(logit11+logit12,x1)
    p_r = [v for v in tf.all_variables() if v.name.startswith(enc.name)]
##############parameters1 update#########################3
#    X1c1=tf.layers.dense(
#        inputs=X1,
#        units=144,
#        use_bias=False,
#        name='C1',
#        reuse=True
#    )#(1,4,4,144))
#    X1a1 = tf.layers.dense(
#        inputs=X1_r,
#        units=d_x1,
#        use_bias=False,
#        name='A1',
#        reuse=True
#    )#(1,4,4,150))
#    X1pool = tf.abs(X1[:,0,0,:,:])+tf.abs(X1[:,0,1,:,:])+tf.abs(X1[:,1,0,:,:])+tf.abs(X1[:,1,1,:,:])
#    X1pool = tf.expand_dims(X1pool,axis=1)
#    X1pool = tf.expand_dims(X1pool,axis=1)
#    U1b1=tf.layers.dense(
#        inputs=U1,
#        units=d_x1,
#        use_bias=False,
#        name = 'B1',
#        reuse = True
#    )#(1, 2, 2, 150)
#    
    costc1 = tf.reduce_sum((inputimg-tf.matmul(tf.transpose(p_r[0]),X1))**2)
    costa1 = lamda1*tf.reduce_sum(tf.abs(X1-tf.matmul(tf.transpose(p_r[1]),X1_r)))
    costb1 = tf.reduce_sum((1+tf.exp(-tf.matmul(tf.transpose(p_r[2]),U1)))/2*Xp)
    
    
    costc2 = tf.reduce_sum((inputimg2-tf.matmul(tf.transpose(C_2),X2))**2)
    costa2 = lamda2*tf.reduce_sum(tf.abs(X2-tf.matmul(tf.transpose(A_2),X2_r)))
    costb2 = tf.reduce_sum((1+tf.exp(-tf.matmul(tf.transpose(B_2),U2)))/2*Xp2)
#    x2c2=tf.layers.dense(
#        inputs=x2,
#        units=40,
#        name='C2',
#        use_bias=False,
#        reuse=None
#    )#(1, 2, 2, 40)
#    x2a2 = tf.layers.dense(
#        inputs=x2_r,
#        units=60,
#        use_bias=False,
#        name='A2',
#        reuse=None
#    )#(1,2,2,60))
#
#    x2pool = tf.layers.average_pooling2d(
#        tf.abs(x2),
#        pool_size=2,
#        strides=2,
#        padding='valid',
#        data_format='channels_last',
#        name=None
#    )#(1, 1, 1, 60)
#    u2b2=tf.layers.dense(
#        inputs=u2,
#        units=60,
#        use_bias=False,
#        name = 'B2'
#        )#(1, 1, 1, 60)
#    gama2 = gamu2/2*(1+tf.exp(-u2b2))/2
#    logit21 = tf.losses.mean_squared_error(u1,x2c2)*2*2*40+gamx2*tf.reduce_sum(tf.sqrt(x2**2+sigma))#tf.reduce_sum(tf.log(x2**2+1))#*tf.reduce_sum(tf.abs(x2))
#    logit22 =lamda2*tf.reduce_sum(tf.sqrt((x2_r-x2)**2+sigma))#lamda2*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x2, labels=x2_r))#lamda2*tf.reduce_sum(tf.sqrt((x2_r-x2)**2+sigma))#+gamx2*tf.losses.huber_loss(tf.zeros(tf.shape(x2)),x2)#lamda2*tf.reduce_sum(tf.losses.huber_loss(x2_r,x2))+gamx2*tf.losses.huber_loss(tf.zeros(tf.shape(x2)),x2)#lamda2*tf.losses.absolute_difference(x2_r,x2)+gamx2*tf.reduce_sum(tf.abs(x2))
#    logit23 = tf.reduce_sum(gama2*x2pool)+ beta2 * tf.reduce_sum(tf.sqrt(u2**2+sigma))#tf.reduce_sum(tf.log(u2**2+1))#beta2 * tf.losses.huber_loss(tf.zeros(tf.shape(u2)),u2)#tf.losses.mean_squared_error(u2b2,x2pool)#tf.reduce_sum(gama2*x2pool)+ beta2 * tf.losses.huber_loss(tf.zeros(tf.shape(u2)),u2)
#    
#    u2b2_p=tf.layers.dense(
#        inputs=u2_p,
#        units=60,
#        use_bias=False,
#        name = 'B2',
#        reuse = True
#        )#(1, 1, 1, 60)  
#    gama2_p=gamu2/2*tf.exp(-u2b2_p)/2
#    mask2 = gama2_p<lamda2
#    mask2 = tf.cast(mask2,dtype=tf.float32)
#    mask2 = tf.tile(mask2,[1,2,2,1])
#    x2_p =  x2a2*mask2
#    u1_p=tf.layers.dense(
#        inputs=x2_p,
#        units=40,
#        use_bias=False,
#        name='C2',
#        reuse=True
#    )#(1, 2, 2, 40)
#    logit24 = n2*tf.losses.mean_squared_error(u2_p,u2)
#    logit14 = n1*tf.losses.mean_squared_error(u1_p_r,u1)
    

#train_op_x1 = tf.train.ProximalGradientDescentOptimizer(learning_rate = 0.1,l1_regularization_strength=0.2).minimize(logit11+logit12,var_list = [x1])
train_op_x1 = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(logit11+logit12,var_list = [x1])

train_op_u1 = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(logit13,var_list = [u1])
#train_op_u1 = tf.train.ProximalAdagradOptimizer(learning_rate = 0.01,l1_regularization_strength=0.05).minimize(logit13,var_list = [u1])
##train_op_u1 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(logit11+logit12+logit13,var_list = [u1,x1])    
#train_op_x2 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(logit21+logit22,var_list = [x2])
#train_op_u2 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(logit23,var_list = [u2])
##train_op_u2 = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(logit21+logit22+0.1*logit23+0.1*logit24,var_list = [u2,x2])
u1_int = u1.assign(u1_p_r)
#u2_int = u2.assign(u2_p) 
x1_int = x1.assign(x1_r) 
x1_update = x1.assign(x1_up) 
cup = p_r[0].assign(cinput)
bup = p_r[2].assign(binput)
c2up = C_2.assign(c2input)
b2up = B_2.assign(b2input)
#x2_int = x2.assign(tf.zeros([1,2,2,60])) 
 
train_op_1 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(costc1,var_list = [p_r[0]])#[p_r[0],p_r[1],p_r[2],p_r[3],p_r[4],p_r[5]]

train_op_2 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(costa1,var_list = [p_r[1]] )  #[p_r[8],p_r[9],p_r[6],p_r[7],p_r[10],p_r[11]] 

train_op_3 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(costb1,var_list = [p_r[2]] )  




train_op_11 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(costc2,var_list = [C_2])#[p_r[0],p_r[1],p_r[2],p_r[3],p_r[4],p_r[5]]

train_op_22 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(costa2,var_list = [A_2] )  #[p_r[8],p_r[9],p_r[6],p_r[7],p_r[10],p_r[11]] 

train_op_33 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(costb2,var_list = [B_2] )   
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    x1_restore = np.zeros([d_x1,2,2,4])#x1 record
    x2_restore = np.zeros([d_x2,2,2,4])
  #  x2_restore = np.zeros([4,2,2,60])#x2 record
#    u2_restore = np.zeros([1,1,1,3])#u2 record
#    output = np.zeros([1,3,600])
    u1_restore = np.zeros([d_u1,1,1,4])
    lossX1 = []
    lossU1 = []
# #   predict_img = np.zeros([600,4,4,144])
# #   rimg = np.zeros([600,4,4,144])
    C = np.transpose(np.array(sess.run(p_r[0])))#(144,150)ct
    C11 = np.transpose(C/(np.sqrt(np.sum(C**2,0))))
    sess.run(cup, feed_dict={cinput:C11,})
    C = np.transpose(np.array(sess.run(p_r[0])))
    
    A = np.transpose(np.array(sess.run(p_r[1])))
    B = np.transpose(np.array(sess.run(p_r[2])))
    B11 = np.transpose(B/(np.sqrt(np.sum(B**2,0))))
    sess.run(bup, feed_dict={binput:B11,})
    B = np.transpose(np.array(sess.run(p_r[2])))


    
    for tt in range(50):
        print(tt)
        x_ind = np.random.randint(12)
        y_ind = np.random.randint(12)
        seq_ind = np.random.randint(24,size = batch_size)
        vdo = AA[seq_ind,x_ind:x_ind+20,y_ind:y_ind+20,:]

        
 #       #u1_p_restore=np.array(sess.run(u1_p, feed_dict={y: img,x1_r: x1_restore,x2_r: x2_restore,u2_p: u2_restore,}))
 #       #u1_p_restore = u1_restore
 #       #u2_restore = u2_restore

 #############################################33       
#        X_0 = np.zeros([d_x1,2,2,600,batch_size])
#        X_1 = np.zeros([d_x1,2,2,600,batch_size])
#        U_1 = np.zeros([d_u1,600,batch_size])
#        Y = np.zeros([144,2,2,600,batch_size])
#        img_st =np.zeros([batch_size,2,2,600,144])
#        loss =[]
#        lossx =[]
#        lossu =[]
#        for t in range(600) :
#            if t % 50 == 0:
#                loss =[]
#                lossx =[]
#                lossu =[]
#                print(t)
#            img = np.zeros([batch_size,20,20,1])
#            img[:,:,:,0] = vdo[:,:,:,t]
#            u1_p_restore = np.zeros([batch_size,1,1,d_u1])
##        u2_restore = np.zeros([1,1,1,3])
#
#
##            sess.run(u1_int, feed_dict={u1_p_r:u1_p_restore,})
###        sess.run(u2_int, feed_dict={u2_p: u2_restore,}) 
##            sess.run(x1_int, feed_dict={x1_r:x1_restore,})
#            
#            imgout =  np.array(sess.run(yyt, feed_dict={y:img,}))#(144,2,2,4)
#            Y[:,:,:,t,:]=imgout
#            #x_hat = np.array(sess.run(x1a1t, feed_dict={y: img,x1_r: x1_restore,}))#(150,2,2,4)a1txt-1t
#            C = np.transpose(np.array(sess.run(p_r[0], feed_dict={y:img,})))#(144,150)ct
#            C11 = np.transpose(C/(np.sqrt(np.sum(C**2,0))))
#            sess.run(cup, feed_dict={cinput:C11,})
#            
#            A = np.transpose(np.array(sess.run(p_r[1], feed_dict={y:img,})))
#            x_hat = np.zeros([d_x1,2,2,4])
#            for rl in range(2):
#                for cl in range(2):
#                    x_hat[:,rl,cl,:] = np.matmul(A,x1_restore[:,rl,cl,:])
#            B = np.transpose(np.array(sess.run(p_r[2], feed_dict={y:img,})))
#            B11 = np.transpose(B/(np.sqrt(np.sum(B**2,0))))
#            sess.run(bup, feed_dict={binput:B11,})
#            
#            
#            
#########################fista function############################################
#            
#            cause,state_cell,x1_restore = fista(x_hat,C,B,A,imgout,lamda1,t,lossx,lossu,loss,d_x1,d_u1,batch_size)
#            
#            
#            
#   
################################################################################################################                    
#            reg=0
#            for rl in range(2):
#                for cl in range(2):
#                    x1_restore[:,rl,cl,:] = (state_cell[reg]['xk'])
#                    reg+=1
#            X_1[:,:,:,t,:]=x1_restore
#            rimg = state_cell[0]["y"]
#            rpred = np.matmul(C,state_cell[0]['xk'])
#            
#            U_1[:,t,:] = cause['uk']
#            aushow = U_1
#            bxshow = X_1[:,0,0,:,:]
#        X_0[:,:,:,1:,:] = X_1[:,:,:,:-1,:]
#        np.save("Uapi.npy", U_1)
 
 
        Y = np.zeros([144,2,2,600,batch_size])
        for t in range(600):
            img = np.zeros([batch_size,20,20,1])
            img[:,:,:,0] = vdo[:,:,:,t]
            imgout =  np.array(sess.run(yyt, feed_dict={y:img,}))
            Y[:,:,:,t,:]=imgout
            
        
        X_1,U_1,X_0,lossx,lossu,loss= inference(Y,lamda1,d_input1,d_x1,d_u1,batch_size,sess,A,B,C,x1_restore)
        aushow = U_1
        bxshow = X_1[:,0,0,:,:]
        Xinput = np.reshape(X_1,[d_x1,2*2*600*4])
        X0input = np.reshape(X_0,[d_x1,2*2*600*4])
        Uinput = np.reshape(U_1,[d_u1,600*4])
        Yinput = np.reshape(Y,[144,2*2*600*4])
        Xpool = np.zeros([d_x1,600,4])
        for rl in range(2):
            for cl in range(2):
                Xpool = Xpool+np.abs(X_1[:,rl,cl,:,:])
        Xpoolinput = np.reshape(Xpool,[d_x1,600*4])
        
 ################################################################33
 
 
 
 
        #parameters optimaize###############33
        train_stepc1=tf.contrib.opt.ScipyOptimizerInterface(costc1, var_list=[p_r[0]], method='CG',   options={'maxiter': 2})
        train_stepc1.minimize(session=sess, feed_dict={inputimg: Yinput, X1: Xinput,})
#        sess.run(train_op_1, feed_dict={inputimg: Yinput, X1: Xinput,})
#        sess.run(train_op_1, feed_dict={inputimg: Yinput, X1: Xinput,})
        C = np.transpose(np.array(sess.run(p_r[0])))#(144,150)ct
        C11 = np.transpose(C/(np.sqrt(np.sum(C**2,0))))
        sess.run(cup, feed_dict={cinput:C11,})
        C = np.transpose(np.array(sess.run(p_r[0])))
        train_stepa1=tf.contrib.opt.ScipyOptimizerInterface(costa1, var_list=[p_r[1]], method='CG',   options={'maxiter': 2})
        train_stepa1.minimize(session=sess, feed_dict={X1: Xinput,X1_r:X0input,})
#        sess.run(train_op_2, feed_dict={X1: Xinput,X1_r:X0input,})
#        sess.run(train_op_2, feed_dict={X1: Xinput,X1_r:X0input,})
        A = np.transpose(np.array(sess.run(p_r[1])))
        train_stepb1=tf.contrib.opt.ScipyOptimizerInterface(costb1, var_list=[p_r[2]], method='CG',   options={'maxiter': 2})
        train_stepb1.minimize(session=sess, feed_dict={Xp:Xpoolinput,U1:Uinput})
#        sess.run(train_op_3, feed_dict={Xp:Xpoolinput,U1:Uinput})
#        sess.run(train_op_3, feed_dict={Xp:Xpoolinput,U1:Uinput})
        B = np.transpose(np.array(sess.run(p_r[2])))
        B11 = np.transpose(B/(np.sqrt(np.sum(B**2,0))))
        sess.run(bup, feed_dict={binput:B11,})
        B = np.transpose(np.array(sess.run(p_r[2])))
        
        
        #################################3
        lossX1.append(sess.run(costc1, feed_dict={inputimg: Yinput, X1: Xinput,}))
        lossU1.append(sess.run(costb1, feed_dict={Xp:Xpoolinput,U1:Uinput}))
        np.save("Uapi.npy", U_1)
        
        
        
    np.save("C.npy", C11)
    np.save("A.npy", A)
    np.save("B.npy", B11)
    inputu1 = np.ones([d_u1,2,2,600,batch_size])
    C2 = np.transpose(np.array(sess.run(C_2)))#(144,150)ct
    C22 = np.transpose(C2/(np.sqrt(np.sum(C2**2,0))))
    sess.run(c2up, feed_dict={c2input:C22,})
    C2 = np.transpose(np.array(sess.run(C_2)))
        
    A2 = np.transpose(np.array(sess.run(A_2)))
    B2 = np.transpose(np.array(sess.run(B_2)))
    B22 = np.transpose(B2/(np.sqrt(np.sum(B2**2,0))))
    sess.run(b2up, feed_dict={b2input:B22,})
    B2 = np.transpose(np.array(sess.run(B_2)))
    C=np.transpose(np.load("C.npy"))
    A=np.transpose(np.load("A.npy"))
    B=np.transpose(np.load("B.npy"))
    for tt in range(50):
        print("layer2")
        print(tt)
        seq_ind = np.random.randint(24,size = batch_size)
#        C = np.transpose(np.array(sess.run(p_r[0])))
#        A = np.transpose(np.array(sess.run(p_r[1])))
#        B = np.transpose(np.array(sess.run(p_r[2])))



        for ii in range(2):
            for jj in range(2):

                x_ind = int(0+ii*12)
                y_ind = int(0+jj*12)

                vdo = AA[seq_ind,x_ind:x_ind+20,y_ind:y_ind+20,:]#(4,20,20,600)
                Y = np.ones([144,2,2,600,batch_size])*0.14
                for t in range(600):
                    x1_restore = np.zeros([d_x1,2,2,4])
                    img = np.zeros([batch_size,20,20,1])
                    img[:,:,:,0] = vdo[:,:,:,t]
                    imgout =  np.array(sess.run(yyt, feed_dict={y:img,}))
                    Y[:,:,:,t,:]=imgout
                X_1,U_1,X_0,lossx,lossu,loss= inference(Y,lamda1,d_input1,d_x1,d_u1,batch_size,sess,A,B,C,x1_restore)
                inputu1[:,ii,jj,:,:]=U_1
        print("loaded")
        X_2,U_2,X_0,lossx,lossu,loss= inference(inputu1,lamda2,d_u1,d_x2,d_u2,batch_size,sess,A2,B2,C2,x2_restore)
        
        X2input = np.reshape(X_2,[d_x2,2*2*600*batch_size])
        X02input = np.reshape(X_0,[d_x2,2*2*600*batch_size])
        U2input = np.reshape(U_2,[d_u2,600*batch_size])
        Y2input = np.reshape(inputu1,[d_u1,2*2*600*batch_size])
        Xpool2 = np.zeros([d_x2,600,batch_size])
        for rl in range(2):
            for cl in range(2):
                Xpool2 = Xpool2+np.abs(X_2[:,rl,cl,:,:])
        Xpoolinput2 = np.reshape(Xpool2,[d_x2,600*batch_size])
        train_stepc2=tf.contrib.opt.ScipyOptimizerInterface(costc2, var_list=[C_2], method='CG',   options={'maxiter': 2})
        train_stepc2.minimize(session=sess, feed_dict={inputimg2: Y2input, X2: X2input,})
        #sess.run(train_op_11, feed_dict={inputimg2: Y2input, X2: X2input,})
        #sess.run(train_op_11, feed_dict={inputimg2: Y2input, X2: X2input,})
        C2 = np.transpose(np.array(sess.run(C_2)))#(144,150)ct
        C22 = np.transpose(C2/(np.sqrt(np.sum(C2**2,0))))
        sess.run(c2up, feed_dict={c2input:C22,})
        C2 = np.transpose(np.array(sess.run(C_2)))
        train_stepa2=tf.contrib.opt.ScipyOptimizerInterface(costa2, var_list=[A_2], method='CG',   options={'maxiter': 2})
        train_stepa2.minimize(session=sess, feed_dict={X2: X2input,X2_r:X02input,})
#        sess.run(train_op_22, feed_dict={X2: X2input,X2_r:X02input,})
#        sess.run(train_op_22, feed_dict={X2: X2input,X2_r:X02input,})
        A2 = np.transpose(np.array(sess.run(A_2)))

        train_stepb2=tf.contrib.opt.ScipyOptimizerInterface(costb2, var_list=[B_2], method='CG',   options={'maxiter': 2})
        train_stepb2.minimize(session=sess, feed_dict={Xp2:Xpoolinput2,U2:U2input})
#        sess.run(train_op_33, feed_dict={Xp2:Xpoolinput2,U2:U2input})
#        sess.run(train_op_33, feed_dict={Xp2:Xpoolinput2,U2:U2input})
        B2 = np.transpose(np.array(sess.run(B_2)))
        B22 = np.transpose(B2/(np.sqrt(np.sum(B2**2,0))))
        sess.run(b2up, feed_dict={b2input:B22,})
        B2 = np.transpose(np.array(sess.run(B_2)))
        np.save("U_2.npy", U_2)            
                    
                    
                
                
        
        
            
                
                
                
                
                    
            
                
##############newton-BASED#######################          
#            for i in range(1):
#            
#                train_stepx1=tf.contrib.opt.ScipyOptimizerInterface(logit11+logit12, var_list=[x1], method='L-BFGS-B')
#                train_stepx1.minimize(session=sess, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#                train_stepu1=tf.contrib.opt.ScipyOptimizerInterface(logit13, var_list=[u1], method='L-BFGS-B')
#                train_stepu1.minimize(session=sess, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#
#
##            x1_loss = sess.run(logit12+logit11, feed_dict={y: img,x1_r: x1_restore,})
##            if np.abs((x1_loss_r - x1_loss)/x1_loss)> 0.000001 or i<10:
##                sess.run(train_op_x1, feed_dict={y: img,x1_r: x1_restore,}) 
##                x1_loss_r = x1_loss
#            #lossx1.append(sess.run(logit12, feed_dict={y: img,x1_r: x1_restore,}))
##            u1_loss = sess.run(logit13, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
##            if np.abs((u1_loss_r - u1_loss)/u1_loss)> 0.000001 or i<10:
##                sess.run(train_op_u1, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
##                u1_loss_r = u1_loss
#                if t % 50 == 0:
#                    
#                    lossx1.append(sess.run(logit12+logit11, feed_dict={y: img,x1_r: x1_restore,}))
#                    lossu1.append(sess.run(logit13, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,}))
#                    
#            x1_restore = np.array(sess.run(x1, feed_dict={y: img,x1_r: x1_restore,}))        
#            x1show = x1_restore[0,:,:,:]      
#            X_1[:,:,:,t,:] = x1_restore 
#            U_1[:,:,:,t,:] = np.array(sess.run(u1))
#            img_st[:,:,:,t,:]=np.array(sess.run(yy,feed_dict={y: img,}))
#         
#
#            UU1 = U_1*(U_1>0.1)
#            UU1 = UU1[:,0,0,:,:]

###############SGD-BASED#######################      
#            nxs = np.sum(x1_restore>2.2204e-15)
#            for i in range(50):
#                
#                sess.run(train_op_x1, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#                #gradv = sess.run(x1_grad, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#                sess.run(train_op_u1, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#                cost11=sess.run(logit11+logit12, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#                x_proximal= np.array(sess.run(x1))
#                L1 = 1
#                l1_s = np.array(sess.run(gama1))*L1#np.max(x_proximal)*0.001#0.2#np.array(sess.run(gama1))
#                stop_linesearch = np.zeros([5,1])
##                while 1:
##                    x_input = (x_proximal-l1_s)*((x_proximal>l1_s).astype(float))
##                    x_input = (x_proximal-l1_s)*((x_proximal>l1_s).astype(float))
##                    x_input = x_input + (x_proximal+l1_s)*((x_proximal<-l1_s).astype(float))
##                    gradv = sess.run(x1_grad, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
##                    sess.run(x1_update, feed_dict={x1_up:x_input,})
##                    temp1 =  np.sum(sess.run(logit11+logit12, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,}))
##                    sess.run(x1_update, feed_dict={x1_up:x_proximal,})
##                    
##                    temp2=cost11+np.sum(gradv)*(x_proximal-x_input)+1/(2*L1)*np.sum((x_proximal-x_input)**2)
##                    if temp1<=temp2:
##                        break
##                    else:
##                        L1=L1/2
##                sess.run(x1_update, feed_dict={x1_up:x_input,})
#                
##                nxs_r = nxs
##                nxs = np.sum(x1_restore>2.2204e-15)
#                
#                
#                sess.run(x1_update, feed_dict={x1_up:x_input,})
#                x1_restore = x_input
#                sess.run(train_op_u1, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
#                l1_u=0.015
#                u_proximal= np.array(sess.run(u1))
#                u_input = (u_proximal-l1_u)*((u_proximal>l1_u).astype(float))
#                u_input = u_input + (u_proximal+l1_u)*((u_proximal<-l1_u).astype(float))
#                sess.run(u1_int, feed_dict={u1_p_r:u_input,})
##                nxs_r = nxs
##                nxs = np.sum(x_input>2.2204e-15)
##                if  0:#np.abs(nxs_r-nxs)/nxs_r > 0.01:
##                    break
#
# 
#
#                
#                
#                    
#                
#                
#                
#
#
##            x1_loss = sess.run(logit12+logit11, feed_dict={y: img,x1_r: x1_restore,})
##            if np.abs((x1_loss_r - x1_loss)/x1_loss)> 0.000001 or i<10:
##                sess.run(train_op_x1, feed_dict={y: img,x1_r: x1_restore,}) 
##                x1_loss_r = x1_loss
#            #lossx1.append(sess.run(logit12, feed_dict={y: img,x1_r: x1_restore,}))
##            u1_loss = sess.run(logit13, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
##            if np.abs((u1_loss_r - u1_loss)/u1_loss)> 0.000001 or i<10:
##                sess.run(train_op_u1, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,})
##                u1_loss_r = u1_loss
#                if t % 50 == 0:
#                    
#                    lossx1.append(sess.run(logit11, feed_dict={y: img,x1_r: x1_restore,}))
#                    lossu1.append(sess.run(logit13, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,}))
#                    
#             
##            x1_restore = np.array(sess.run(x1, feed_dict={y: img,x1_r: x1_restore,}))
#            x1show = x1_restore[0,:,:,:]
#            predict_img = np.array(sess.run(x1c1, feed_dict={x1: x1_restore,}))[0,:,:,:]
#            rimg = np.array(sess.run(yy, feed_dict={y:img,}))[0,:,:,:]
#            X_1[:,:,:,t,:] = x1_restore 
#            U_1[:,:,:,t,:] = np.array(sess.run(u1))
#            img_st[:,:,:,t,:]=np.array(sess.run(yy,feed_dict={y: img,}))
#         
#
##            UU1 = U_1*(U_1>0.1)
##            UU1 = UU1[:,0,0,:,:]
##           
#            b = X_1[:,0,0,:,:]
#            #a = U_1*(np.abs(U_1)>0.0001)
#            a = U_1[:,0,0,:,:]
#        X_10 = np.zeros([4,2,2,600,d_x1])
#        U_10 = np.zeros([4,1,1,600,40])   
#        
#        X_10 = np.zeros([4,2,2,600,d_x1])
#        U_10 = np.zeros([4,1,1,600,40])
#        X_10[:,:,:,1:,:] = X_1[:,:,:,:-1,:]
#        U_10[:,:,:,1:,:] = U_1[:,:,:,:-1,:]
#        train_stepc1=tf.contrib.opt.ScipyOptimizerInterface(costc1+costa1+costb1, var_list=[p_r[0],p_r[1],p_r[2]], method='CG',   options={'maxiter': 6})
#        train_stepc1.minimize(session=sess, feed_dict={v: img_st, X1: X_1,X1_r:X_10,U1:U_1})
#        loss.append(sess.run(costc1+costa1+costb1, feed_dict={v: img_st, X1: X_1,X1_r:X_10,U1:U_1}))
##        train_stepa1=tf.contrib.opt.ScipyOptimizerInterface(costa1, var_list=[p_r[1]], method='CG',   options={'maxiter': 2})
##        train_stepa1.minimize(session=sess, feed_dict={v: img_st, X1: X_1,X1_r:X_10,U1:U_1})
##        train_stepb1=tf.contrib.opt.ScipyOptimizerInterface(costb1, var_list=[p_r[2]], method='CG',   options={'maxiter': 2})
##        train_stepb1.minimize(session=sess, feed_dict={v: img_st, X1: X_1,X1_r:X_10,U1:U_1})
#        lossX1.append(sess.run(logit12+logit11, feed_dict={y: img,x1_r: x1_restore,}))
#        lossU1.append(sess.run(logit13, feed_dict={y: img,x1_r: x1_restore,u1_p_r:u1_p_restore,}))
#        np.save("U.npy", U_1)
#        np.save("img.npy", rimg)
#        np.save("pimg.npy", predict_img)
#        np.save("loss.npy", loss)
#        np.save("lossu.npy", lossu1)
#        np.save("lossx.npy", lossx1)
#        
#        
##        if t % 50 == 0:        
##            lossu2 = []
##            lossx2= []
##        x2_loss_r = 0
##        u2_loss_r = 0
##        for i in range(100):
##            sess.run(train_op_x2, feed_dict={x2_r: x2_restore, u2_p: u2_restore,})
##            sess.run(train_op_u2, feed_dict={x2_r: x2_restore, u2_p: u2_restore,})
##            
##            
##            
###            x2_loss = sess.run(logit22+logit21,  feed_dict={x2_r: x2_restore, u2_p: u2_restore,})
###            if np.abs((x2_loss_r - x2_loss)/x2_loss)> 0.001 or i < 10:
###                sess.run(train_op_x2, feed_dict={x2_r: x2_restore, u2_p: u2_restore,})
###                x2_loss_r = x2_loss
###            u2_loss = sess.run(logit23, feed_dict={x2_r: x2_restore, u2_p: u2_restore,})
###            if np.abs((u2_loss_r - u2_loss)/u2_loss)> 0.001 or i<10:
###                sess.run(train_op_u2, feed_dict={x2_r: x2_restore, u2_p: u2_restore,})
###                u2_loss_r = u2_loss
##            if t % 50 == 0:
##                lossx2.append(sess.run(logit22+logit21,  feed_dict={x2_r: x2_restore, u2_p: u2_restore,}))
##                lossu2.append(sess.run(logit23, feed_dict={x2_r: x2_restore, u2_p: u2_restore,}))
##        
##        #sess.run(train_op_1, feed_dict={y: img,x1_r: x1_restore,})
##        #sess.run(train_op_2, feed_dict={x2_r: x2_restore,})
##        
##        
##        
##        #sess.run(train_op_1, feed_dict={y: img,x1_r: x1_restore,})
##        #sess.run(train_op_2, feed_dict={x2_r: x2_restore,})
#        #loss.append(sess.run(logit23, feed_dict={x2_r: x2_restore, u2_p: u2_restore,}))
#         
#        
#        
#        
#        
#        
##        x1_restore = np.array(sess.run(x1, feed_dict={y: img,x1_r: x1_restore,}))
##        a =  np.array(sess.run(x1c1, feed_dict={x1: x1_restore,}))[0,:,:,:]
##        predict_img[t,:,:,:] = a 
##        b=  np.array(sess.run(yy, feed_dict={y:img,}))[0,:,:,:]
##        rimg[t,:,:,:] = b
##        x2_restore = np.array(sess.run(x2, feed_dict={y: img,x1_r: x1_restore,}))
##        u2_restore = np.array(sess.run(u2))
##        u1_restore = np.array(sess.run(u1))
##        c1 = np.array(sess.run(p_r[0]))
##        B = np.array(sess.run(p_r[4], feed_dict={y: img,x1_r: x1_restore,}))
##        output[:,:,t] = u2_restore[0,0,:,:]
##        print('echo:')
##        print(t)
##        
##        output1=output[0,0,:]
##        output2=output[0,1,:]
##        output3=output[0,2,:]
##        x11 = x1_restore[0,:,:,:]
##        x22 = x2_restore[0,:,:,:]
##        u11 = u1_restore[0,:,:,:]
##        u22 = u2_restore[0,:,:,:]
##    predict_img = np.reshape(predict_img,(600,48,48))
##    rimg = np.reshape(rimg,(600,48,48))
##    output = output[0,:,:]
##    ax = plt.subplot(111, projection='3d')
##    ax.scatter(output1, output2, output3, c='y')
##    plt.show()
##     a= np.load('U.npy')
##     pca = PCA(n_components=3, svd_solver='full')
##     c=pca.fit_transform(b)