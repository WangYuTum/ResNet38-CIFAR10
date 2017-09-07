from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
import data_utils as dt
from core import resnet38

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.8

train_data_params = {'data_path': 'data/cifar-10-batches-py/',
                     'batch_size': 128,
                     'mode': 'Train'}
dataset = dt.CIFAR10(train_data_params)

params = {'batch_size': 128,
          'decay_rate': 0.0005,
          #'feed_path': 'data/trained_weights/empty.npy',
          'feed_path': 'data/saved_weights/modelA_momen_e1e2e3_90.npy',
          'save_path': 'data/saved_weights/',
          'tsboard_save_path': 'data/tsboard/'}

train_ep = 91
# val_step_iter = 100
save_ep = 10

#with tf.Session() as sess:
with tf.Session(config=config_gpu) as sess:
    res38 = resnet38.ResNet38(params['feed_path'])
    save_path = params['save_path']
    batch_size = params['batch_size']

    train_img = tf.placeholder(tf.float32, shape=[batch_size, 32, 32,
                                                  3])
    train_label = tf.placeholder(tf.int64, shape=[batch_size])

    [train_op, total_loss, train_acc, correct_preds] = res38.train(image=train_img, label=train_label, params=params)

    save_dict_op = res38._var_dict
    TrainLoss_sum = tf.summary.scalar('train_loss', total_loss)
    TrainAcc_sum = tf.summary.scalar('train_acc', train_acc)
    # ValLoss_sum = tf.summary.scalar('val_loss', total_loss)
    # ValAcc_sum = tf.summary.scalar('val_acc', train_acc)
    Train_summary = tf.summary.merge_all()
    # Val_summary = tf.summary.merge([ValLoss_sum, ValAcc_sum])

    writer = tf.summary.FileWriter(params['tsboard_save_path']+'modelA_momentum', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    num_iters = np.int32(50000 / batch_size) + 1
    print('Start training...')
    for epoch in range(train_ep):
        print('Eopch %d'%epoch)
        for iters in range(num_iters):
            next_images, next_labels = dataset.next_batch()
            train_feed_dict = {train_img: next_images, train_label: next_labels}
            [train_op_, total_loss_, train_acc_, Train_summary_] = sess.run([train_op, total_loss, train_acc, Train_summary], train_feed_dict)
            writer.add_summary(Train_summary_, iters)
            if iters % 50 == 0 and iters !=0:
                print('Iter %d loss: %f'%(iters, total_loss_))
        if epoch % save_ep == 0 and epoch !=0:
            print('Save trained weight after epoch: %d'%epoch)
            save_npy = sess.run(save_dict_op)
            save_path = params['save_path']
            if len(save_npy.keys()) != 0:
                save_name = 'modelA_momen_e1e2e3_%d.npy'%(epoch+90)
                save_path = save_path + save_name
                np.save(save_path, save_npy)
        # Shuffle and flip dataset
        dataset.shuffle()
        # dataset.flip()


