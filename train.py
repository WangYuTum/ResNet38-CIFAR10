from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import tensorflow as tf
import data_utils as dt
from core.resnet38 import ResNet38


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 1.0

train_data_params = {'data_path': 'data/cifar-10-batches-py/',
                     'batch_size': 64,
                     'mode': 'Train'}
dataset = dt.CIFAR10(train_data_params)

params = {'num_class': 10,
          'batch_size': 64,
          'decay_rate': 0.0002,
          'feed_path': '../data/trained_weights/empty.npy',
          'save_path': '../data/saved_weights/',
          'tsboard_save_path': '../data/tsboard/'}

train_ep = 1
val_step_iter = 5
save_ep = 1

with tf.Session(config=config_gpu) as sess:
    res38 = ResNet38(params['feed_path'])
    save_path = params['save_path']
    batch_size = params['batch_size']

    train_img = tf.placeholder(tf.float32, shape=[batch_size, 32, 32,
                                                  3])
    train_label = tf.placeholder(tf.int32, shape=[batch_size])

    [train_op, total_loss, train_acc, correct_preds] = res38.train(image=train_img, label=train_label, params=params)

    save_dict_op = res38._var_dict
    TrainLoss_sum = tf.summary.scalar('train_loss', loss)
    TrainAcc_sum = tf.summary.scalar('train_acc', train_acc)
    Train_summary = tf.summary.merge([TrainLoss_sum, TrainAcc_sum])

    train_writer = tf.summary.FileWriter(params['tsboard_save_path']+'train', sess.graph)
    val_writer = tf.summary.FileWriter(params['tsboard_save_path']+'val', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    num_iters = 45000 / batch_size + 1
    print('Start training...')
    for epoch in range(train_ep):
        print('Eopch %d'%epoch)
        for iters in range(num_iters):
            print('iter %d'%iters)
            if iters % val_step_iter == 0 and iters != 0:
                # run validation
                next_images, next_labels = train_dataset._get_TestPool()
                val_feed_dict = {train_img: next_images, train_label:
                                 next_labels}
                [val_accuracy_, total_loss_] = sess.run([train_acc,total_loss], val_feed_dict)
                val_writer.add_summary(Train_summary, iters)

                # test save weight, comment later
                print('Save trained weight after epoch: %d'%epoch)
                save_npy = sess.run(save_dict_op)
                if len(save_npy.keys()) != 0:
                    save_name = 'CIFAR10_ResNet38_%d.npy'%(iters+1)
                    save_path = save_path + save_name
                    np.save(save_path, save_npy)
                # test save weight ended
            else:
                # run training
                next_images, next_labels = train_dataset.next_batch()
                train_feed_dict = {train_img: next_batch_image, train_label: next_batch_label}
                [train_op_, total_loss_, train_acc_, merged_summary_] = sess.run([train_op, total_loss, train_acc, merged_summary], train_feed_dict)
                train_writer.add_summary(Train_summary, iters)
                print('Iter %d loss: %f'%(iters, total_loss_))
        # if epoch % save_ep == 0 and epoch !=0:
        #    print('Save trained weight after epoch: %d'%epoch)
        #    save_npy = sess.run(save_dict_op)
        # Shuffle dataset
        train_dataset.shuffle()


