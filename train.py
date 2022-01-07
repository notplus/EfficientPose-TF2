'''
Description: 
Author: notplus
Date: 2022-01-07 14:18:25
LastEditors: notplus
LastEditTime: 2022-01-07 16:54:45
FilePath: /train.py

Copyright (c) 2022 notplus
'''

import tensorflow as tf
import matplotlib.pyplot as plt

import tfrecord
import config as cfg
from model.efficient_pose import create_efficient_pose

train_dataset = tfrecord.get_dataset(cfg.TRAIN_TFREC, cfg.BATCH_SIZE)
val_dataset = tfrecord.get_dataset(cfg.VAL_TFREC, cfg.BATCH_SIZE)

model = create_efficient_pose(arch='EfficientPoseRTLite', input_size=224)

name = 'efficient_pose_rt_lite'

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "./weight/" + name, save_best_only=True
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True
)
csv_logger_cb = tf.keras.callbacks.CSVLogger("./log/" + name + ".csv")

## Complile model
initial_learning_rate = 0.001
# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate, decay_steps=20000, alpha=0.0
# )
opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, decay=1e-4)
# opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)#, amsgrad=True)
model.compile(optimizer=opt, run_eagerly=True)

hist = model.fit(
    train_dataset,
    epochs=250,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb]
)

# if not os.path.exists('model'):
#     os.makedirs('model')
print('Trainig finished.')

# Loss History
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('rate')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./log/loss.jpg')