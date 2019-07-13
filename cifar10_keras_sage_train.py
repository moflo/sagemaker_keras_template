'''
# CIFAR10 SageMaker Keras Training Fuction.

Use the SageMaker SDK Script syntax to train a Keras CNN.

'''

from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
import keras

from cifar10_keras_sage_model import get_model


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
        
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(train_dir):
    
    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)

    return x_train, y_train


def get_val_data(val_dir):
    
    x_val = np.load(os.path.join(val_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(val_dir, 'y_val.npy'))
    print('x val', x_val.shape,'y val', y_val.shape)

    return x_val, y_val
 
def main(args):
    x_train, y_train = get_train_data(args.train)
    x_val, y_val = get_val_data(args.val)
    
    # Convert class vectors to binary class matrices.
    num_classes = 10

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = get_model(x_train.shape[1:],num_classes)

    # initialize global variables in session (required during save)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
            #   callbacks=[time_callback],
              validation_data=(x_val, y_val),
              shuffle=True)
    

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    model_version_dir = os.path.join(args.model_dir, f'model-{time.time() * 10}')
    os.makedirs(model_version_dir, exist_ok=True)

    # Save the h5 model
    model_h5 = os.path.join(model_version_dir, 'model.h5')
    model.save(model_h5)

    saved_model = os.path.join(model_version_dir, 'saved_model')
    os.makedirs(saved_model, exist_ok=True)

    # Error : model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True) TypeError: save_weights() got an unexpected keyword argument 'save_format'
    # tf.contrib.saved_model.save_keras_model(model, saved_model)

    # https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/keras_script_mode_pipe_mode_horovod/source_dir/cifar10_keras_main.py
    # Deprication warning: simple_save (from tensorflow.python.saved_model.simple_save) is deprecated and will be removed in a future version.
    # Error : model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True) TypeError: save_weights() got an unexpected keyword argument 'save_format'
    # tf.saved_model.simple_save(
    #         tf.Session(),
    #         saved_model,
    #         inputs={'inputs': model.input},
    #         outputs={t.name: t for t in model.outputs})

    # Export the model to a SavedModel
    # tf.keras.experimental.export_saved_model(model, saved_model)




if __name__ == "__main__":
            
    args, _ = parse_args()
    main(args)

    