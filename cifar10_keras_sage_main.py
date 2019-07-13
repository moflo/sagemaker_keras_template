'''
# CIFAR10 SageMaker Keras Local Test.

Use the SageMaker SDK to train a CNN on the CIFAR10 using Keras.

'''

from __future__ import print_function
import argparse
import os
import sys
import subprocess

import keras
from keras.datasets import cifar10
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
import numpy as np

import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

def test():
   print('Download CIFAR10 data, natively on local machine.')
   # # Load CIFAR10 data via Keras, split between train and test sets:
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   print('x_train shape:', x_train.shape)
   print(x_train.shape[0], 'train samples')
   print(x_test.shape[0], 'test samples')


   data_dir = os.path.join(os.getcwd(), 'data')
   train_dir = os.path.join(os.getcwd(), 'data/train')
   val_dir = os.path.join(os.getcwd(), 'data/test')

   if( os.path.isdir(data_dir) == False ):
      print('Saving Keras training and test data locally...')
      os.makedirs(data_dir, exist_ok=True)

      os.makedirs(train_dir, exist_ok=True)

      os.makedirs(val_dir, exist_ok=True)

      np.save(os.path.join(train_dir, 'x_train.npy'), x_train)
      np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
      np.save(os.path.join(val_dir, 'x_val.npy'), x_test)
      np.save(os.path.join(val_dir, 'y_val.npy'), y_test)

   # Call training natively
   from cifar10_keras_sage_train import main
   from types import SimpleNamespace

   inputs = {'train': f'{train_dir}',
               'val': f'{val_dir}',
               'model_dir': f'{data_dir}',
               'epochs': 1,
               'batch_size': 64
               }

   # argparse uses SimpleNamespace parameters, need to convert
   arg = SimpleNamespace(**inputs)

   main(arg)
   

def main(train_inst):
   print('Download CIFAR10 data, start Keras training on *{}*.'.format(train_inst))

   # Check to make sure local training data exists
   data_dir = os.path.join(os.getcwd(), 'data')
   train_dir = os.path.join(os.getcwd(), 'data/train')
   val_dir = os.path.join(os.getcwd(), 'data/test')
   if( os.path.isdir(data_dir) == False or os.path.isdir(train_dir) == False or os.path.isdir(val_dir) == False):
      print('Error: need to first save local training data. Run with "--train-inst test" first')
      return

   # For local runs, make sure Docker is available
   if (train_inst == 'local' or train_inst == 'local-gpu'):
      try:
         s = subprocess.check_output('docker --version', shell=True)
         print('Checking for Docker: ' + s.decode())
      except:
         print('Error: Docker is needed to run Sagemaker locally with "local" or "local-gpu" options.')
         return


   # Upload data to S3
   s3_prefix = 'sagemaker-datastore-name'
   print('Preparing training data on s3 at "{}"...'.format(s3_prefix))

   traindata_s3_prefix = '{}/data/train'.format(s3_prefix)
   valdata_s3_prefix = '{}/data/val'.format(s3_prefix)

   # Check to see if the data exists on s3
   inputs = {}
   s3 = boto3.resource('s3')
   bucket_default = sagemaker.Session().default_bucket()
   bucket = s3.Bucket('{}'.format(bucket_default))
   key = '{}/data/train/y_train.npy'.format(s3_prefix)   # Note: need exhaustive check?
   objs = list(bucket.objects.filter(Prefix=key))
   if len(objs) == 4 and objs[0].key == key:
      print('Found existing files in default bucket:')
      for t in objs:
         print(' {}'.format(t.key))
      train_saved = 's3://{}/{}/data/train'.format(bucket_default,s3_prefix)
      val_saved = 's3://{}/{}/data/val'.format(bucket_default,s3_prefix)
      inputs = {'train':train_saved,'val':val_saved}
   else:
      print('\nDid not find existing files in default bucket. Uploading data...')

      train_s3 = sagemaker.Session().upload_data(path='./data/train/', key_prefix=traindata_s3_prefix)
      val_s3 = sagemaker.Session().upload_data(path='./data/test/', key_prefix=valdata_s3_prefix)

      inputs = {'train':train_s3, 'val': val_s3}
      print('Done. Inputs: {}'.format(inputs))

   # Start Sagemaker training session

   model_dir = '/opt/ml/model'
   train_instance_type = train_inst
   hyperparameters = {'epochs': 1, 
                     'batch_size': 128
                  }

   local_estimator = TensorFlow(entry_point='cifar10_keras_sage_train.py',
                        source_dir='.',
                        model_dir=model_dir,
                        train_instance_type=train_instance_type,
                        train_instance_count=1,
                        hyperparameters=hyperparameters,
                        #   sagemaker_session=sagemaker_session,
                        # role=sagemaker.get_execution_role(),
                        role='arn:aws:iam::75xxxxxx',
                        base_job_name='tf-cifar-keras',
                        framework_version='1.12.0',
                        py_version='py3',
                        script_mode=True)

   print('\nBegining training eith parameters {}...'.format(hyperparameters))

   local_estimator.fit(inputs)

   # try:
   #    local_estimator.fit(inputs)
   # except OSError as err:
   #    print("OS error: {0}".format(err))
   # except ValueError:
   #    print("Could not convert data to an integer.")
   # except:
   #    print("Unexpected error:", sys.exc_info()[0])

   print('Done')

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument(
      '--train-inst',
      type=str,
      default='test',
      help='AWS Sagemaker training instance type. [test] local | local-gpu | ml.m5.4xlarge | ml.m4.16xlarge\n Default [test] funs training natively on local data.')

   args = parser.parse_args()
   if (args.train_inst == 'test') :
      test()
   else:
      main(args.train_inst)
