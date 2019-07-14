# Sagemaker Keras Template
Template to run Keras examples (eg., CIFAR10) using AWS Sagemaker and the new `Script` API, with local testing methods. This approach refactors the Keras demo into a Sagemaker compatible `train.py` and `model.py` file, and uses local training data, downloaded using `keras.datasets.cifar10.load_data()` method and saved locally. Executing on either a hosted Docker container or on AWS Sagemaker requires the data to be uploaded to S3. This template provides a method for efficiently uploading the Keras data to an S3 bucket.

## Usage

```
python3 cifar10_keras_sage_main.py --help
```

Options are provided to run the Keras model on the local machine, without Sagemaker, to test the refactoring of the Keras examples. Running locally with Sagemaker in either the 'local' or 'local-gpu' mode is also supported, as well as running remote jobs on Sagemaker by specifying an instance type, eg. 'ml.m5.4xlarge'.

## Options


- `python3 cifar10_keras_sage_train.py --train-inst test`
  This will run the Keras CIFAR10 demo training locally on the host machine. Requires `python3`, `tensorflow` and `keras` to be installed locally.

- `python3 cifar10_keras_sage_train.py --train-inst local`
  This will run the Keras CIFAR10 demo training within a Docker container on the host machine. Requires `docker` and `sagemaker` to be installed.

- `python3 cifar10_keras_sage_train.py --train-inst ml.m5.4xlarge`
  This will run the Keras CIFAR10 demo training on AWS Sagemaker with a default of `1` training instance count. Options for the taining instance currently include: `[ml.p2.xlarge, ml.m5.4xlarge, ml.m4.16xlarge, ml.p3.16xlarge, ml.m5.large, ml.p2.16xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.c5.4xlarge, ml.c4.8xlarge, ml.c5.9xlarge, ml.c5.xlarge, ml.c4.xlarge, ml.c5.18xlarge, ml.p3.2xlarge, ml.m5.xlarge, ml.m4.10xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, ml.p2.8xlarge, ml.m5.2xlarge, ml.p3.8xlarge, ml.m4.4xlarge]`


## Prerequisites

Use of this Python script requires both specific host software and AWS account configurations to work correctly.

### AWS Configuration

- Sagemaker instance (please note the availability zone, as the s3 bucket needs to be in the same zone)
- IAM Role with S3 and Sagemaker privileges (this `arn` is needed for the script to run)
- S3 bucket (this bucket name is required, and it must be in the same availability region as the Sagemaker instance)

### Host Configuration

- Python3 and Pip3 (this script requires local host installation of `tensorflow` and `keras` and `sagemaker`)
- AWS CLI (recommend you install with `pip3 install awscli` and configure with `aws configuration` to set s3 defaults)
- Docker (optional, but required if you're using the `--train-inst local` modes)

## AWS Results

The AWS Sagemaker training output can be found in both the S3 buckets as well as the `CloudWatch` training logs. And all Sagemaker training runs are summarized in the `Training Jobs` dashboard.

## Errata

1. The test mode (`--train-inst test`) needs to be run at least once in order to download the Keras cifar10 data
2. This script currently only saves a Keras H5 model, but a `saved_model` structure is *required* for hosted Tensorflow hosting
3. Saving a model using the recommended methods leads to an error

```
Frustrating to see errors only at the end of a Sagemaker run! Looks like there is a problem saving models using the recommended `tf.contrib.save_model()` :

    model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True)
    TypeError: save_weights() got an unexpected keyword argument 'save_format'

Recommendations from Sagemaker SDK examples also caused errors at the end of run:

    # https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/keras_script_mode_pipe_mode_horovod/source_dir/cifar10_keras_main.py
     tf.saved_model.simple_save(
             tf.Session(),
             os.path.join(args.model_dir, 'model/1'),
             inputs={'inputs': model.input},
             outputs={t.name: t for t in model.outputs})
    
Depreciation warning: simple_save (from tensorflow.python.saved_model.simple_save) is deprecated and will be removed in a future version.
```

# TODO
1. Format with [Black](https://black.readthedocs.io/en/latest/) and test again
2. Fix `save_model` error
3. Add an example to show how to deploy the model to production
