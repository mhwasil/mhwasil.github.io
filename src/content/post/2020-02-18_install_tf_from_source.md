---
title: "Install TensorfFlow from Source on HPC Cluster Node"
description: "Tutorial for installing TensorFlow from source on HPC cluster node"
publishDate: "18 January 2020"
tags: ["tutorial"]
draft: false
---

As I run into trouble by not being able to build my own tensor ops (tensor operators) which I needed to
train [PointNet](https://github.com/charlesq34/pointnet), [PointNet2](https://github.com/charlesq34/pointnet2) and [SpiderCNN](https://github.com/xyf513/SpiderCNN) for 3D object classification. 
Some solution, like [this](https://github.com/tensorflow/tensorflow/issues/9137#issuecomment-294097780), 
[this](https://github.com/google/sentencepiece/issues/293#issuecomment-497573645),
and [this](https://github.com/google/sentencepiece/issues/293#issuecomment-510806920),
did not work for my setup.

One of the reasons is due to the fact that tensorflow was compiled with [different gcc version](https://github.com/deepsense-ai/roi-pooling/issues/1).
Of course, I could have just used the same version as suggested in the original repository,
for example use tensorflow 1.4 and cuda 8.0, but sometimes it does not work that way on our cluster 
since the sysadmin may update the version of nvidia driver and gcc version and then remove the older version. 
Thus, tensorflow needs to be compiled locally in order to match the gcc version.

But, finally I got it working by following the steps
and match the cuda, nccl and cudnn described on [this page](https://www.tensorflow.org/install/source). The following is the cluster environment, setup and installation process.

* Environment of the cluster (compute node)
  * os distribution 
    ```
    $cat /etc/*-release
    NAME="Scientific Linux"
    PRETTY_NAME="Scientific Linux 7.7 (Nitrogen)"
    REDHAT_BUGZILLA_PRODUCT="Scientific Linux 7"
    REDHAT_BUGZILLA_PRODUCT_VERSION=7.7
    REDHAT_SUPPORT_PRODUCT="Scientific Linux"
    REDHAT_SUPPORT_PRODUCT_VERSION="7.7"
    Scientific Linux release 7.7 (Nitrogen)
    ```
  * jdk version 10.0.1
    ```
    java version
    ```
  * cuda version 9.2

* Requirements
  * cudnn 7.5.6
    * [Download cudnn](https://developer.nvidia.com/cudnn), login required
    * Unzip and move them to $HOME/local/share
  * nccl 2.5.6
    * [Download nccl](https://developer.nvidia.com/nccl), login required
    * Unzip and move them to $HOME/local/share
  * bazel 0.24.1 as tested [on this page](https://www.tensorflow.org/install/source)
    * Install with argument --user to install it locally in $HOME/bin
      ```
      chmod +x bazel-<version>-installer-linux-x86_64.sh 
      ./bazel-0.24.1-installer-linux-x86_64.sh --user
      ```
    * export the path
      ```
      export PATH="$PATH:$HOME/bin"
      ```
  * TensorFlow r1.14
    * clone the repository and checkout
      ```
      git clone https://github.com/tensorflow/tensorflow.git
      cd tensorflow && git checkout r1.14
      ```
* Build TensorFlow 1.14 on compute node
  * Load cuda library
    ```
    module load cuda/9.2
    ```
    The availability of this command depends on your system admin. But if this is not available, you can just export the path
    ```
    export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH
    ```
  * Add your cudnn and nccl library into LD_LIBRARY_PATH as well
    ```
    export LD_LIBRARY_PATH=/home/emha/local/share/cudnn/lib64:/home/emha/local/share/nccl_2.5.6/lib:$LD_LIBRARY_PATH
    ```
    replace "emha" with your home directory name
  * Activate your conda environment if you use conda
  * Go to tensorflow directory and configure
    ```
    ./configure
    ```
    The following is my configuration
    ```
    You have bazel 0.24.1 installed.
    Please specify the location of python. [Default is /home/emha/anaconda3/envs/tf-source/bin/python]: 

    Found possible Python library paths:
    /home/emha/anaconda3/envs/tf-source/lib/python3.6/site-packages
    Please input the desired Python library path to use.  Default is [home/emha/anaconda3/envs/tf-source/lib/python3.6/site-packages]

    Do you wish to build TensorFlow with XLA JIT support? [y/N]: Y
    No XLA JIT support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
    No OpenCL SYCL support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with ROCm support? [y/N]: N

    Do you wish to build TensorFlow with CUDA support? [y/N]: y
    CUDA support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with TensorRT support? [y/N]: N
    No TensorRT support will be enabled for TensorFlow.

    Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.2]: 9.2

    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.0

    Please specify the locally installed NCCL version you want to use.  [Leave empty to use http://github.com/nvidia/nccl]: 2.5.6

    Please specify the comma-separated list of base paths to look for CUDA libraries and headers. [Leave empty to use the default]: /usr/local/cuda-9.2,/home/emha/local/cudnn,/home/emha/local/nccl_2.5.6
    ```
  * Build with bazel
    ```
    bazel build --configopt --config=cuda --copt=--cuda_log //tensorflow/tools/pip_package:build_pip_package
    ```
    If you encounter [a problem](https://github.com/tensorflow/tensorflow/issues/26249#issuecomment-468633974) with *http_archive*, add the following to WORKSPACE file before the *http_archive* function
    ```
     load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
    ```
  * If everything is fine, now you start building tensorflow
    ```
    /bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkgs
    ```
    You may want to replace tmp directory, and use your home directory for use later
  * Install the built tensorflow
    ```
    pip install /tmp/tensorflow-1.14.1-cp36-linux_x86_64.whl
    ```
    *You should be in the same env as the one use for building tf*
  * After installation, if you encounter error like [*cannot find -ltensorflow_framework*](https://github.com/bgshih/aster/issues/56#issuecomment-501973315), it's because the name of the tensorflow_framework.so is changed to tensorflow_framework.so.1, the solution is create a symlink to that file
    ```
    ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
    ```
* **Note**
  * You need to export cudnn and nccl you used to build tf inorder to load tensorflow
    ```
    export LD_LIBRARY_PATH=/home/emha/local/share/nccl_2.5.6/lib
    export LD_LIBRARY_PATH=/home/emha/local/share/cudnn/lib64
    ```

**Compile your own tensor operators**

Example bash script how to compile [tensor operators](https://github.com/charlesq34/pointnet2/tree/master/tf_ops/grouping). Create bash script like the following and compile it.
```
#/bin/bash
/usr/local/cuda-9.2/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler 
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so_hk.so -shared -fPIC -I /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.2/include -I /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.2/lib64 -L /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

```
