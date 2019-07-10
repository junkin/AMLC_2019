#!/bin/bash


pushd /data/mobilenet_trt/
wget https://s3-us-west-2.amazonaws.com/com.nvidia.tensorrt-laboratory/open_source_images.tar.gz
tar xf open_source_images.tar.gz

wget https://junkin-amlc-2019.s3-us-west-2.amazonaws.com/mobilenet_v1_1.0_224.uff
popd
