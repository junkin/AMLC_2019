AMLC 2019 workshop



### setup VM
git clone https://github.com/junkin/AMLC_2019.git
cd AMLC_2019


### launch the container - TRT first

~/AMLC_2019$ ./launch_trt.sh

root@d60afd88d87a:/workspace# cd /data

root@d60afd88d87a:/data# bash setup_trt_container.sh

This container fwds ports 8081 for the jupyter notebooks.

### launch the TF container
~/AMLC_2019$ ./launch_tensorflow_trt.sh

root@d60afd88d87a:/workspace# cd /data

root@d60afd88d87a:/data# source setup_tf_container.sh

root@d60afd88d87a:/data# ./run_jupyter.sh 8080
