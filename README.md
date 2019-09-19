2019 workshop


### configure EC2 instance
Navigate to the EC2 dashboard and perform the following:
1. select launch instance.
2. for your AMI select "My ami's" and check the box "shared with me"
3. select the NVDA_WORKSHOP_2019
35. choose a p3.2xlarge instance type
4. edit storage - expand to 100GB
5. add tags - for key use name, for value alias.  use(IE Scot, sjunkin@)
6. select create new key - use your alias for key name (IE sjunkin@) save keypair someplace.
7. launch instance.

## connect
Wait for a few minutes for the instance to spin up.
ssh -L 8080:127.0.0.1:8080 -i ubuntu@IP_ADDRESS_HERE
example:

    ssh -i /mnt/c/Users/sjunkin/Downloads/win10-oregon.pem -L 8080:127.0.0.1:8080 ubuntu@54.149.221.127


### setup VM
cd AMLC_2019

### launch amp tutorial
~/AMLC_2019$ ./launch_mxnet.sh

### launch container

~/AMLC_2019$ ./launch_trt.sh

root@d60afd88d87a:/workspace# cd /data

root@d60afd88d87a:/data# source setup_trt_container.sh

This container fwds ports 8080 for the jupyter notebooks.

### launch the TF container
~/AMLC_2019$ ./launch_tensorflow_trt.sh

root@d60afd88d87a:/workspace# cd /data

root@d60afd88d87a:/data# source setup_tf_container.sh

root@d60afd88d87a:/data# ./run_jupyter.sh 8080
