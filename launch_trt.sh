 sudo nvidia-docker run -it --privileged  -p 8080:8080 -v `pwd`:/data nvcr.io/nvidia/tensorrt:19.05-py3
