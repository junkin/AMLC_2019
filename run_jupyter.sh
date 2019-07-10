echo `./public_ip.sh`:$1
jupyter-notebook --ip=0.0.0.0 --port=$1 --allow-root
