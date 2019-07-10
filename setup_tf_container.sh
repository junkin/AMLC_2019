pushd /workspace/nvidia-examples/tensorrt
python setup.py install
popd

pushd /workspace/nvidia-examples/tensorrt/tftrt/examples/image-classification
bash ./install_dependencies.sh
cd ../third_party/models
export PYTHONPATH="$PYTHONPATH:$PWD"

popd


