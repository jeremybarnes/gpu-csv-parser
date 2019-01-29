 1872  virtualenv -p python3.5m virtualenv-3.5
 1873  . virtualenv-3.5/bin/activate
 1874  pip install pycuda
 1875  PATH=$PATH:/usr/local/cuda-10.0/bin pip install pycuda
 1876  pip install numpy
 1877  PATH=$PATH:/usr/local/cuda-10.0/bin pip install pycuda

