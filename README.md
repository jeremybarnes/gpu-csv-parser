
## Executing

 ```sh
virtualenv -p python3.5m virtualenv-3.5
. virtualenv-3.5/bin/activate
pip install pycuda
PATH=$PATH:/usr/local/cuda-10.0/bin pip install pycuda
pip install numpy
PATH=$PATH:/usr/local/cuda-10.0/bin pip install pycuda
```
