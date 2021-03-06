import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

import tracemalloc
import time

tracemalloc.start()

start_time = time.time();
maxrows = 1000000 # of 152 million

maxbytes = 1000 * 1000 * 1000 # 1GB

# Read our CUDA kernel
with open('v5-cuda-parallel.cc', 'r') as cudasourcefile:
    cudasource = cudasourcefile.read()
    
# Compile it for the GPU
mod = SourceModule(cudasource)

# Get out our function
count_lines = mod.get_function("count_lines")

# We read our data in chunks, as it's too big for some GPUs
blocksize = 100 * 1000 * 1000 # 100MB

# Read our CSV file in to an numpy array
csvfile = numpy.fromfile('airlines-10M.csv', dtype='int8', count=maxbytes)

# Transfer a block at a time to the GPU

numrows = 0
numbytes = 0

chunk_gpu = drv.mem_alloc(blocksize)

numlines = numpy.zeros(1, dtype='int32')

while numbytes < len(csvfile) and numrows < maxrows:
    chunk = csvfile[numbytes:numbytes + blocksize]

    print('chunk of {} bytes from {} to {}'
          .format(chunk.nbytes, numbytes, numbytes + blocksize))

    numbytes += chunk.nbytes

    before_transfer = time.time()
    
    # transfer to the GPU
    drv.memcpy_htod(chunk_gpu, chunk)

    after_transfer = time.time()

    elapsed_transfer = after_transfer - before_transfer
    
    nbytesarray = numpy.array([chunk.nbytes], dtype='int32')    

    before_kernel = time.time()
    
    # run the line counter
    count_lines(chunk_gpu,
                numpy.uint32(nbytesarray),
                drv.InOut(numlines),
                block=(512,1,1),
                grid=(1024,1))

    after_kernel = time.time()
    elapsed_kernel = after_kernel - before_kernel

    print('numlines = {} in {:8.4}s transfer, {:8.4}s compute'
          .format(numlines[0], elapsed_transfer, elapsed_kernel))

end_time = time.time()
        
stats = tracemalloc.take_snapshot().statistics('filename')

totalblocks = 0
totalbytes = 0

numrows = numlines[0]

for st in stats:
    totalblocks += st.count
    totalbytes += st.size

print('{} blocks, {} Mbytes allocated'.format(totalblocks, totalbytes / 1000000.0))
print('{:8.6} seconds elapsed'.format(end_time - start_time))
print('{:8} lines per second'.format(numlines[0] / (end_time - start_time)))
