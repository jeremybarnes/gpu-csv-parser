import pycuda.autoinit
import pycuda.driver as drv
import pycuda.scan
import pycuda.gpuarray
import numpy

from pycuda.compiler import SourceModule

import tracemalloc
import time
import csv

tracemalloc.start()

maxrows = 1000000000 # of 152 million

maxbytes = 2000 * 1000 * 1000 # 2GB

# Read our CUDA kernel
with open('v6-cuda-prefix-scan.cc', 'r') as cudasourcefile:
    cudasource = cudasourcefile.read()
    
# Compile it for the GPU
mod = SourceModule(cudasource, options=['-std=c++14'])

# Get out our map function
map_line_endings = mod.get_function("map_line_endings")

# Generate our scan function that turns a list of EOL markers into line numbers
scan_line_numbers = pycuda.scan.InclusiveScanKernel('int32', "a+b")

# Get our line starts extraction function
extract_line_starts = mod.get_function("extract_line_starts")

# Get our line parsing function
parse_lines = mod.get_function("parse_lines")

# We read our data in chunks, as it's too big for some GPUs
blocksize = 500 * 1000 * 1000 # 100MB

headers = []

# Figure out number of fields from file headers
with open('airlines-10M.csv') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',', quotechar='\'')
    for row in filereader:
        headers = row
        break

num_fields = len(headers)
    
print('{} columns: {}'.format(num_fields, headers))
    
start_time = time.time();

# Read our CSV file in to an numpy array
csvfile = numpy.fromfile('airlines-10M.csv', dtype='int8', count=maxbytes)

# Transfer a block at a time to the GPU

numrows = 0
numbytes = 0

chunk_gpu = drv.mem_alloc(blocksize)
line_numbers_gpu = pycuda.gpuarray.empty([blocksize], 'int32')

# Python representation of field data
class FieldData:
    mem_size = 400


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
    
    before_kernel = time.time()

    numlines[0] = 0
    
    # run the line endings map
    map_line_endings(chunk_gpu,
                     numpy.uint32(0), # offset
                     line_numbers_gpu,
                     numpy.uint32(chunk.nbytes),
                     drv.InOut(numlines),
                     block=(512,1,1),
                     grid=(1024,1))

    after_map = time.time()
    
    # run the line numbers scan
    scan_line_numbers(line_numbers_gpu)

    after_scan = time.time()

    chunk_num_lines = numlines[0]
    
    # get line starts, one extra at the end for the last character
    line_starts_gpu = pycuda.gpuarray.empty([chunk_num_lines + 1], 'int32')

    extract_line_starts(line_numbers_gpu,
                        line_starts_gpu,
                        numpy.uint32(chunk_num_lines),
                        block=(512,1,1),
                        grid=(int((chunk_num_lines + 511) / 512),1))

    after_extract = time.time()

    fields_gpu = pycuda.gpuarray.empty([chunk_num_lines,num_fields,2], 'int64')

    print('allocated {}Mb for {} fields'.format(fields_gpu.nbytes / 1000000.0,
                                                chunk_num_lines * num_fields))
    
    #print(line_starts_gpu[0:100])
    
    # parse lines in parallel
    parse_lines(chunk_gpu,
                numpy.uint32(0), # offset
                fields_gpu,
                line_starts_gpu,
                numpy.uint32(chunk_num_lines),
                numpy.uint32(num_fields),
                block=(512,1,1),
                grid=(int((chunk_num_lines + 511) / 512), 1))
    
    
    after_kernel = time.time()
    elapsed_kernel = after_kernel - before_kernel

    print('numlines = {} in {:8.4}s transfer, {:6.4}s compute ({:.3}s+{:6.4}s+{:6.4}s+{:6.4}s)'
          .format(numlines[0], elapsed_transfer, elapsed_kernel,
                  after_map-before_kernel, after_scan-after_map,
                  after_extract-after_scan, after_kernel-after_extract))

    numrows += numlines[0]

end_time = time.time()
        
stats = tracemalloc.take_snapshot().statistics('filename')

totalblocks = 0
totalbytes = 0

for st in stats:
    totalblocks += st.count
    totalbytes += st.size

print('{} blocks, {} Mbytes allocated'.format(totalblocks, totalbytes / 1000000.0))
print('{:8.6} seconds elapsed'.format(end_time - start_time))
print('{:8} lines per second'.format(numrows / (end_time - start_time)))
