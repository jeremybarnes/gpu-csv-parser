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

maxrows = 20000000 # of 152 million

maxbytes = 2000 * 1000 * 1000 # 2GB

# Read our CUDA kernel
with open('v9-column-dedup.cc', 'r') as cudasourcefile:
    cudasource = cudasourcefile.read()
    
# Compile it for the GPU
mod = SourceModule(cudasource, options=['-std=c++14','-lineinfo'],
                   no_extern_c=True,
                   keep=True # for cuda-gdb to work
                   )

# Get our scan lz4 function
lz4_find_blocks = mod.get_function("lz4_find_blocks")

# Get our decompress lz4 function
lz4_decompress_blocks = mod.get_function("lz4_decompress_blocks")

# Get out our map function
map_line_endings = mod.get_function("map_line_endings")

# Generate our scan function that turns a list of EOL markers into line numbers
scan_line_numbers = pycuda.scan.InclusiveScanKernel('int32', "a+b")

# Get our line starts extraction function
extract_line_starts = mod.get_function("extract_line_starts")

# Get our line parsing function
parse_lines = mod.get_function("parse_lines")

# Get our column analysis function
analyze_columns = mod.get_function("analyze_columns")

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

# Read our compressed CSV file in to an numpy array
compressed_csvfile = numpy.memmap('airlines-10M-sml2.csv.lz4', dtype='uint8', mode='r')

# Transfer compressed data to the GPU to save on PCI Express bandwidth
compressed_csvfile_gpu = drv.mem_alloc(compressed_csvfile.nbytes)

before_transfer = time.time()

drv.memcpy_htod(compressed_csvfile_gpu, compressed_csvfile)

after_transfer = time.time()

elapsed_transfer = after_transfer - before_transfer;

print('{:8.4}s to transfer {:8.4}MB uncompressed at {:8.4}MB/sec'
      .format(elapsed_transfer, compressed_csvfile.nbytes / 1000000,
              compressed_csvfile.nbytes / 1000000 / elapsed_transfer))

max_blocks = 8192

blocks_gpu = pycuda.gpuarray.empty([max_blocks,4], 'uint32')

num_blocks = numpy.zeros(1, dtype='uint32')
total_size = numpy.zeros(1, dtype='uint32')

# find where the lz4 blocks are in the file

lz4_find_blocks(compressed_csvfile_gpu,
                numpy.uint32(compressed_csvfile.nbytes),
                blocks_gpu,
                numpy.uint32(max_blocks),
                numpy.uint32(maxbytes),
                drv.InOut(num_blocks),
                drv.InOut(total_size),
                block=(1,1,1),
                grid=(1,1))

num_decompressed_bytes = int(total_size[0])

print('{} blocks out with {} bytes'.format(num_blocks[0], num_decompressed_bytes))

# and decompress them

chunk_gpu = drv.mem_alloc(num_decompressed_bytes)

lz4_decompress_blocks(compressed_csvfile_gpu,
                      chunk_gpu,
                      numpy.uint32(compressed_csvfile.nbytes),
                      blocks_gpu,
                      numpy.uint32(num_decompressed_bytes),
                      block=(32,1,1),
                      grid=(int(num_blocks[0]), 1))

after_decompress = time.time()

elapsed_decompress = after_decompress - after_transfer

print('decompressed {:8.4}MB in {:8.4}s at {:8.4}MB/s'
      .format(num_decompressed_bytes / 1000000, elapsed_decompress,
              num_decompressed_bytes / 1000000 / elapsed_decompress))


#chunk_debug= numpy.empty(1000, 'c')
#drv.memcpy_dtoh(chunk_debug, chunk_gpu)

#print(chunk_debug.view('S100'))

# Transfer our compressed CSV file to the GPU

line_numbers_gpu = pycuda.gpuarray.empty([num_decompressed_bytes], 'int32')

numlines = numpy.zeros(1, dtype='int32')

numlines[0] = 0

before_kernel = time.time()

# run the line endings map
map_line_endings(chunk_gpu,
                 numpy.uint32(0), # offset
                 line_numbers_gpu,
                 numpy.uint32(num_decompressed_bytes),
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

# analyze columns in parallel
analyze_columns(chunk_gpu,
                numpy.uint32(0), # offset
                fields_gpu,
                numpy.uint32(chunk_num_lines),
                numpy.uint32(num_fields),
                block=(256,1,1),
                grid=(num_fields,1))
                
                

after_kernel = time.time()
elapsed_kernel = after_kernel - before_kernel

print('numlines = {} in {:8.4}s transfer, {:6.4}s compute ({:.3}s+{:6.4}s+{:6.4}s+{:6.4}s)'
      .format(numlines[0], elapsed_transfer, elapsed_kernel,
              after_map-before_kernel, after_scan-after_map,
              after_extract-after_scan, after_kernel-after_extract))

end_time = time.time()
        
stats = tracemalloc.take_snapshot().statistics('filename')

totalblocks = 0
totalbytes = 0

for st in stats:
    totalblocks += st.count
    totalbytes += st.size

print('{} python memory blocks, {} Mbytes allocated'.format(totalblocks, totalbytes / 1000000.0))
print('{:8.6} seconds elapsed'.format(end_time - start_time))
print('{:8} lines per second'.format(chunk_num_lines / (end_time - start_time)))
