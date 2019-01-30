__global__ void
count_lines(const char * csvdata,
            const unsigned blockSize,
            int * numLinesOut)
{
    // This variable is shared across all threads in the block
    __shared__ int blockNumLines;

    // If we're the first thread in the block, we initialize to zero
    if (threadIdx.x == 0) {
        blockNumLines = 0;
    }

    // Wait for everyone to get here so we all see an initialized variable
    __syncthreads();

    // Local (register) count of the number of lines in the block
    int threadNumLines = 0;

    // Divide memory up per thread and per block, and start at a unique place
    // per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < blockSize;
         i += blockDim.x * gridDim.x) {

        // We read our character.  This is done massively in parallel
        char c = csvdata[i];

        // Count our newlines
        if (c == '\n') {
            ++threadNumLines;
        }
    }

    // Add our thread count to our block count.  This needs to be atomic since
    // all threads will arrive here at the same time with their count.
    atomicAdd_block(&blockNumLines, threadNumLines);

    // Ensure that all threads have added their count to our block count
    __syncthreads();

    // The first thread in the block adds atomically to the global output
    // variable
    if (threadIdx.x == 0) {
        atomicAdd(numLinesOut, blockNumLines);
    }
}
