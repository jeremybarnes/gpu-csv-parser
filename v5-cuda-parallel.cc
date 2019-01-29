enum {
    NO_ACTION = 0,
    EMIT_FIELD = 1,
    EMIT_EOL = 128
};

typedef int Action;

enum State {
    ST_FIELD,
    ST_INVALID
};

__device__ Action nextState(State & state, char c)
{
    Action action;
    
    switch (c) {
    case ',':
        action = EMIT_FIELD;
        break;
    case '\n':
        action = EMIT_FIELD | EMIT_EOL;
        break;
    default:
        action = NO_ACTION;
        break;
    }

    return action;
}


__global__ void parse_csv(const char * csvdata,
                          const int * blockSizePtr,
                          const int * initialState,
                          int * states,
                          int * numLinesOut)
{
    __shared__ int blockNumLines;

    if (threadIdx.x == 0) {
        blockNumLines = 0;
    }
    __syncthreads();
    
    const int blockSize = *blockSizePtr;

    int threadNumLines = 0;

    //printf("bidx %d bdim %d tidx %d gdim %d start %d inc %d\n",
    //       blockIdx.x, blockDim.x, threadIdx.x, gridDim.x,
    //       blockIdx.x * blockDim.x + threadIdx.x,
    //       blockDim.x * gridDim.x);
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < blockSize;
         i += blockDim.x * gridDim.x) {
        char c = csvdata[i];
        if (c == '\n') {
            ++threadNumLines;
        }
    }

    atomicAdd_block(&blockNumLines, threadNumLines);

    __syncthreads();

    if (threadIdx.x == 0) {
        //printf("blockNumLines = %d\n", blockNumLines);
        atomicAdd(numLinesOut, blockNumLines);
    }
}
