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
    const int i = threadIdx.x;

    printf("i = %d\n", i);
    
    if (i != 0)
        return;

    int blockSize = *blockSizePtr;
    
    int numLinesLocal;

    numLinesLocal = 0;

    printf("blockSize = %d\n", blockSize);

    State state = ST_FIELD;
    
    for (int i = 0;  i < blockSize;  ++i) {

        char c = csvdata[i];
        Action action = nextState(state, c);

        if (action & EMIT_EOL)
            ++numLinesLocal;
    }

    printf("numLinesLocal = %d\n", numLinesLocal);
    
    *numLinesOut += numLinesLocal;
}
