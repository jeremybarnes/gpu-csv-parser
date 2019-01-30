__global__ void count_lines(const char * csvdata,
                            const unsigned blockSize,
                            int * numLinesOut)
{
    int numLinesLocal = 0;

    for (int i = 0;  i < blockSize;  ++i) {

        char c = csvdata[i];
        if (c == '\n') {
            ++numLinesLocal;
        }
    }
    
    *numLinesOut += numLinesLocal;
}
