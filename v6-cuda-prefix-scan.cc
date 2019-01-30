__global__ void map_line_endings(const char * csvData,
                                 const unsigned csvDataOffset,
                                 int * lineNumbers,
                                 unsigned blockSize,
                                 unsigned * numLinesOut)
{
    csvData += csvDataOffset;

    __shared__ int blockNumLines;

    if (threadIdx.x == 0) {
        blockNumLines = 0;
    }
    __syncthreads();
    
    int threadNumLines = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < blockSize;
         i += blockDim.x * gridDim.x) {
        char c = csvData[i];
        int isEol = (c == '\n' ? 1 : 0);
        lineNumbers[i] = isEol;
        threadNumLines += isEol;
    }

    atomicAdd_block(&blockNumLines, threadNumLines);

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(numLinesOut, blockNumLines);
    }
}

__global__ void extract_line_starts(const int * lineNumbers,
                                    int * lineStarts,
                                    const unsigned numLines)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numLines;
         i += blockDim.x * gridDim.x) {
        if (i > 0 && lineNumbers[i] != lineNumbers[i - 1]) {
            lineStarts[lineNumbers[i - 1]] = i;
        }
    }
}

enum FieldType {
    FT_NULL,
    FT_INT,
    FT_DOUBLE,
    FT_RAW_STRING,
    FT_QUOTED_STRING,
    FT_ERROR,
    
    FT_SHORT_STRING_L0 = 8,
    FT_SHORT_STRING_L1,
    FT_SHORT_STRING_L2,
    FT_SHORT_STRING_L3,
    FT_SHORT_STRING_L4,
    FT_SHORT_STRING_L5,
    FT_SHORT_STRING_L6,
    FT_SHORT_STRING_L7,
};

struct FieldData {
    unsigned type;
    unsigned hash;
    union {
        long long i;
        double d;
        char shortStr[8];
        struct {
            unsigned stringOffset;
            unsigned stringLength;
        };
    };
};

__device__ FieldData
parse_field(const char * & lineData,
            const char * lineEnd,
            const char * allDataStart)
{
    const bool debug = false; //(threadIdx.x == 0 && blockIdx.x == 0);

    if (debug) {
        printf("parsing field with %d chars left at char %c offset %d\n",
               (int)(lineEnd - lineData), (int)lineData[0],
               (int)(lineData - allDataStart));
    }
    
    FieldData result;
    result.type = FT_NULL;
    result.hash = 0;
    result.i = 0;

    int strLen = 0;
    int numQuotes = 0;
    char shortStr[8];
    unsigned hash = 5381;
    long long sign = 1;
    unsigned long long digits = 0;
    bool intPossible = true;
    bool floatPossible = false;
    bool mustBeString = false;
    
    auto error = [&] (const char * msg) -> FieldData &
        {
            result.type = FT_ERROR;
            return result;
        };

    auto write = [&] (char c)
        {
            if (debug)
                printf("  write char %c\n", c);

            if (strLen < 8)
                shortStr[strLen] = c;
            ++strLen;

            // djb2 xor hash (avoids imul)
            hash = ((hash << 5) + hash) ^ c;

            if (intPossible) {
                if (strLen == 1) {
                    if (c == '+') {
                        sign = 1;
                    }
                    else if (c == '-') {
                        sign = -1;
                    }
                    else if (c >= '0' && c <= '9') {
                        digits = c - '0';
                    }
                    else {
                        intPossible = false;
                    }
                }
                else {
                    // TODO: deal with overflow
                    if (c >= '0' && c <= '9') {
                        digits = digits * 10 + c - '0';
                    }
                    else intPossible = false;
                }

                if (debug && intPossible) {
                    printf("int: sign %lld digits %lld possible %d\n",
                           sign, digits, intPossible);
                }
            }
            // TODO: deal with float possible
        };

    if (lineData >= lineEnd)
        return result;

    const char * stringDataStart = nullptr;
    
    char c = *lineData++;
    switch (c) {
    case '\n':
    case ',':
        break;
    case '\"': {
        mustBeString = true;
        intPossible = false;
        floatPossible = false;
        stringDataStart = lineData;
        
        // Quoted string; special loop (string type)
        while (lineData < lineEnd) {
            char c = *lineData++;
            if (c == '\"') {
                if (lineData == lineEnd) {
                    break;
                }
                c = *lineData++;
                if (c == '\"') {
                    ++numQuotes;
                    // double quote, so we output a single
                    write('\"');
                }
                else if (c == '\n' || c == ',') {
                    break;
                }
                else {
                    return error("invalid quoting");
                }
            }
            else {
                write(c);
            }
        }
        break;
    }
    default: {
        stringDataStart = lineData - 1;
        while (c != '\n' && c != ',') {
            write(c);
            if (lineData == lineEnd)
                break;
            c = *lineData++;
        }
        break;
    }
    }

    if (debug) {
        printf("strLen %d mustBeString %d intPossible %d hash %d\n",
               strLen, mustBeString, intPossible, hash);
    }
    
    if (strLen == 0 && !mustBeString) {
        result.type = FT_NULL;
        result.hash = 0;
        return result;
    }

    if (intPossible) {
        result.type = FT_INT;
        result.hash = hash;
        result.i = sign * digits;
        if (debug) {
            printf("returning int with value %d\n", result.i);
        }
        return result;
    }

    if (floatPossible) {
        // ... TODO: do...
    }

    // Otherwise it's a string
    result.hash = hash;
    if (strLen < 8) {
        result.type = FT_SHORT_STRING_L0 + strLen;
        for (int i = 0;  i <  strLen;  ++i) {
            result.shortStr[i] = shortStr[i];
        }
    }
    else if (numQuotes == 0) {
        result.type = FT_RAW_STRING;
        result.stringOffset = stringDataStart - allDataStart;
        result.stringLength = strLen;
    }
    else {
        result.type = FT_QUOTED_STRING;
        result.stringOffset = stringDataStart - allDataStart;
        result.stringLength = strLen;
    }

    return result;
}

__device__ void parse_line(const char * lineData,
                           int lineLength,
                           FieldData * fields,
                           int numFields,
                           const char * allDataStart,
                           unsigned fieldStride)
{
    const bool debug = false;//(threadIdx.x == 0 && blockIdx.x == 0);

    const char * e = lineData + lineLength;
    for (int i = 0;  i < numFields;  ++i) {
        fields[i * fieldStride] = parse_field(lineData, e, allDataStart);
        if (debug) {
            printf("field %d type %d hash %d i %ld\n",
                   i, fields[i].type, fields[i].hash, fields[i].i);
        }
    }
}

__global__ void parse_lines(const char * csvData,
                            const unsigned csvDataOffset,
                            FieldData * fields,
                            const int * lineStarts,
                            const int numLines,
                            const int numFields)
{
    csvData += csvDataOffset;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numLines;
         i += blockDim.x * gridDim.x) {
        unsigned startOfs = lineStarts[i] + 1;
        unsigned endOfs = lineStarts[i + 1];

        if (i == 0) {
            printf("i %d startOfs %d endOfs %d numLines %d numFields %d sz %d\n",
                   i, startOfs, endOfs, numLines, numFields, (int)sizeof(FieldData));
        }

        parse_line(csvData + startOfs, endOfs - startOfs,
                   fields + i /* fields + (i * numFields) */, numFields,
                   csvData, numLines /* fieldStride */);

        //if (i == 0) {
        //    for (int j = 0;  j < numFields;  ++j) {
        //        printf("field %d type %d hash %d i %ld\n",
        //               j, fields[j].type, fields[j].hash, fields[j].i);
        //    }
        //}
    }
}
