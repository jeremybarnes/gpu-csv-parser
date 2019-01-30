extern "C" {

typedef unsigned uint32_t;
typedef unsigned long uint64_t;

struct Lz4Block {
    uint32_t offset;
    uint32_t length;
    uint32_t outOffset;
    bool compressed;
};

__global__ void lz4_find_blocks(const unsigned char * inputData,
                                uint32_t inputDataLength,
                                Lz4Block * blocks,
                                uint32_t maxBlocks,
                                uint32_t maxBytes,
                                unsigned * numBlocksOut,
                                unsigned * totalBytesOut)
{
    //printf("inputDataLength %d\n", inputDataLength);

    const unsigned char * inputDataStart = inputData;
    const unsigned char * inputDataEnd = inputData + inputDataLength;

    auto read_u32 = [&] () -> unsigned
        {
            //printf("got chars %02x %02x %02x %02x\n",
            //       (int)inputData[0], (int)inputData[1],
            //       (int)inputData[2], (int)inputData[3]);
            unsigned result
                = inputData[0]
                + (inputData[1] << 8)
                + (inputData[2] << 16)
                + (inputData[3] << 24);
            inputData += 4;
            return result;
        };

    auto read_u64 = [&] () -> unsigned long
        {
            uint64_t result = read_u32();
            result = result | ((uint64_t)read_u32() << 32); 
            return result;
        };

    auto error = [&] (const char * msg)
        {
            printf(msg);
            printf("\n");
            *numBlocksOut = -1;
        };
    
    uint32_t checksum = read_u32();

    if (checksum != 0x184d2204) {
        printf("checksum is %x\n", checksum);
        return error("invalid lz4 checksum");
    }

    unsigned char flags = *inputData++;
    unsigned char bd = *inputData++;

    int version = flags >> 6;
    bool independent = flags & (1 << 5);
    bool bchecksum = flags & (1 << 4);
    bool csize = flags & (1 << 3);
    //bool cchecksum = flags & (1 << 2);
    bool dict = flags & (1 << 0);

    if (version != 1) {
        return error("invalid version");
    }

    if (!independent) {
        return error("only independent blocks supported");
    }

    long long contentSize __attribute__((__unused__)) = -1;
    if (csize) {
        contentSize = read_u64();
        printf("contentSize %lld\n", contentSize);
    }

    if (dict) {
        return error("can't use dictionary");
    }

    unsigned blockSize = 256 << 2 * ((bd >> 4) & 7);

    printf("blockSize %d\n", blockSize);
    
    unsigned char headerChecksum = *inputData++;
    uint32_t bytesOut = 0;
    int blockNum = 0;

    // Now we have decoded the header, create blocks one by one
    for (; blockNum < maxBlocks
             && bytesOut < maxBytes
             && inputData + 4 < inputDataEnd;
         ++blockNum) {
        unsigned inputBlockSize = read_u32();

        if (inputBlockSize == 0)
            break;

        unsigned blockOffset = inputData - inputDataStart;
        bool isUncompressed = inputBlockSize >> 31;
        inputBlockSize &= 0x7fffffffU;

        //printf("block %d inputBlockSize %d uncompressed %d out %d at %d\n",
        //       blockNum, inputBlockSize,
        //       isUncompressed,
        //       int(bytesOut),
        //       int(inputData - inputDataStart));

        inputData += inputBlockSize;

        if (inputData >= inputDataEnd)
            break;
        
        if (bchecksum) {
            printf("block checksum\n");
            
            //unsigned checksum __attribute__((__unused__)) = read_u32();
            inputData += 4;
        }

        
        blocks[blockNum] = { blockOffset, inputBlockSize, bytesOut,
                             isUncompressed };
        bytesOut += blockSize;
    }

    *numBlocksOut = blockNum;
    if (contentSize == -1) {
        contentSize = bytesOut;
    }
    *totalBytesOut = contentSize;
}

__device__ void
lz4_decompress_block(const unsigned char * inputData,
                     uint32_t inputDataLength,
                     unsigned char * outputData,
                     uint32_t outputDataLength)
{
    // expects to have a single thread block to do all of the work
    unsigned pos = 0;
    unsigned opos = 0;

    // Decode and return a linear small integer code value
    auto lsic = [&] () -> unsigned
        {
            unsigned result = 0;
            unsigned char c;
            
            do {
                c = inputData[pos++];
                //if (threadIdx.x == 0)
                //    printf("lsic read byte %d\n", (int)c);
                result += c;
            } while (c == 255);
            
            return result;
        };
    
    while (pos < inputDataLength && opos < outputDataLength) {

        //if (blockIdx.x == 474 && threadIdx.x == 0) {
        //    printf("pos %d inputDataLength %d opos %d outputDataLength %d\n",
        //           pos, inputDataLength, opos, outputDataLength);
        //}
        
        // Decode an operation
        unsigned char token = inputData[pos++];
        unsigned t1 = token >> 4;
        unsigned t2 = token & 15;

        if (t1 == 15) {
            t1 += lsic();
        }

        unsigned tocopy = min(t1, outputDataLength - opos);

        // copy t1 bytes from input to output
        //if (threadIdx.x == 0)
        //    printf("copying %d bytes from input\n", t1);

        for (int j = threadIdx.x;  j < tocopy;  j += blockDim.x) {
            outputData[opos + j] = inputData[pos + j];
            //printf("    out[%d] = %c\n", opos + j, outputData[opos + j]);
        }

        pos += t1;
        opos += tocopy;

        if (tocopy < t1)
            break;

        // Offset of data to copy in already decoded stream
        unsigned o = inputData[pos++];
        o += inputData[pos++] << 8;

        // Extra bits for number of bytes
        if (t2 == 15) {
            t2 += lsic();
        }
        t2 += 4;

        tocopy = min(t2, outputDataLength - opos);
        
        //if (threadIdx.x == 0)
        //    printf("pos = %d opos = %d inputDataLength = %d t1 = %d t2 = %d o = %d\n",
        //           pos, opos, inputDataLength, t1, t2, o);

        unsigned done = 0;
        
        // The maximum we can do per iteration is o, as further bytes
        // are a de-duplication
        while (done < tocopy) {
            unsigned todo = min(o, tocopy - done);
            //if (threadIdx.x == 0)
            //    printf("copying %d bytes from offset -%d\n",
            //           todo, o);
            for (int j = threadIdx.x;  j < todo; j += blockDim.x) {
                outputData[opos + j] = outputData[opos - o + j];
                //printf("    out[%d] = %c\n", opos + j, outputData[opos + j]);
            }
            done += todo;
            opos += todo;
            
            if (done < tocopy) {
                // Ensure all data from this group is copied before the
                // next group starts
                __syncthreads();
            }
        }

        if (tocopy < t2)
            break;
    }
}

__global__ void
lz4_decompress_blocks(const unsigned char * inputData,
                      unsigned char * outputData,
                      uint32_t inputDataLength,
                      const Lz4Block * blocks,
                      uint32_t outputDataLength)
{
    int blockNum = blockIdx.x;
    int numBlocks = gridDim.x;

    const Lz4Block & block = blocks[blockNum];

    lz4_decompress_block(inputData + block.offset,
                         block.length,
                         outputData + block.outOffset,
                         blockNum == numBlocks - 1
                         ? (outputDataLength - block.outOffset)
                         : (blocks[blockNum + 1].outOffset - block.outOffset));
}


__global__ void
map_line_endings(const char * csvData,
                 const unsigned csvDataOffset,
                 int * lineNumbers,
                 unsigned blockSize,
                 unsigned * numLinesOut)
{
    //if (threadIdx.x == 0)
    //    printf("map_line_endings %d\n", blockIdx.x);
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
    FT_ERROR,
    
    FT_RAW_STRING,
    FT_QUOTED_STRING,

    FT_SHORT_STRING_L0 = 8,
    FT_SHORT_STRING_L1,
    FT_SHORT_STRING_L2,
    FT_SHORT_STRING_L3,
    FT_SHORT_STRING_L4,
    FT_SHORT_STRING_L5,
    FT_SHORT_STRING_L6,
    FT_SHORT_STRING_L7,
};

__device__ bool isString(FieldType t)
{
    return t >= FT_RAW_STRING;
}
    
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

struct HashBucket {
    union {
        struct {
            uint32_t key;
            uint32_t fieldIndex;
        };
        uint64_t bits;
    };
};

__device__ void analyze_column(const char * csvData,
                               FieldData * fields,
                               const int numLines)
{
    constexpr int MAX_SIZE = 2048;

    __shared__ unsigned occupancy;
    __shared__ HashBucket buckets[MAX_SIZE];
    
    if (threadIdx.x == 0)
        occupancy = 0;

    for (int i = threadIdx.x;  i < numLines && numStringHashes < 4000;
         i += blockDim.x) {
        buckets[i].key = buckets[i].fieldIndex = 0;
    }

    __syncthreads();

    for (int i = threadIdx.x;  i < numLines && numStringHashes < 4000;
         i += blockDim.x) {
        if (isString((FieldType)fields[i].type)) {
            

            unsigned hashNum = atomicAdd(&numStringHashes, 1);
            //printf("got hash num %d for hash %d col %d\n", hashNum, fields[i].hash,
            //       blockIdx.x);
            if (hashNum >= 4000)
                break;
            stringHashes[hashNum] = fields[i].hash;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
        printf("got %d hashes for column %d\n", numStringHashes, blockIdx.x);
}

__global__ void analyze_columns(const char * csvData,
                                const unsigned csvDataOffset,
                                FieldData * fields,
                                const int numLines,
                                const int numFields)
{
    int field = blockIdx.x;
    analyze_column(csvData + csvDataOffset,
                   fields + (numLines * field),
                   numLines);
}

} // extern "C"
