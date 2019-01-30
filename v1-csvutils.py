import csv
import tracemalloc
import time

tracemalloc.start()

maxrows = 1000000 # of 152 million

start_time = time.time();

numrows = 0

with open('airlines-10M.csv') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',', quotechar='\'')
    rows = []
    for row in filereader:
        numrows = numrows + 1
        if numrows % 100000 == 0 or numrows == maxrows:
            print('{:8} rows in {:8.6}s at {:8.6} rows/second'
                  .format(numrows, time.time() - start_time,
                          numrows / (time.time() - start_time)))
        if numrows == maxrows:
            break
        rows.append(row)

end_time = time.time()
        
stats = tracemalloc.take_snapshot().statistics('filename')

totalblocks = 0
totalbytes = 0

for st in stats:
    totalblocks += st.count
    totalbytes += st.size

print('{} blocks, {} Mbytes allocated'
      .format(totalblocks, totalbytes / 1000000.0))
print('{:8} lines per second'.format(numrows / (end_time - start_time)))
print('{} bytes per row'.format(totalbytes / numrows))
