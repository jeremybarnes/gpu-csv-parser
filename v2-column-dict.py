import csv
import tracemalloc
import time

tracemalloc.start()

maxrows = 10000000 # of 152 million

start_time = time.time();

with open('airlines-10M.csv') as csvfile:
    filereader = csv.reader(csvfile, delimiter=',', quotechar='\'')
    numrows = 0
    rows = []
    for row in filereader:
        numrows = numrows + 1

        # Header row, use it to figure out how many dictionaries we want
        if numrows == 1:
            print('{} columns: {}'.format(len(row),row))
            dicts = [{} for col in row]
            continue

        if numrows % 100000 == 0 or numrows == maxrows:
            print('{:8} rows in {:8.6}s at {:8.6} rows/second'
                  .format(numrows, time.time() - start_time,
                          numrows / (time.time() - start_time)))
        if numrows == maxrows:
            break

        # Look up values in dictionary to save memory on duplicates
        for i in range(1, len(row)):
            val = dicts[i].get(row[i])
            if val is None:
                dicts[i][row[i]] = row[i]
                val = row[i]
            row[i] = val
            
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
