import csv
with open('Data/DNA.tsv') as file:
    out = csv.reader(file, delimiter='\t')
    length = 0
    raw = []
    for i in out:
        length += 1
        raw.append(i)
        
    for i in raw:
        for x in range(8): i.pop(0)
        i.pop(8)
        i.pop(9)
    
    print(raw)