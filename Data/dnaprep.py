import csv
with open('Data/DNA.tsv') as file:
    out = csv.reader(file, delimiter='\t')
    
    raw = []
    for i in out:
        
        raw.append(i)
        
    for i in raw:
        for x in range(8): i.pop(0)
        i.pop(8)
        i.pop(9)
        for x in range(49): i.pop(12)
        for x in range(9): i.pop()
    
    
    x = []
    for i in range(len(raw)):
        try:
            if 'N' in raw[i][12]:
                x.append(i)
        except:
            break
        
    popped = 0 
    for i in x:
        raw.pop(i - popped)
        popped += 1
    
    for i in range(len(raw)):
        pass
    # for i in raw:
    #     if i != ['phylum_taxID', 'phylum_name', 'class_taxID', 'class_name', 'order_taxID', 'order_name', 'family_taxID', 'family_name', 'subfamily_name', 'genus_name', 'species_taxID', 'species_name', 'nucleotides']:
    #         for x in [1, 3, 5, 7, 11]: # Positions of integers
    #             i[x-1] = float(i[x-1])
            
            
# print(raw[1])