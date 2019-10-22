import csv

csvfile=open('order_products__train.csv', newline='')

rows=csv.reader(csvfile)

this_order_id=None

first_id=False

dataset=[]

for row in rows:
    #print(row)   
    
    if first_id==False:
        dataset.append([])
        this_order_id=row[0]
        first_id=True    

    if row[0]==this_order_id:
        dataset[-1].append(row[1])
    
    else:
        dataset.append([])
        dataset[-1].append(row[1])

    this_order_id=row[0]

dataset=dataset[1:]

print(dataset[:3])