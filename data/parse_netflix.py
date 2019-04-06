movie_count = 17770 #set this to smaller number if tensor os too large
route = "training_set/"

uid_dict = {}
uid_counter = 0
date_dict = {}
date_counter = 0

for i in range(1, movie_count+1):
    if i % 100 == 0:
        print("indexing movie #" + str(i)) #print progress

    index = str(i)
    for j in range(7 - len(index)):
        index = "0" + index
    filename = route + "mv_" + index + ".txt" #generate file name
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(line.strip())
    for j in range(1, len(data)):
        entry = data[j].split(",")
        if entry[0] not in uid_dict:
            uid_dict[entry[0]] = str(uid_counter) # if uid haven't seen before, assign new index
            uid_counter += 1
        if entry[2] not in date_dict:
            date_dict[entry[2]] = date_counter # if date haven't seen before, add to dict
            date_counter += 1
dates = list(date_dict.keys())
date_dict = {}
dates.sort() # sort dates


for i in range(len(dates)):
    date_dict[dates[i]] = str(i) # put sorted dates back into dict for indexing

output = []
for i in range(1, movie_count+1):
    if i % 100 == 0:
        print("writing movie #" + str(i))
    index = str(i)
    for j in range(7 - len(index)):
        index = "0" + index
    filename = "../../Downloads/download/training_set/mv_" + index + ".txt"
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(line.strip())
    for j in range(1, len(data)):
        entry = data[j].split(",")
        output.append(" ".join([uid_dict[entry[0]], str(i-1), date_dict[entry[2]], entry[1]])) # put entry into ctf text format

out = open ("tensor.txt", "w")
print("\n".join(output), file = out)
out.close()

#print(len(uid_dict.keys()), movie_count, len(dates))
print(uid_counter, movie_count, date_counter)
print("finished")
