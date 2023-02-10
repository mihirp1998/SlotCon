import pickle
import ipdb
st = ipdb.set_trace

txtval = open('map_clsloc.txt','r')
lines = txtval.readlines()
dict_val = {}
for line in lines:
    line = line.split(' ')
    dict_val[line[0]] = line[1:]

pickle.dump(dict_val,open('map_clsloc.p','wb'))

# st()
# print(lines)