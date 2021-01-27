
import sys
n = sys.argv[1]
exp_name = 'MO_LEDUC_EVAL'
iter_number = n

path_name_json = '/home/ec2-user/poker_ai_data/logs/' + exp_name + '/' + iter_number + '/as_json/logs.json'
path_dir_crayon = '/home/ec2-user/poker_ai_data/logs/' + exp_name + '/' + iter_number + '/crayon/'

import pandas as pd
import json
import pdb

def list_to_dict(input_list):
    my_dict = {}
    for a in input_list:
        akeys = a.keys()
        for t in akeys:
            mydict[t] = a[t]
    return my_dict

def fix_format(inp):
    mydict = {}
    for key in inp.keys():
        mydict[key] = list_to_dict(inp[key])
    return mydict

with open(path_name_json) as f:
    full = json.load(f)

dfs = {}
for x in full.keys():
    y = fix_format(full[x])
    df = pd.read_json(path_or_buf=y, orient='columns')
    dfs[x] = df

pdb.set_trace()


