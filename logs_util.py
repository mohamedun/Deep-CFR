
import sys
import pandas as pd
import json
import pdb

def logs_to_dfs(exp_name, iter_number):
    path_name_json = '/home/ec2-user/poker_ai_data/logs/' + exp_name + '/' + str(iter_number-1) + '/as_json/logs.json'

    def list_to_dict(input_list):
        my_dict = {}
        for a in input_list:
            akeys = a.keys()
            for t in akeys:
                my_dict[t] = a[t]
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
        df = pd.DataFrame.from_dict(data=y, orient='columns')
        df.index = df.index.map(int)
        df.sort_index(inplace=True)
        dfs[x] = df

    return dfs

