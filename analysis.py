# -*- coding: utf-8 -*- 
import json
import glob

for data_file in glob.glob(f"result.json"):
    #print(data_file)
    with open(data_file) as f:
        res = json.load(f)

    ntotal = len(res)-1
    count = 0
    for x in res:
        if x.get('elapsed_time') == None:
            pass
        else:
            break
        if x['hit']:
            count = count + 1

    print(f"{data_file.ljust(60)} : {float(count)/ntotal}")

