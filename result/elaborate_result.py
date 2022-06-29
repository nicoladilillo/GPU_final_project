import os
import re
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('_mpl-gallery')

dfgs = { }

folder_name = "log_file"
for filename in os.listdir(folder_name):
    with open(os.path.join(folder_name, filename), 'r') as f: # open in readonly mode
        name_app = (filename.split('.')[0]).split('_')
        name = str(name_app[1] + "." + name_app[2])
        # print(name)
        textfile = f.read()
        matches_name = re.findall("DFG is ../DFGs_new/(.*)\.txt?",textfile )
        matches_area = re.findall("Area Limit is ([0-9]*)\nBest", textfile)
        matches_time = re.findall("The elapsed time is ([0-9]*) seconds?", textfile)
        for i in range(len(matches_name)):
            # print(f"{matches_name[i]} {matches_area[i]} {matches_time[i]}")
            dfg_name = str(matches_name[i] + "_" + matches_area[i])
            # print(dfg_name)
            if not dfg_name in dfgs:
                dfgs[dfg_name] = {}
            # print(f"{dfg_name} {name}")
            dfgs[dfg_name][name] = matches_time[i]

dfgs_CPU = { }
folder_name = "log_file_CPU"
for filename in os.listdir(folder_name):
    with open(os.path.join(folder_name, filename), 'r') as f: # open in readonly mode
        name_app = (filename.split('.')[0]).split('_')
        name = str(name_app[1] + "." + name_app[2])
        textfile = f.read()
        matches_name = re.findall("DFG is ../DFGs_new/(.*)\.txt?",textfile )
        matches_area = re.findall("Area Limit is ([0-9]*)\n", textfile)
        matches_time = re.findall("The elapsed time is ([0-9]*) seconds?", textfile)
        for i in range(len(matches_name)):
            dfg_name = str(matches_name[i] + "_" + matches_area[i])
            if not dfg_name in dfgs_CPU:
                dfgs_CPU[dfg_name] = {}
            dfgs_CPU[dfg_name][name] = matches_time[i]


# create a graph for each DFG where on x-axis put the name of version 
# program used while on y-axis the seconds

for k in dfgs.keys():
    # print(k)
    x = []
    y = []

    for v in dfgs.get(k):
        x.append(v)
        y.append(int(dfgs[k][v]))
        # print(f"\t {v}: {dfgs[k][v]}")
    if k in dfgs_CPU:
        for v in dfgs_CPU.get(k):
            x.append(v)
            y.append(int(dfgs_CPU[k][v]))
        
    # print(x)
    # print(y)

    # plot only GPU
    if max(y) >= 10 :

        # Create bars
        fig, ax = plt.subplots(figsize=(9,6))
        f = ax.bar(np.arange(len(x)) , y, 0.35)
        # Create names on the x-axis
        ax.set_ylabel('Time [s]') 
        ax.set_xticks(np.arange(len(x)), x)
        ax.set_xlabel('Version')
        name = re.search('^(.*)_([0-9]*)?', k)
        ax.set_title(str("DFG " + name.group(1) + " with area " + name.group(2)))
        ax.bar_label(f, padding=3)

        fig.tight_layout()

        plt.savefig(str("histogram/"+k+".png"), dpi = 600)
        

