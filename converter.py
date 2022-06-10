import os
import sys

folder = sys.argv[1]

for filename in os.listdir(folder):
    with open(str(folder + "/" + filename), 'r') as f: # open in readonly mode do your stuff
        print(str(folder + "/" + filename))

        # discharge first two rows
        f.readline()
        f.readline()

        # check number nodes
        nodes = 0
        l = f.readline().split()
        while l[1] != "->":
            name = l[0].strip()
            operation = l[3].strip()
            nodes += 1
            l = f.readline().split()
        print(str(nodes) + " nodes")

        # check number edges
        edges = 1
        l = f.readline().split()
        while l[0] != "}":
            node1 = l[0].strip()
            node2 = l[2].strip()
            edges += 1
            l = f.readline().split()
        print(str(edges) + " edges")

    # where new results will be saved
    filename_new = filename.replace("dot", "txt")
    folder_new   = str(folder + "_new")
    print("new folder: " + str(folder_new + "/" + filename_new))
    # reopen file
    with open(str(folder + "/" + filename), 'r') as f: # open in readonly mode do your stuff
        # discharge first two rows
        f.readline()
        f.readline()
        with open(str(folder_new + "/" + filename_new), 'w') as f_w:
            f_w.write(str(nodes) + "\n")
            for i in range(nodes):
                l = f.readline().split()
                name = l[0].strip()
                operation = l[3].strip()
                f_w.write(name + " " + operation + "\n")
                
            f_w.write(str(edges) + "\n")
            for i in range(edges):
                l = f.readline().split()
                node1 = l[0].strip()
                node2 = l[2].strip()
                f_w.write(node1 + " " + node2 + "\n")