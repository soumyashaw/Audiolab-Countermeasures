import sys
import numpy as np

file_path = sys.argv[1]

y_pred_proba = []
y_test = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.split()
        #print(line)
        if line[2] == "bonafide" and float(line[3])>0.5:
            truePositive += 1
        elif line[2] == "bonafide" and float(line[3])<=0.5:
            falseNegative +=1
        elif line[2] == "spoof" and float(line[3])<=0.5:
            trueNegative +=1
        else:
            falsePositive += 1
