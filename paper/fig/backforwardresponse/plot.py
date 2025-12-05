#!/bin/python

import sys
import numpy as np

from decimal import *

macro_file = open(sys.argv[1],'r')


backvalues =[]
forwardvalues =[]

for line in macro_file:
    line = line.strip()
    if 'backgraphtime' in line:
        backvalues.append(float(line.split('{')[2].split('}')[0]))
    if 'forwardgraphtime' in line:
        forwardvalues.append(float(line.split('{')[2].split('}')[0]))

back_N = len(backvalues)
forward_N = len(forwardvalues)

backvalues = sorted(backvalues)
forwardvalues = sorted(forwardvalues)

with open('./backvalues.txt', 'w') as fil:
    prev = -1
    for i in range(0,len(backvalues)):
        if prev != backvalues[i]:
            fil.write(str(backvalues[i]) + " "+str( float(i) / float(back_N))+"\n")
        prev = backvalues[i]

with open('./forwardvalues.txt', 'w') as fil:
    prev = -1
    for i in range(0,len(forwardvalues)):
        if prev != forwardvalues[i]:
            fil.write(str(forwardvalues[i])+" "+str(float(i) / float(forward_N))+"\n")
        prev = forwardvalues[i]
