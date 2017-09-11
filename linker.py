#!/usr/bin/python3

import os
import shutil
import subprocess

events = [
	"bridge", 
	"city_entry", 
	"city_exit", 
	"road_bump",
	"screen_wipers",
	"zebra",
]

for ev in events:
	try:
		shutil.rmtree(ev)
	except:
		pass

for ev in events:
	os.mkdir(ev)

with open("train.txt", "r") as fin:
	for filename, mask in [line.split() for line in fin.readlines()]:
		for i in range(len(mask)):
			if int(mask[i]):
				subprocess.run(["ln", filename, events[i] + "/" + filename])