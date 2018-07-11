import os
from pathlib import Path

import subprocess
from threading import Timer

home = str(Path.home())
config_dir = home + "/dsbox/runs2/config-ll0/"

# https://www.blog.pythonlibrary.org/2016/05/17/python-101-how-to-timeout-a-subprocess/
kill = lambda process: process.kill()
timers = []
for conf in os.listdir(config_dir):
	command = "python ta2-search " + config_dir + conf
	print(command)

	curr_proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	curr_timer = Timer(60, kill, [curr_proc])

	timers.append(curr_timer)

for timer in timers:
	timer.start()


# for conf in os.listdir(config_dir):
# 	command = "python ta2-search " + config_dir + conf
# 	print(command)
# 	subprocess.Popen(command, shell=True)
