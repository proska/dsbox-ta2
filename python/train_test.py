import os
import subprocess
import sys

from multiprocessing.pool import ThreadPool
from pathlib import Path

timeout = 20
cpus = 10
num_threads = 10

def call_ta2search(command):
    print(command)

    p = subprocess.Popen(command, shell=True)

    try:
        p.communicate(timeout=timeout * 60)
    except TimeoutExpired:
        p.kill()
        print(command, "took too long and was terminated" + "\n\n")

tp = ThreadPool(num_threads)

home = str(Path.home())
# config_dir = sys.argv[2]
config_dir = home + "/dsbox/runs2/config-ll0/"

for conf in os.listdir(config_dir):
    # command = "python3 ta1-run-single-template --template " + sys.argv[1] + " " + os.path.join(config_dir, conf, 'search_config.json')
    command = "python ta2-search " + config_dir + conf + " --timeout " + str(timeout) + " --cpus " + str(cpus)

    tp.apply_async(call_ta2search, (command,))

tp.close()
tp.join()
