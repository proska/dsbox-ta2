#!/usr/bin/env python

"""
Command Line Interface for running the DSBox TA2 Search
"""

from dsbox_dev_setup import path_setup
import argparse
import json
import os
import signal
import sys
import traceback

from pathlib import Path
from pprint import pprint

from importlib import reload
import dsbox.controller.controller
reload(dsbox.controller.controller)
from dsbox.controller.controller import Controller
import os
controller = Controller('/', development_mode=True)

def main(args):

    library_directory = os.path.dirname(
        os.path.realpath(__file__)) + "/library"
    timeout = 0
    configuration_file = args.configuration_file
    debug = args.debug

    controller = Controller(library_directory, development_mode=debug)

    with open(configuration_file) as data:
        config = json.load(data)

    # Define signal handler to exit gracefully
    # Either on an interrupt or after a certain time
    def write_results_and_exit(a_signal, frame):
        print('SIGNAL exit: {}'.format(configuration_file))
        try:
            # Reset to handlers to default as not to output multiple times
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            print('SIGNAL exit done reset signal {}'.format(configuration_file))

            controller.write_training_results()
            print('SIGNAL exit done writing: {}'.format(
                configuration_file), flush=True)
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # sys.exit(0) generates SystemExit exception, which may
            # be caught and ignore.

            # This os._exit() cannot be caught.
            print('SIGNAL exiting {}'.format(configuration_file), flush=True)
            # persist running
            return -1
#            os._exit(0)

    # Set timeout, alarm and signal handler
    if 'timeout' in config:
        # Timeout less 1 minute to give system chance to clean up
        timeout = int(config['timeout']) - 1
    if args.timeout > -1:
        # Overide config timeout
        timeout = args.timeout
    if timeout > 0:
        signal.signal(signal.SIGINT, write_results_and_exit)
        signal.signal(signal.SIGTERM, write_results_and_exit)
        signal.signal(signal.SIGALRM, write_results_and_exit)
        signal.alarm(60*timeout)
    config['timeout'] = timeout

    if args.cpus > -1:
        config['cpus'] = args.cpus

    # Replace output directories
    if args.output_prefix is not None:
        for key in ['pipeline_logs_root', 'executables_root', 'temp_storage_root']:
            if not '/output/' in config[key]:
                print(
                    'Skipping. No "*/output/" for config[{}]={}.'.format(key, config[key]))
            else:
                suffix = config[key].split('/output/', 1)[1]
                config[key] = os.path.join(args.output_prefix, suffix)

    os.system('clear')
    print('Using configuation:')
    pprint(config)

    # controller.initialize_from_config(config)
    controller.initialize_from_config_train_test(config)

    status = controller.train()

    return status.value


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run DSBox TA2 system on all the datasets'
    )

    parser.add_argument('--configuration_file', action='store', type=str, default=None,
                        help='D3M TA2 json configuration file')
    parser.add_argument('--timeout', action='store', type=int, default=-1,
                        help='Overide configuation timeout setting. In minutes.')
    parser.add_argument('--cpus', action='store', type=int, default=-1,
                        help='Overide configuation number of cpus usage setting')
    parser.add_argument('--output_prefix', action='store', type=str, default=None,
                        help='''Overide configuation output directories paths (pipeline_logs_root, executables_root, temp_storage_root).
                        Replace path prefix "*/output/" with argument''')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode. No timeout and no output redirection')

    args = parser.parse_args()

    home = str(Path.home())

    for conf in os.listdir(home + "/dsbox/runs2/config-seed/"):
        print("Working for", conf)

        args.configuration_file = "/nas/home/stan/dsbox/runs2/config-seed/" + conf

        try:
            result = main(args)
            print("[INFO] Run succesfull")
        except:
            print("[ERROR] Failed dataset", conf)

        print("\n" * 10)

    #result = main(args)
    #os._exit(result)
