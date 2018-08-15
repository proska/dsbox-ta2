import time
import json
import os
import signal
import subprocess
import traceback
import sys
from pprint import pprint

from dsbox.controller.controller import Controller
from dsbox.controller.controller import Status

start_time = time.time()


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_pid, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    assert retcode == 0, "ps command returned %d" % retcode
    print('parent id={}'.format(parent_pid), flush=True)
    for pid_str in ps_output.decode('utf-8').split("\n")[:-1]:
        try:
            print('chdild id={}'.format(pid_str), flush=True)
            os.kill(int(pid_str), sig)
        except:
            pass


def main():
    timeout = 0
    if os.environ["D3MRUN"] == "search":
        config = json.load(open(os.path.join(os.environ["D3MINPUTDIR"], "search_config.json"), 'r'))
        write_results_time = 2
        timeout = int(os.environ["D3MTIMEOUT"]) - write_results_time
    else:
        config = json.load(open(os.path.join(os.environ["D3MINPUTDIR"], "test_config.json"), 'r'))
        timeout = 60

    config["cpus"] = os.environ["D3MCPU"]
    config["ram"] = os.environ["D3MRAM"]

    # Time to write results (in minutes)
    config["timeout"] = timeout

    controller = Controller(development_mode=False)

    # Define signal handler to exit gracefully
    def write_results_and_exit(a_signal, frame):
        print('==== Times up ====')
        time_used = (time.time() - start_time) / 60.0
        print("[INFO] The time used so far is {:0.2f} minutes.".format(time_used))
        try:
            # Reset to handlers to default as not to output multiple times
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

            print('[INFO] Killing child processes', flush=True)
            process_id = os.getpid()
            kill_child_processes(process_id)

            print('[INFO] writing results', flush=True)
            controller.write_training_results()

            print('==== Done cleaning up ====', flush=True)
            time_used = (time.time() - start_time) / 60.0
            print("[INFO] The time used so far is {:0.2f} minutes.".format(time_used), flush=True)
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # sys.exit(0) generates SystemExit exception, which may
            # be caught and ignored.

            # This os._exit() cannot be caught.
            # print('SIGNAL exiting {}'.format(configuration_file), flush=True)
            os._exit(0)

    if timeout > 0:
        signal.signal(signal.SIGALRM, write_results_and_exit)
        signal.alarm(60 * timeout)
    else:
        raise Exception('Negative timeout {}'.format(timeout))

    print('Using configuation:')
    pprint(config)

    if 'training_data_root' in config:
        print("[INFO] Now in training process")
        controller.initialize_from_config_for_evaluation(config)
        status = controller.train()
        print("[INFO] Training Done")
        # print("*+"*10)
    elif 'test_data_root' in config:
        print("[INFO] Now in testing process")
        controller.initialize_from_config_for_evaluation(config)
        pipelines_dir = os.path.join(os.environ["D3MOUTPUTDIR"], 'pipelines')
        pipelines = [os.path.join(pipelines_dir, f) for f in os.listdir(pipelines_dir) if f.endswith(".json")]
        rank_lst = list()
        for pipeline in pipelines:
            try:
                pipeline_json = json.load(open(pipeline, 'r'))
                rank_lst.append((pipeline_json["pipeline_rank"], pipeline_json['id']))
            except:
                pass
        fitted_pipeline_id = min(rank_lst)[1]
        status = controller.test_fitted_pipeline(fitted_pipeline_id=fitted_pipeline_id)
        print("[INFO] Testing Done")
    else:
        status = Status.PROBLEM_NOT_IMPLEMENT
        print("[ERROR] Neither train or test root was given, the program will exit.")

    time_used = (time.time() - start_time) / 60.0
    print("[INFO] The time used for running program is {:0.2f} minutes.".format(time_used))

    return status.value


if __name__ == "__main__":

    result = main()
    os._exit(result)
