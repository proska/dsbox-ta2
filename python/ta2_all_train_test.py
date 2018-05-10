from dsbox_dev_setup import path_setup
path_setup()

import argparse
import sys
import os
import pandas
import json
import shutil
import signal
import sklearn.externals

from dsbox.executer.executionhelper import ExecutionHelper
from dsbox.planner.common.data_manager import Dataset, DataManager
from dsbox.planner.common.problem_manager import Problem
from dsbox.planner.controller import Controller, Feature
from dsbox.planner.event_handler import PlannerEventHandler

from pathlib import Path


TIMEOUT = 25*60 # Timeout after 25 minutes

DEBUG = 0
LIB_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/library"

DATA_DIR = '/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data'

def process_dataset(config):
    if "timeout" in config:
        # Timeout less 60 seconds, to give system chance to clean up
        TIMEOUT = int(config.get("timeout"))*60 - 60

    # Start the controller
    controller = Controller(LIB_DIRECTORY)
    controller.initialize_from_config(config)
    controller.load_problem()

    # Setup a signal handler to exit gracefully
    # Either on an interrupt or after a certain time
    def write_results_and_exit(signal, frame):
        controller.write_training_results()
        sys.exit(0)
    signal.signal(signal.SIGINT, write_results_and_exit)
    signal.signal(signal.SIGTERM, write_results_and_exit)
    signal.signal(signal.SIGALRM, write_results_and_exit)
    signal.alarm(TIMEOUT)
    
    # Load in data
    controller.initialize_training_data_from_config()

    # Start training
    controller.initialize_planners()
    for result in controller.train(PlannerEventHandler()):
        if result == False:
            print("ProblemNotImplemented")
            sys.exit(148)
        pass

    # Start testing
    problem = Problem()
    problem.load_problem(config["problem_root"], config["problem_schema"])

    # FIXME: considering test_data_root as training_data_root
    if "test_data_root" not in config:
        config["test_data_root"] = config["training_data_root"]

    # FIXME HACKY - to save space, output results in home directory and remove current outputs folder
    home = "/nas/home/stan"
    problem_name = problem.prID.rsplit("_", 1)[0]
    output_file = home + "/outputs/" + problem_name + ".txt"
    f = open(output_file, "w+")

    # Load in the dataset, data manager and execution helper
    dataset = Dataset()
    dataset.load_dataset(config["test_data_root"], config["dataset_schema"])

    data_manager = DataManager()
    data_manager.initialize_data(problem, [dataset], view='TEST')

    hp = ExecutionHelper(problem, data_manager)

    initial_testdata = data_manager.input_data

    # for each pipeline compute its result on the test data by looking at the pickle files
    for ppln in controller.exec_pipelines:
        print("Processing", ppln.primitives)

        primfile_base = config["temp_storage_root"] + "/models/" + ppln.id
        num_primitives = len(ppln.primitives)

        curr_testdata = initial_testdata.copy()
        last_primitive = None

        for i in range(num_primitives):
            primfile = primfile_base + ".primitive_" + str(i + 1) + ".pkl"

            primitive = sklearn.externals.joblib.load(primfile)
            last_primitive = primitive

            if primitive.task == "PreProcessing":
                new_testdata = hp.test_execute_primitive(primitive, curr_testdata)
            elif primitive.task == "FeatureExtraction":
                new_testdata = hp.test_featurise(primitive, curr_testdata)
            elif primitive.task == "Modeling":
                continue

            curr_testdata = new_testdata.copy()

        assert last_primitive != None

        result = pandas.DataFrame(last_primitive.executables.produce(inputs=curr_testdata).value, index = curr_testdata.index, \
               columns = [cn["colName"] for cn in data_manager.target_columns])
#        result.to_csv("/nas/home/stan/outputs/script_result.csv")

        # For each metric, compute the result
        for metric in problem.metrics:
            metric_function = problem._get_metric_function(metric)

            training_score = ppln.planner_result.metric_values[metric.name]
            test_score = hp._call_function(metric_function, data_manager.target_data, result)

            f.write("%s %s %s %s %s\n" % (metric.name, ppln.id, ppln.primitives, training_score, test_score))

    f.close()

    print("Cleaning temp folder...")

    temp_output_folder = config["temp_storage_root"].rsplit("/", 1)[0]
    shutil.move(temp_output_folder, home + "/outputs")

#    print("Succesfully removed", temp_output_folder)


def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--include", dest="include",type=str, nargs='+', default=[], help="list of families, algo types, or primitives to include")
    parser.add_argument("-e", "--exclude", dest="exclude",type=str, nargs='+', default=[],  help="list of families, algo types, or primitives to exclude")
    parser.add_argument("-d", "--dataset", dest="dataset",type=str, default= '/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/training_datasets/LL0/', help="dataset path")
    parser.add_argument("-o", "--output", dest="output",type=str, default= 'outputs/', help='output dir')
    args = parser.parse_args()

    dataset_folder = args.dataset #sys.argv[1]
    output_dir = args.output #sys.argv[2]
        
    include = args.include
    exclude = args.exclude #sys.argv[4]
    print(include)
    print(exclude)

    prnt = True
    failures = []
    count = 0
    for folder, sub_dirs, _ in os.walk(dataset_folder):

        dataset = folder.split('/')[-1]

        print(dataset)

        if folder != dataset_folder and '.git' not in folder:
            config = {}
            del sub_dirs[:]
            for i in range(20):
                print()
            print('RUNNING ', dataset)
            config["problem_schema"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_problem'), 'problemDoc.json')
            config["problem_root"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_problem'))
            config["dataset_schema"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_dataset'), 'datasetDoc.json')
            config["training_data_root"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_dataset'))
            config["pipeline_logs_root"] =  os.path.join(os.getcwd(), output_dir, str(dataset),'logs')
            config["executables_root"] =  os.path.join(os.getcwd(), output_dir, str(dataset),'executables')  
            config["temp_storage_root"] =  os.path.join(os.getcwd(), output_dir, str(dataset),'temp')
            config["timeout"] = TIMEOUT

            config["include"] = include
            config["exclude"] = exclude

            #config["include"] = ["d3m.primitives.sklearn_wrap.SKAdaBoostClassifier"]
            #config["exlucde"] = ["*"]

            # don't change timeout, cpus, ram?

            try:
                process_dataset(config)
                count = count + 1
            except:
                failures.append(dataset)
                continue

#        if count == 3:
#            break

    print("Completed Running ", count, " Pipelines")
    print("Failed on datasets:")
    for fail in failures:
        print(fail)


if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
    sys.exit(main())
    #main()
