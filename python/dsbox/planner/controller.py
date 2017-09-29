import os
import sys
import os.path
import uuid
import copy
import math
import json
import numpy
import shutil
import traceback
import pandas as pd

from dsbox.planner.leveltwo.l1proxy import LevelOnePlannerProxy
from dsbox.planner.leveltwo.planner import LevelTwoPlanner
from dsbox.schema.data_profile import DataProfile
from dsbox.schema.problem_schema import TaskType, TaskSubType, Metric
from dsbox.executer.executionhelper import ExecutionHelper
from dsbox.planner.common.data_manager import DataManager
from dsbox.planner.common.pipeline import Pipeline, PipelineExecutionResult
from dsbox.planner.common.data_manager import DataManager
from dsbox.planner.common.schema_manager import SchemaManager

class Feature:
    def __init__(self, data_directory, feature_id):
        self.data_directory = data_directory
        self.feature_id = feature_id

class Controller(object):
    """
    This is the overall "planning" coordinator. It is passed in the data directory
    and the primitives library directory, and it generates plans by calling out to L1, L2
    and L3 planners.
    """
    def __init__(self, libdir):
        self.libdir = os.path.abspath(libdir)
        self.train_dm = DataManager()
        self.test_dm = DataManager()
        self.sm = SchemaManager()
        self.helper = ExecutionHelper(self.train_dm, self.sm)

        self.exec_pipelines = []
        self.l1_planner = None
        self.l2_planner = None

    '''
    Set config directories and data schema file
    '''
    def set_config(self, config):
        self.config = config
        self.data_schema = config.get('dataset_schema', None)
        self.problem_schema = config.get('problem_schema', None)
        self.train_dir = self._dir(config, 'training_data_root')
        self.log_dir = self._dir(config, 'pipeline_logs_root')
        self.exec_dir = self._dir(config, 'executables_root')
        self.tmp_dir = self._dir(config, 'temp_storage_root')

        # Create some debugging files
        self.logfile = open("%s%slog.txt" % (self.tmp_dir, os.sep), 'w')
        self.errorfile = open("%s%sstderr.txt" % (self.tmp_dir, os.sep), 'w')
        self.pipelinesfile = open("%s%spipelines.txt" % (self.tmp_dir, os.sep), 'w')

        # Redirect stderr to error file
        sys.stderr = self.errorfile

    '''
    Set config directories and schema from just datadir and outputdir
    '''
    def set_config_simple(self, datadir, outputdir):
        self.set_config({
            'dataset_schema': datadir + os.sep + "dataSchema.json",
            'problem_schema': datadir + os.sep + ".." + os.sep + "problemSchema.json",
            'training_data_root': datadir,
            'pipeline_logs_root': outputdir + os.sep + "logs",
            'executables_root': outputdir + os.sep + "executables",
            'temp_storage_root': outputdir + os.sep + "temp"
        })


    """
    Set the task type, metric and output type via the schema
    """
    def load_problem_schema(self):
        self.sm.load_problem_schema(self.problem_schema)

    def initialize_training_data_from_defaults(self):
        self.train_dm.initialize_training_data_from_defaults(self.data_schema, self.train_dir)

    """
    Initialize the L1 and L2 planners
    """
    def initialize_planners(self):
        self.l1_planner = LevelOnePlannerProxy(self.libdir, self.helper)
        self.l2_planner = LevelTwoPlanner(self.libdir, self.helper)

    """
    Train and select pipelines
    """
    def train(self, planner_event_handler, cutoff=10):
        self.exec_pipelines = []
        self.l2_planner.primitive_cache = {}
        self.l2_planner.execution_cache = {}

        self.logfile.write("Task type: %s\n" % self.helper.sm.task_type)
        self.logfile.write("Metrics: %s\n" % self.helper.sm.metrics)

        pe = planner_event_handler

        self._show_status("Planning...")

        # Get data details
        df = pd.DataFrame(copy.copy(self.train_dm.data.input_data))
        df_lbl = pd.DataFrame(copy.copy(self.train_dm.data.target_data))

        df_profile = DataProfile(df)
        self.logfile.write("Data profile: %s\n" % df_profile)

        l1_pipelines_handled = {}
        l2_pipelines_handled = {}
        l1_pipelines = self.l1_planner.get_pipelines(df)
        self.exec_pipelines = []

        while len(l1_pipelines) > 0:
            self.logfile.write("\nL1 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l1_pipelines))
            self.logfile.write("-------------\n")

            l2_l1_map = {}

            self._show_status("Exploring %d basic pipeline(s)..." % len(l1_pipelines))

            l2_pipelines = []
            for l1_pipeline in l1_pipelines:
                if l1_pipelines_handled.get(str(l1_pipeline), False):
                    continue
                l2_pipeline_list = self.l2_planner.expand_pipeline(l1_pipeline, df_profile)
                l1_pipelines_handled[str(l1_pipeline)] = True
                if l2_pipeline_list:
                    for l2_pipeline in l2_pipeline_list:
                        if not l2_pipelines_handled.get(str(l2_pipeline), False):
                            l2_l1_map[l2_pipeline.id] = l1_pipeline
                            l2_pipelines.append(l2_pipeline)
                            yield pe.SubmittedPipeline(l2_pipeline)

            self.logfile.write("\nL2 Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(l2_pipelines))

            self._show_status("Found %d executable pipeline(s). Testing them..." % len(l2_pipelines))

            for l2_pipeline in l2_pipelines:
                yield pe.RunningPipeline(l2_pipeline)

                # TODO: Execute in parallel (fork, or separate thread)
                exec_pipeline = self.l2_planner.patch_and_execute_pipeline(l2_pipeline, df, df_lbl)
                l2_pipelines_handled[str(l2_pipeline)] = True
                yield pe.CompletedPipeline(l2_pipeline, exec_pipeline)

                if exec_pipeline:
                    self.exec_pipelines.append(exec_pipeline)

            self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))
            self.logfile.write("\nL2 Executed Pipelines:\n-------------\n")
            self.logfile.write("%s\n" % str(self.exec_pipelines))

            # TODO: Do Pipeline Hyperparameter Tuning

            # Pick top N pipelines, and get similar pipelines to it from the L1 planner to further explore
            l1_related_pipelines = []
            for index in range(0, cutoff):
                if index >= len(self.exec_pipelines):
                    break
                l1_pipeline = l2_l1_map.get(self.exec_pipelines[index].id)
                if l1_pipeline:
                    related_pipelines = self.l1_planner.get_related_pipelines(l1_pipeline)
                    for related_pipeline in related_pipelines:
                        if not l1_pipelines_handled.get(str(related_pipeline), False):
                            l1_related_pipelines.append(related_pipeline)

            self.logfile.write("\nRelated L1 Pipelines to top %d L2 Pipelines:\n-------------\n" % cutoff)
            self.logfile.write("%s\n" % str(l1_related_pipelines))
            l1_pipelines = l1_related_pipelines

        self.write_training_results()


    '''
    Write training results to file
    '''
    def write_training_results(self):
        # Sort pipelines
        self.exec_pipelines = sorted(self.exec_pipelines, key=lambda x: self._sort_by_metric(x))

        # Ended planners
        self._show_status("Found total %d successfully executing pipeline(s)..." % len(self.exec_pipelines))

        # Create executables
        self.pipelinesfile.write("# Pipelines ranked by metrics (%s)\n" % self.sm.metrics)
        for index in range(0, len(self.exec_pipelines)):
            pipeline = self.exec_pipelines[index]
            rank = index + 1
            # Format the metric values
            metric_values = []
            for metric in pipeline.planner_result.metric_values.keys():
                metric_value = pipeline.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))

            self.pipelinesfile.write("%s ( %s ) : %s\n" % (pipeline.id, pipeline, metric_values))
            self.helper.create_pipeline_executable(pipeline, self.config)
            self.create_pipeline_logfile(pipeline, rank)

    '''
    Predict results on test data given a pipeline
    '''
    def test(self, pipeline, test_event_handler):
        helper = ExecutionHelper(self.test_dm, self.sm)
        testdf = pd.DataFrame(copy.copy(self.test_dm.data.input_data))
        target_col = self.test_dm.data.target_columns[0]['varName']
        print("** Evaluating pipeline %s" % str(pipeline))
        for primitive in pipeline.primitives:
            # Initialize primitive
            try:
                print("Executing %s" % primitive)
                if primitive.task == "Modeling":
                    result = pd.DataFrame(primitive.executables.predict(testdf), index=testdf.index, columns=[target_col])
                    pipeline.test_result = PipelineExecutionResult(result, None)
                    break
                elif primitive.task == "PreProcessing":
                    testdf = helper.test_execute_primitive(primitive, testdf)
                elif primitive.task == "FeatureExtraction":
                    testdf = helper.test_featurise(testdf, primitive)
                if testdf is None:
                    break
            except Exception as e:
                sys.stderr.write(
                    "ERROR test(%s) : %s\n" % (pipeline, e))
                traceback.print_exc()

        yield test_event_handler.ExecutedPipeline(pipeline)

    def stop(self):
        '''
        Stop planning, and write out the current list (sorted by metric)
        '''

    def create_pipeline_logfile(self, pipeline, rank):
        logfilename = "%s%s%s.json" % (self.log_dir, os.sep, pipeline.id)
        logdata = {
            "problem_id": self.sm.problem_id,
            "pipeline_rank": rank,
            "name": pipeline.id,
            "primitives": []
        }
        for primitive in pipeline.primitives:
            logdata['primitives'].append(primitive.cls)
        with(open(logfilename, 'w')) as pipelog:
            json.dump(logdata, pipelog,
                sort_keys=True, indent=4, separators=(',', ': '))
            pipelog.close()

    def _dir(self, config, key):
        dir = config.get(key)
        if dir is None:
            return None
        dir = os.path.abspath(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _show_status(self, status):
        sys.stdout.write("%s\n" % status)
        sys.stdout.flush()

    def _sort_by_metric(self, pipeline):
        # NOTE: Sorting/Ranking by first metric only
        metric_name = self.sm.metrics[0].name
        mlower = metric_name.lower()
        if "error" in mlower or "loss" in mlower or "time" in mlower:
            return pipeline.planner_result.metric_values[metric_name]
        return -pipeline.planner_result.metric_values[metric_name]