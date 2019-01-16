import copy
import logging
import sys
import time
import typing
import enum
# import eventlet

from multiprocessing import current_process
from pprint import pprint
from warnings import warn

from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.exceptions import NotSupportedError
from d3m.metadata.base import Metadata
from d3m.metadata.problem import PerformanceMetric
from dsbox.JobManager.cache import PrimitivesCache
from dsbox.combinatorial_search.search_utils import get_target_columns
from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.utils import larger_is_better
from dsbox.schema.problem import OptimizationType
from dsbox.schema.problem import optimization_type
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.configuration_space import ConfigurationSpace
from dsbox.template.template import DSBoxTemplate

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)

class Mode(enum.IntEnum):
    CROSS_VALIDATION_MODE = 1
    TRAIN_TEST_MODE = 2

class MetaMetric(type):
    @property
    def classification_metric(cls):
        return cls.classification_metric
    @property
    def regression_metric(cls):
        return cls.regression_metric

class SpecialMetric(object, metaclass=MetaMetric):
    # TODO These variables have not been used at all
    classification_metric = ('accuracy', 'precision', 'normalizedMutualInformation',
                                  'recall', 'f1', 'f1Micro', 'f1Macro', 'rocAuc', 'rocAucMicro',
                                  'rocAucMacro')
    regression_metric = ('meanSquaredError', 'rootMeanSquaredError',
                              'rootMeanSquaredErrorAvg', 'meanAbsoluteError', 'rSquared',
                              'jaccardSimilarityScore', 'precisionAtTopK')

class ConfigurationSpaceBaseSearch(typing.Generic[T]):
    """
    Search configuration space on dimension at a time.

    Attributes
    ----------
    evaluate : Callable[[typing.Dict], float]
        Evaluate given point in configuration space
    configuration_space: ConfigurationSpace[T]
        Definition of the configuration space
    minimize: bool
        If True, minimize the value returned by `evaluate` function

    TODO:
        1. break the evaluation method into multiple train-test method calls.
        2. Make the evaluation parametrized on the sort of evaluation mode
           ( ('partial', 'whole'), ('bootstrap', 'cross-validation'))
    """

    def __init__(self, template: DSBoxTemplate,
                 configuration_space: ConfigurationSpace[T],
                 problem: Metadata, train_dataset1: Dataset,
                 train_dataset2: typing.List[Dataset], test_dataset1: Dataset,
                 test_dataset2: typing.List[Dataset], all_dataset: Dataset,
                 ensemble_tuning_dataset: Dataset,
                 performance_metrics: typing.List[typing.Dict], output_directory: str,
                 log_dir: str, ) -> None:

        self.template = template
        self.task_type = self.template.template["taskType"]

        self.configuration_space = configuration_space
        # self.dimension_ordering = configuration_space_list.get_dimension_search_ordering()

        self.problem = problem
        self.train_dataset1 = train_dataset1
        self.train_dataset2 = train_dataset2
        self.test_dataset1 = test_dataset1
        self.test_dataset2 = test_dataset2
        self.all_dataset = all_dataset
        if ensemble_tuning_dataset:
            self.do_ensemble_tuning = True
            self.ensemble_tuning_dataset = ensemble_tuning_dataset
        else:
            self.do_ensemble_tuning = False
            self.ensemble_tuning_dataset = None

        self.performance_metrics = list(map(
            lambda d: {'metric': d['metric'].unparse(), 'params': d['params']},
            performance_metrics
        ))

        self.output_directory = output_directory
        self.log_dir = log_dir

        self.minimize = optimization_type(performance_metrics[0]['metric']) == \
                        OptimizationType.MINIMIZE

        self.quick_mode = False
        self.testing_mode = 0  # set default to not use cross validation mode
        # testing_mode = 0: normal testing mode with test only 1 time
        # testing_mode = 1: cross validation mode
        # testing_mode = 2: multiple testing mode with testing with random split data n times
        self.validation_config = None

        for each_step in template.template['steps']:
            if 'runtime' in each_step:
                self.validation_config = each_step['runtime']
                if "cross_validation" in each_step['runtime']:
                    self.testing_mode = 1
                    _logger.debug("Will use cross validation(n ={}) to choose best primitives".format(int(self.validation_config['cross_validation'])))
                    print("!!!!!@@#### CV mode!!!")
                    break
                else:
                    self.testing_mode = 2
                    _logger.debug("Will use test_dataset to choose best primitives")
                    print("!!!!!@@#### normal mode!!!")
                    break

        # new searching method: first check whether we should train a second time with
        # dataset_train1
        self.go_quick_inputType = ["image", "audio", "video"]
        self.quick_mode = self._use_quick_mode_or_not()

    def _use_quick_mode_or_not(self) -> bool:
        """
        The function to determine whether to use quick mode or now
            Now it is hard coded
        Returns:
            use_quick mode?
        """
        for each_type in self.template.template['inputType']:
            if each_type in self.go_quick_inputType:
                return True
        return False

    # def dummy_evaluate(self, ) -> None:
    #     """
    #     This method is only used to import tensorflow before running the parallel jobs
    #     Args:
    #         configuration:
    #
    #     Returns:
    #
    #     """
    #     _logger.info("Dummy evaluation started")
    #     configuration: ConfigurationPoint[PrimitiveDescription] = \
    #         self.configuration_space.get_first_assignment()
    #
    #     pipeline = self.template.to_pipeline(configuration)
    #     return pipeline

    def evaluate_pipeline(self, args) -> typing.Dict:
        """
        Evaluate at configuration point.
        Note: This methods will modify the configuration point, by updating its data field.
        """

        configuration: ConfigurationPoint[PrimitiveDescription] = dict(args[0])
        cache: PrimitivesCache = args[1]
        dump2disk = args[2] if len(args) == 3 else True

        _logger.info(f"START Evaluation of {hash(str(configuration))} in {current_process()}")

        evaluation_result = self._evaluate(configuration, cache, dump2disk)

        evaluation_result.pop('fitted_pipeline')

        _logger.info(f"END Evaluation of {hash(str(configuration))} in {current_process()}")

        return evaluation_result
        # assert hasattr(evaluation_result['fitted_pipeline'], 'runtime'), \
        #     'Eval does not have runtime'

        # try:
        #     evaluation_result = self._evaluate(configuration, cache, dump2disk)
        # except:
        #     traceback.print_exc()
        #     return None
        # configuration.data.update(new_data)


    def _evaluate(self,
                  configuration: ConfigurationPoint,
                  cache: PrimitivesCache,
                  dump2disk: bool = True) -> typing.Dict:

        start_time = time.time()
        pipeline = self.template.to_pipeline(configuration)
        # Todo: update ResourceManager to run pipeline:  ResourceManager.add_pipeline(pipeline)

        # if in cross validation mode
        if self.testing_mode == Mode.CROSS_VALIDATION_MODE:
            repeat_times = int(self.validation_config['cross_validation'])
            # start training and testing
            fitted_pipeline = FittedPipeline(
                pipeline=pipeline,
                dataset_id=self.train_dataset1.metadata.query(())['id'],
                log_dir=self.log_dir,
                metric_descriptions=self.performance_metrics,
                template=self.template, problem=self.problem)

            fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset1])

            training_ground_truth = get_target_columns(self.train_dataset1, self.problem)
            training_prediction = fitted_pipeline.get_fit_step_output(
                self.template.get_output_step_number())
            training_metrics = calculate_score(training_ground_truth, training_prediction,
                self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

            cv_metrics = fitted_pipeline.get_cross_validation_metrics()
            test_metrics = copy.deepcopy(training_metrics)

            # use cross validation's avg value as the test score
            for i in range(len(test_metrics)):
                test_metrics[i]["value"] = cv_metrics[i]["value"]

            _logger.info("CV finish")

        # if in normal testing mode(including default testing mode with train/test one time each)
        else:
            if self.testing_mode == Mode.TRAIN_TEST_MODE:
                repeat_times = int(self.validation_config['test_validation'])
            else:
                repeat_times = 1

            _logger.info("Will use normal train-test mode ( n ={}) to choose best primitives.".format(repeat_times))

            training_metrics = []
            test_metrics = []

            for each_repeat in range(repeat_times):
                # start training and testing
                fitted_pipeline = FittedPipeline(
                    pipeline=pipeline,
                    dataset_id=self.train_dataset2[each_repeat].metadata.query(())['id'],
                    log_dir=self.log_dir,
                    metric_descriptions=self.performance_metrics,
                    template=self.template, problem=self.problem)

                fitted_pipeline.fit(cache=cache, inputs=[self.train_dataset2[each_repeat]])
                # fitted_pipeline.fit(inputs=[self.train_dataset2[each_repeat]])
                training_ground_truth = get_target_columns(self.train_dataset2[each_repeat],
                                                           self.problem)
                training_prediction = fitted_pipeline.get_fit_step_output(
                    self.template.get_output_step_number())
                training_metrics_each = calculate_score(training_ground_truth, training_prediction,
                self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

                # only do test if the test_dataset exist
                if self.test_dataset2[each_repeat] is not None:
                    results = fitted_pipeline.produce(inputs=[self.test_dataset2[each_repeat]])
                    test_ground_truth = get_target_columns(self.test_dataset2[each_repeat],
                                                           self.problem)
                    # Note: results == test_prediction
                    test_prediction = fitted_pipeline.get_produce_step_output(
                        self.template.get_output_step_number())
                    test_metrics_each = calculate_score(test_ground_truth, test_prediction,
                        self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
                else:
                    test_ground_truth = None
                    test_prediction = None
                    test_metrics_each = copy.deepcopy(training_metrics_each)
                    if larger_is_better(training_metrics_each):
                        for each in test_metrics_each:
                            each["value"] = 0
                    else:
                        for each in test_metrics_each:
                            each["value"] = sys.float_info.max

                training_metrics.append(training_metrics_each)
                test_metrics.append(test_metrics_each)
            # END for TRAIN_TEST_MODES
            # sample format of the output
            # [{'metric': 'f1Macro', 'value': 0.48418535913661614, 'values': [0.4841025641025641,
            #  0.4841025641025641, 0.4843509492047203]]
            # modify the test_metrics and training_metrics format to fit the requirements
            # print("[INFO] Testing finish.!!!")

            if len(training_metrics) > 1:
                training_value_dict = {}
                # convert for training matrics
                for each in training_metrics:
                    # for condition only one exist?
                    if type(each) is dict:
                        if each['column_name'] in training_value_dict:
                            # if this key exist, we append it
                            training_value_dict[each['column_name']].append(each['value'])
                        else:
                            # otherwise create a new key-value pair
                            training_value_dict[each['column_name']] = [each['value']]
                    else:
                        for each_target in each:
                            if each_target['column_name'] in training_value_dict:
                                training_value_dict[each_target['column_name']].append(
                                    each_target['value'])
                            else:
                                training_value_dict[each_target['column_name']] = [
                                    each_target['value']]
                # training_metrics part

                training_metrics_new = training_metrics[0]
                count = 0
                for (k, v) in training_value_dict.items():
                    training_metrics_new[count]['value'] = sum(v) / len(v)
                    training_metrics_new[count]['values'] = v
                    count += 1
                training_metrics = [training_metrics_new]

            else:
                if type(training_metrics[0]) is list:
                    training_metrics = training_metrics[0]

            if len(test_metrics) > 1:
                test_value_dict = {}
                # convert for test matrics
                for each in test_metrics:
                    # for condition only one exist?
                    if type(each) is dict:
                        if each['column_name'] in test_value_dict:
                            # if this key exist, we append it
                            test_value_dict[each['column_name']].append(each['value'])
                        else:
                            # otherwise create a new key-value pair
                            test_value_dict[each['column_name']] = [each['value']]
                    else:
                        for each_target in each:
                            if each_target['column_name'] in test_value_dict:
                                test_value_dict[each_target['column_name']].append(
                                    each_target['value'])
                            else:
                                test_value_dict[each_target['column_name']] = [each_target['value']]

                # test_metrics part
                test_metrics_new = test_metrics[0]
                count = 0
                for (k, v) in test_value_dict.items():
                    test_metrics_new[count]['value'] = sum(v) / len(v)
                    test_metrics_new[count]['values'] = v
                    count += 1
                test_metrics = [test_metrics_new]

            else:
                if type(test_metrics[0]) is list:
                    test_metrics = test_metrics[0]
        # END evaluation part

        # Save results
        if self.test_dataset1 is None:
            # print("The dataset no need to split of split failed, will not train again.")
            fitted_pipeline2 = fitted_pipeline
            # set the metric for calculating the rank
            fitted_pipeline2.set_metric(training_metrics[0])
            ensemble_tuning_result = None
            ensemble_tuning_metrics = None
            cv = fitted_pipeline2.get_cross_validation_metrics()
            if not cv:
                cv = {}

            data = {
                'id': fitted_pipeline2.id,
                'fitted_pipeline': fitted_pipeline2,
                'training_metrics': training_metrics,
                'cross_validation_metrics': cv,
                'test_metrics': training_metrics,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'ensemble_tuning_result': ensemble_tuning_result,
                'ensemble_tuning_metrics': ensemble_tuning_metrics,
            }
            fitted_pipeline.auxiliary = dict(data)

            # print("!!!! No test_dataset1")
            # pprint(data)
            # print("!!!!")

            if _logger.getEffectiveLevel() <= 10:
                data_to_logger_info = []
                if 'metric' in data['test_metrics']:
                    data_to_logger_info.append(data['test_metrics']['metric'])
                else:
                    data_to_logger_info.append("No test metrics metric found")
                if 'value' in data['test_metrics']:
                    data_to_logger_info.append(data['test_metrics']['value'])
                else:
                    data_to_logger_info.append("No test metrics value found")
                _logger.info(
                    'fitted id: %(fitted_pipeline_id)s, metric: %(metric)s, value: %(value)s',
                    {
                        'fitted_pipeline_id': fitted_pipeline2.id,
                        'metric': data_to_logger_info[0],
                        'value': data_to_logger_info[1]
                    })

            # Save fitted pipeline
            if self.output_directory is not None and dump2disk:
                fitted_pipeline2.save(self.output_directory)

            # Pickle test
            if self.output_directory is not None and dump2disk:
                _logger.debug("Test pickled pipeline. id: {}".format(fitted_pipeline2.id))
                self.test_pickled_pipeline(folder_loc=self.output_directory,
                                           pipeline_id=fitted_pipeline2.id,
                                           test_dataset=self.train_dataset2[0],
                                           test_metrics=training_metrics,
                                           test_ground_truth=get_target_columns(
                                               self.train_dataset2[0], self.problem))
        else:
            if self.quick_mode:
                _logger.info("[INFO] Now in quick mode, will skip training with train_dataset1")
                # if in quick mode, we did not fit the model with dataset_train1 again
                # just generate the predictions on dataset_test1 directly and get the rank
                fitted_pipeline2 = fitted_pipeline
            else:
                _logger.info("[INFO] Now in normal mode, will add extra train with train_dataset1")
                # otherwise train again with dataset_train1 and get the rank
                fitted_pipeline2 = FittedPipeline(
                    pipeline=pipeline,
                    dataset_id=self.train_dataset1.metadata.query(())['id'],
                    log_dir=self.log_dir,
                    metric_descriptions=self.performance_metrics,
                    template=self.template, problem=self.problem)
                # retrain and compute ranking/metric using self.train_dataset
                # fitted_pipeline2.fit(inputs = [self.train_dataset1])
                fitted_pipeline2.fit(cache=cache, inputs=[self.train_dataset1])

            fitted_pipeline2.produce(inputs=[self.test_dataset1])
            test_ground_truth = get_target_columns(self.test_dataset1, self.problem)
            test_prediction = fitted_pipeline2.get_produce_step_output(
                self.template.get_output_step_number())
            test_metrics2 = calculate_score(test_ground_truth, test_prediction,
                self.performance_metrics, self.task_type, SpecialMetric().regression_metric)
            # update here:
            # Now new version of d3m runtime don't allow to run ".fit()" again on a given runtime
            #  object second time
            # So here we need to create a new FittedPipeline object to run second time's
            # runtime.fit()
            fitted_pipeline_final = FittedPipeline(
                pipeline=pipeline,
                dataset_id=self.all_dataset.metadata.query(())['id'],
                log_dir=self.log_dir,
                metric_descriptions=self.performance_metrics,
                template=self.template, problem=self.problem)
            # set the metric for calculating the rank
            fitted_pipeline_final.set_metric(test_metrics2[0])

            # finally, fit the model with all data and save it
            _logger.info(
                "[INFO] Now are training the pipeline with all dataset and saving the pipeline.")
            fitted_pipeline_final.fit(cache=cache, inputs=[self.all_dataset])

            if self.ensemble_tuning_dataset:
                fitted_pipeline_final.produce(inputs=[self.ensemble_tuning_dataset])
            ensemble_tuning_result = fitted_pipeline_final.get_produce_step_output(self.template.get_output_step_number())
            ensemble_tuning_result_ground_truth = get_target_columns(self.ensemble_tuning_dataset, self.problem)
            ensemble_tuning_metrics = calculate_score(ensemble_tuning_result_ground_truth, ensemble_tuning_result,
                self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

            cv = fitted_pipeline_final.get_cross_validation_metrics()
            if not cv:
                cv = {}
            data = {
                'id': fitted_pipeline_final.id,
                'fitted_pipeline': fitted_pipeline_final,
                'training_metrics': training_metrics,
                'cross_validation_metrics': cv,
                'test_metrics': test_metrics2,
                'total_runtime': time.time() - start_time,
                'configuration': configuration,
                'ensemble_tuning_result': ensemble_tuning_result,
                'ensemble_tuning_metrics': ensemble_tuning_metrics,
            }
            fitted_pipeline.auxiliary = dict(data)

            # Save fiteed pipeline
            if self.output_directory is not None and dump2disk:
                fitted_pipeline_final.save(self.output_directory)

            # Pickle test
            if self.output_directory is not None and dump2disk:
                _ = fitted_pipeline_final.produce(inputs=[self.test_dataset1])
                test_prediction3 = fitted_pipeline_final.get_produce_step_output(
                    self.template.get_output_step_number())

                test_metrics3 = calculate_score(test_ground_truth, test_prediction3,
                    self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

                _logger.info("Test pickled pipeline. id: {}".format(fitted_pipeline_final.id))
                self.test_pickled_pipeline(folder_loc=self.output_directory,
                                           pipeline_id=fitted_pipeline_final.id,
                                           test_dataset=self.test_dataset1,
                                           test_metrics=test_metrics3,
                                           test_ground_truth=test_ground_truth)

        # still return the original fitted_pipeline with relation to train_dataset1
        return data

    def test_pickled_pipeline(self, folder_loc: str, pipeline_id: str, test_dataset: Dataset,
                              test_metrics: typing.List, test_ground_truth) -> None:

        fitted_pipeline, run = FittedPipeline.load(folder_loc=folder_loc, pipeline_id=pipeline_id,
                                                   log_dir=self.log_dir)
        results = fitted_pipeline.produce(inputs=[test_dataset])

        pipeline_prediction = fitted_pipeline.get_produce_step_output(
            self.template.get_output_step_number())
        pipeline_prediction = graph_problem_conversion(self.task_type, pipeline_prediction)

        test_pipeline_metrics2 = calculate_score(test_ground_truth, pipeline_prediction,
            self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

        test_pipeline_metrics = list()
        for metric_description in self.performance_metrics:
            metricDesc = PerformanceMetric.parse(metric_description['metric'])
            metric: typing.Callable = metricDesc.get_function()
            params: typing.Dict = metric_description['params']

            try:
                if metric_description["metric"] in SpecialMetric().regression_metric:
                    # if the test_ground_truth do not have results
                    if test_ground_truth.iloc[0, -1] == '':
                        test_ground_truth.iloc[:, -1] = 0
                    test_pipeline_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(
                            test_ground_truth.iloc[:, -1].astype(float),
                            pipeline_prediction.iloc[:, -1].astype(float),
                            **params
                        )
                    })
                else:
                    test_pipeline_metrics.append({
                        'metric': metric_description['metric'],
                        'value': metric(
                            test_ground_truth.iloc[:, -1].astype(str),
                            pipeline_prediction.iloc[:, -1].astype(str),
                            **params
                        )
                    })
            except:
                raise NotSupportedError(
                    '[ERROR] metric calculation failed in test pickled pipeline')

        _logger.info(f'=== original:{test_metrics}')
        _logger.info(f'=== test:{test_pipeline_metrics}')
        _logger.info(f'=== test2:{test_pipeline_metrics2}')

        pairs = zip(test_metrics, test_pipeline_metrics2)
        if any(x != y for x, y in pairs):
            warn("[WARN] Test pickled pipeline mismatch. id: {}".format(fitted_pipeline.id))
            print(
                {
                    'id': fitted_pipeline.id,
                    'test__metric': test_metrics,
                    'pickled_pipeline__metric': test_pipeline_metrics
                }
            )
            print("\n" * 5)
            _logger.warning(
                "Test pickled pipeline mismatch. 'id': '%(id)s', 'test__metric': '%("
                "test__metric)s', 'pickled_pipeline__metric': '%(pickled_pipeline__metric)s'.",
                {
                    'id': fitted_pipeline.id,
                    'test__metric': test_metrics,
                    'pickled_pipeline__metric': test_pipeline_metrics
                },
            )
            print(
                "Test pickled pipeline mismatch. 'id': '%(id)s', 'test__metric': '%("
                "test__metric)s', 'pickled_pipeline__metric': '%("
                "pickled_pipeline__metric)s'.".format(
                    {
                        'id': fitted_pipeline.id,
                        'test__metric': test_metrics,
                        'pickled_pipeline__metric': test_pipeline_metrics
                    })
            )
            print("\n" * 5)
        else:
            _logger.debug(("\n" * 5) + "Pickling succeeded" + ("\n" * 5))

def graph_problem_conversion(task_type, prediction):
    """
        Inner function used to process with graph type tasks
    """
    if isinstance(task_type, set):
        for t in task_type:
            if t == "GRAPH_MATCHING" or t == "VERTEX_NOMINATION" or t == "LINK_PREDICTION":
                prediction.iloc[:, -1] = prediction.iloc[:, -1].astype(int)
    else:
        if task_type == "GRAPH_MATCHING" or task_type == "VERTEX_NOMINATION" or task_type == \
                "LINK_PREDICTION":
            prediction.iloc[:, -1] = prediction.iloc[:, -1].astype(int)
    return prediction

def calculate_score(ground_truth:DataFrame, prediction:DataFrame,
                    performance_metrics:typing.List[typing.Dict],
                    task_type, regression_metric:set()):
    """
    static method used to calculate the score based on given predictions and metric tpyes
    Parameters
    ---------
    ground_truth: the ground truth of target
    prediction: the predicted results of target
    performance_metrics: the metehod to calculate the score
    task_type: the task type of the problem
    """
    result_metrics = []
    target_amount = 0
    if prediction is not None:
        prediction = graph_problem_conversion(task_type, prediction)

    for metric_description in performance_metrics:
        metricDesc = PerformanceMetric.parse(metric_description['metric'])
        metric: typing.Callable = metricDesc.get_function()
        params: typing.Dict = metric_description['params']

        # special design for objectDetectionAP
        if metric_description["metric"] == "objectDetectionAP":
            if ground_truth is not None and prediction is not None:
                training_image_name_column = ground_truth.iloc[:,
                                             ground_truth.shape[1] - 2]
                prediction.insert(loc=0, column='image_name',
                                           value=training_image_name_column)
                ground_truth_tosend = ground_truth.iloc[:,
                                               ground_truth.shape[1] - 2:
                                               ground_truth.shape[1]]
                result_metrics.append({
                    'column_name': ground_truth.columns[-1],
                    'metric': metric_description['metric'],
                    'value': metric(
                        ground_truth_tosend.astype(str).values.tolist(),
                        prediction.astype(str).values.tolist(),
                        **params
                    )
                })
            return result_metrics
        # END special design for objectDetectionAP

        do_regression_mode = metric_description["metric"] in regression_metric
        try:
            # generate the metrics for training results
            if ground_truth is not None and prediction is not None:  # if
                # training data exist
                if "d3mIndex" not in prediction.columns:
                    # for the condition that ground_truth have index but
                    # prediction don't have
                    target_amount = len(prediction.columns)
                else:
                    target_amount = len(prediction.columns) - 1

                ground_truth_amount = len(ground_truth.columns) - 1

                if not (ground_truth_amount == target_amount):
                    print('predicition columns :', prediction.columns)
                    print('Ground truth columns:', ground_truth.columns)
                #     from runtime import ForkedPdb
                #     ForkedPdb().set_trace()

                assert ground_truth_amount == target_amount, \
                    "[ERROR] Truth and prediction does not match"

                if do_regression_mode:
                    # regression mode require the targets must be float
                    for each_column in range(-target_amount, 0, 1):
                        result_metrics.append({
                            'column_name': ground_truth.columns[each_column],
                            'metric': metric_description['metric'],
                            'value': metric(
                                ground_truth.iloc[:, each_column].astype(float),
                                prediction.iloc[:, each_column].astype(float),
                                **params
                            )
                        })
                else:
                    # for other modes, we can treat the predictions as str
                    for each_column in range(- target_amount, 0, 1):
                        result_metrics.append({
                            'column_name': ground_truth.columns[each_column],
                            'metric': metric_description['metric'],
                            'value': metric(
                                ground_truth.iloc[:, each_column].astype(str),
                                prediction.iloc[:, each_column].astype(str),
                                **params
                            )
                        })
        except:
            raise NotSupportedError('[ERROR] metric calculation failed')
    # END for loop

    if len(result_metrics) > target_amount:
        _logger.warning("[WARN] Training metrics's amount is larger than target amount.")

    # return the training and test metrics
    return result_metrics
