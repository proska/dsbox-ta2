import typing
import logging
import traceback

import d3m.runtime as d3m_runtime
import d3m.metadata.base as metadata_base
import d3m.container as container
import d3m.metadata.pipeline_run as pipeline_run_module
import d3m.metadata.problem as problem

from d3m.exceptions import NotSupportedError
from d3m.metadata import pipeline as pipeline_module
from d3m.utils import AbstractMetaclass

_logger = logging.getLogger(__name__)

def score_prediction(
        predictions: container.DataFrame,
        score_inputs: typing.Sequence[container.Dataset],
        problem_description: typing.Optional[problem.Problem],
        metrics: typing.Sequence[typing.Dict],
        predictions_random_seed: int = None,
        *,
        scoring_params: typing.Dict[str, str] = None, random_seed: int = 0, volumes_dir: str = None,
        scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> container.DataFrame:
    df = score_prediction_dataframe(
        predictions=predictions, score_inputs=score_inputs, problem_description=problem_description, metrics=metrics,
        predictions_random_seed=predictions_random_seed, scoring_params=scoring_params, random_seed=random_seed,
        volumes_dir=volumes_dir, scratch_dir=scratch_dir, runtime_environment=runtime_environment)
    results = []
    if df is not None:
        for index, row in df.iterrows():
            results.append({
                'metric': problem.PerformanceMetric[row['metric']],
                'value': row['value'],
                'normalized': row['normalized'],
                'random_seed': row['randomSeed'],
                'column_name': 'TO_DO'
                })
    return results

def score_prediction_dataframe(
        predictions: container.DataFrame,
        score_inputs: typing.Sequence[container.Dataset],
        problem_description: typing.Optional[problem.Problem],
        metrics: typing.Sequence[typing.Dict],
        predictions_random_seed: int = None,
        *,
        scoring_params: typing.Dict[str, str] = None, random_seed: int = 0, volumes_dir: str = None,
        scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> container.DataFrame:
    '''
    Returns score of the predictions. The returned dataframe has four columns: metric, value, normalized, and randomSeed.
    The value of the randomSeed is just the input `predictions_random_seed`.

    metric,value,normalized,randomSeed
    F1,0.878048780487805,0.878048780487805,0
    '''
    pipeline_resolver = pipeline_module.get_pipeline
    scoring_pipeline: pipeline_module.Pipeline = pipeline_resolver(
        d3m_runtime.DEFAULT_SCORING_PIPELINE_PATH,
        strict_resolving=False,
        strict_digest=False,
        pipeline_search_paths=[],
    )
    context: metadata_base.Context = metadata_base.Context.TESTING
    score, result = d3m_runtime.score(
        scoring_pipeline=scoring_pipeline, problem_description=problem_description, predictions=predictions,
        score_inputs=score_inputs, metrics=metrics, predictions_random_seed=predictions_random_seed,
        context=context, scoring_params=scoring_params, random_seed=random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, runtime_environment=runtime_environment)
    if score is None:
        _logger.exception(result.error, exc_info=True)
        raise ValueError('Metric missing') from result.error
    return score


def calculate_score(ground_truth: container.DataFrame, prediction: container.DataFrame,
                    performance_metrics: typing.List[typing.Dict],
                    task_type, regression_metric: set):
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
        metricDesc = metric_description['metric']
        params: typing.Dict = metric_description.get('params', {})
        if params:
            metric: problem.PerformanceMetric = metricDesc.get_class()(**params)
        else:
            metric = metricDesc.get_class()
        # updated for d3m v2019.5.8: we need to instantiate the metric class first if it was not done yet
        if type(metric) is AbstractMetaclass:
            metric = metric()

        # special design for objectDetectionAP
        if metric_description["metric"] == problem.PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION:

            if ground_truth is not None and prediction is not None:
                # training_image_name_column = ground_truth.iloc[:,
                #                              ground_truth.shape[1] - 2]
                # prediction.insert(loc=0, column='image_name',
                #                            value=training_image_name_column)

                ground_truth_to_send = ground_truth.iloc[:, ground_truth.shape[1] - 2: ground_truth.shape[1]]
                prediction_to_send = prediction# .iloc[:, prediction.shape[1] - 2: prediction.shape[1]]
                if prediction_to_send['d3mIndex'].dtype.name != ground_truth_to_send['d3mIndex'].dtype.name:
                    ground_truth_to_send = ground_truth_to_send['d3mIndex'].astype(str)
                    prediction_to_send = prediction_to_send['d3mIndex'].astype(str)

                # truth = ground_truth_to_send.astype(str).values.tolist()
                # predictions = prediction_to_send.astype(str).values.tolist()
                value = metric.score(ground_truth_to_send, prediction_to_send)

                result_metrics.append({
                    'column_name': ground_truth.columns[-1],
                    'metric': metric_description['metric'],
                    'value': value
                })
            return result_metrics
        # END special design for objectDetectionAP

        do_regression_mode = metric_description["metric"] in regression_metric
        try:
            # generate the metrics for training results
            if ground_truth is not None and prediction is not None:
                if "d3mIndex" not in ground_truth.columns:
                    raise NotSupportedError("No d3mIndex found for ground truth!")
                else:
                    ground_truth_amount = len(ground_truth.columns) - 1

                if "d3mIndex" not in prediction.columns:
                    # for the condition that ground_truth have index but
                    # prediction don't have
                    target_amount = len(prediction.columns)
                    prediction.insert(0,'d3mIndex' ,ground_truth['d3mIndex'].copy())
                else:
                    target_amount = len(prediction.columns) - 1

                if prediction['d3mIndex'].dtype.name != ground_truth['d3mIndex'].dtype.name:
                    ground_truth['d3mIndex'] = ground_truth['d3mIndex'].astype(str).copy()
                    prediction['d3mIndex'] = prediction['d3mIndex'].astype(str).copy()

                if not (ground_truth_amount == target_amount):
                    _logger.error("Ground truth's amount and prediction's amount does not match")
                    _logger.error('predicition columns :' + str(prediction.columns))
                    _logger.error('Ground truth columns:' + str(ground_truth.columns))
                    raise ValueError("Ground truth's amount and prediction's amount does not match")
                #     from runtime import ForkedPdb
                #     ForkedPdb().set_trace()

                if do_regression_mode:
                    # regression mode require the targets must be float
                    for each_column in range(-target_amount, 0, 1):
                        prediction.iloc[:,each_column] = prediction.iloc[:,each_column].astype(float).copy()

                # update 2019.4.12, now d3m v2019.4.4 have new metric function, we have to change like this
                ground_truth_d3m_index_column_index = ground_truth.columns.tolist().index("d3mIndex")
                prediction_d3m_index_column_index = prediction.columns.tolist().index("d3mIndex")

                for each_column in range(-target_amount, 0, 1):
                    result_metrics.append({
                        'column_name': ground_truth.columns[each_column],
                        'metric': metric_description['metric'],
                        'value': metric.score(truth=ground_truth.iloc[:,[ground_truth_d3m_index_column_index,each_column]],
                                              predictions=prediction.iloc[:,[prediction_d3m_index_column_index,each_column]])
                    })
            elif ground_truth is None:
                raise NotSupportedError("Metric calculation failed because ground truth is None!")
            elif prediction is not None:
                raise NotSupportedError("Metric calculation failed because prediction is None!")

        except Exception:
            traceback.print_exc()
            raise NotSupportedError('[ERROR] metric calculation failed')
    # END for loop

    if len(result_metrics) > target_amount:
        _logger.warning("[WARN] Training metrics's amount is larger than target amount.")

    # return the training and test metrics
    return result_metrics


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


SEMANTIC_TYPES = {
    # Resource types (files collections).
    'image': 'http://schema.org/ImageObject',
    'video': 'http://schema.org/VideoObject',
    'audio': 'http://schema.org/AudioObject',
    'text': 'http://schema.org/Text',
    'speech': 'https://metadata.datadrivendiscovery.org/types/Speech',
    'raw': 'https://metadata.datadrivendiscovery.org/types/UnspecifiedStructure',
    # Resource types (other)
    'graph': 'https://metadata.datadrivendiscovery.org/types/Graph',
    'edgeList': 'https://metadata.datadrivendiscovery.org/types/EdgeList',
    'table': 'https://metadata.datadrivendiscovery.org/types/Table',
    'timeseries': 'https://metadata.datadrivendiscovery.org/types/Timeseries',
    # Column types.
    'boolean': 'http://schema.org/Boolean',
    'integer': 'http://schema.org/Integer',
    'real': 'http://schema.org/Float',
    'string': 'http://schema.org/Text',
    'categorical': 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
    'dateTime': 'http://schema.org/DateTime',
    'realVector': 'https://metadata.datadrivendiscovery.org/types/FloatVector',
    'json': 'https://metadata.datadrivendiscovery.org/types/JSON',
    'geojson': 'https://metadata.datadrivendiscovery.org/types/GeoJSON',
    'unknown': 'https://metadata.datadrivendiscovery.org/types/UnknownType',
    # Column roles.
    'index': 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
    'multiIndex': 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
    'key': 'https://metadata.datadrivendiscovery.org/types/UniqueKey',
    'attribute': 'https://metadata.datadrivendiscovery.org/types/Attribute',
    'suggestedTarget': 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
    'suggestedPrivilegedData': 'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData',
    'timeIndicator': 'https://metadata.datadrivendiscovery.org/types/Time',
    'locationIndicator': 'https://metadata.datadrivendiscovery.org/types/Location',
    'boundaryIndicator': 'https://metadata.datadrivendiscovery.org/types/Boundary',
    'interval': 'https://metadata.datadrivendiscovery.org/types/Interval',
    'instanceWeight': 'https://metadata.datadrivendiscovery.org/types/InstanceWeight',
    'boundingPolygon': 'https://metadata.datadrivendiscovery.org/types/BoundingPolygon',
}


'''
def calculate_score(ground_truth: DataFrame, prediction: DataFrame,
                    performance_metrics: typing.List[typing.Dict],
                    task_type, regression_metric: set()):
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
        metricDesc = problem.PerformanceMetric.parse(metric_description['metric'])
        params: typing.Dict = metric_description['params']
        metric: typing.Callable = metricDesc.get_class()(**params)

        # special design for objectDetectionAP
        if metric_description["metric"] == "objectDetectionAP":
            if ground_truth is not None and prediction is not None:
                # training_image_name_column = ground_truth.iloc[:,
                #                              ground_truth.shape[1] - 2]
                # prediction.insert(loc=0, column='image_name',
                #                            value=training_image_name_column)
                ground_truth_to_send = ground_truth.iloc[:, ground_truth.shape[1] - 2: ground_truth.shape[1]]
                prediction_to_send = prediction.iloc[:, prediction.shape[1] - 2: prediction.shape[1]]
                if prediction_to_send['d3mIndex'].dtype.name != ground_truth_to_send['d3mIndex'].dtype.name:
                    ground_truth_to_send = ground_truth_to_send['d3mIndex'].astype(str)
                    prediction_to_send = prediction_to_send['d3mIndex'].astype(str)

                # truth = ground_truth_to_send.astype(str).values.tolist()
                # predictions = prediction_to_send.astype(str).values.tolist()
                value = metric.score(ground_truth_to_send, prediction_to_send)

                result_metrics.append({
                    'column_name': ground_truth.columns[-1],
                    'metric': metric_description['metric'],
                    'value': value
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
                    # TODO: we also need to add d3mIndex if prediction don't have here

                else:
                    target_amount = len(prediction.columns) - 1
                    if prediction['d3mIndex'].dtype.name != ground_truth['d3mIndex'].dtype.name:
                        ground_truth.loc[:, 'd3mIndex'] = ground_truth['d3mIndex'].astype(str)
                        prediction.loc[:, 'd3mIndex'] = prediction['d3mIndex'].astype(str)

                ground_truth_amount = len(ground_truth.columns) - 1

                if not (ground_truth_amount == target_amount):
                    print('predicition columns :', prediction.columns)
                    print('Ground truth columns:', ground_truth.columns)
                    raise ValueError("Ground truth's amount and prediction's amount does not match")
                #     from runtime import ForkedPdb
                #     ForkedPdb().set_trace()

                if do_regression_mode:
                    # regression mode require the targets must be float
                    for each_column in range(-target_amount, 0, 1):
                        prediction.iloc[:,each_column] = prediction.iloc[:,each_column].astype(float)
                else:
                    for each_column in range(-target_amount, 0, 1):
                        prediction.iloc[:,each_column] = prediction.iloc[:,each_column].astype(str)
                # update 2019.4.12, now d3m v2019.4.4 have new metric function, we have to change like this
                ground_truth_d3m_index_column_index = ground_truth.columns.tolist().index("d3mIndex")
                prediction_d3m_index_column_index = prediction.columns.tolist().index("d3mIndex")
                for each_column in range(-target_amount, 0, 1):
                    result_metrics.append({
                        'column_name': ground_truth.columns[each_column],
                        'metric': metric_description['metric'],
                        'value': metric.score(ground_truth.iloc[:,[ground_truth_d3m_index_column_index,each_column]],
                                              prediction.iloc[:,[prediction_d3m_index_column_index,each_column]])
                    })
        except Exception:
            raise ValueError('[ERROR] metric calculation failed')
    # END for loop

    if len(result_metrics) > target_amount:
        _logger.warning("[WARN] Training metrics's amount is larger than target amount.")

    # return the training and test metrics
    return result_metrics

'''
