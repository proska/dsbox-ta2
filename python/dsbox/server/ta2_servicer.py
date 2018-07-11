import collections
import os
import sys
import typing

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ta3ta2_api = os.path.abspath(os.path.join(
    CURRENT_DIR, '..', '..', '..', '..', 'ta3ta2-api'))
print(ta3ta2_api)
sys.path.append(ta3ta2_api)


import d3m
import d3m.metadata.problem as d3m_problem

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, SubpipelineStep, ArgumentType
from d3m.primitive_interfaces.base import PrimitiveBase

import logging
import random
import string

import core_pb2
import core_pb2_grpc
import problem_pb2
import value_pb2
import pipeline_pb2

from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore
from pprint import pprint

# import autoflowconfig
from core_pb2 import DescribeSolutionResponse
from core_pb2 import EndSearchSolutionsResponse
from core_pb2 import EvaluationMethod
from core_pb2 import GetScoreSolutionResultsResponse
from core_pb2 import GetSearchSolutionsResultsResponse
from core_pb2 import HelloResponse
from core_pb2 import ListPrimitivesResponse
from core_pb2 import PrimitiveStepDescription
from core_pb2 import Progress
from core_pb2 import ProgressState
from core_pb2 import Score
from core_pb2 import ScoreSolutionResponse
from core_pb2 import ScoringConfiguration
from core_pb2 import SearchSolutionsResponse
from core_pb2 import SolutionSearchScore
from core_pb2 import StepDescription
from core_pb2 import SubpipelineStepDescription

from pipeline_pb2 import PipelineDescription
from pipeline_pb2 import PipelineDescriptionInput
from pipeline_pb2 import PipelineDescriptionOutput
from pipeline_pb2 import PipelineDescriptionStep
from pipeline_pb2 import PipelineDescriptionUser
from pipeline_pb2 import PrimitivePipelineDescriptionStep
from pipeline_pb2 import PrimitiveStepArgument
from pipeline_pb2 import PrimitiveStepHyperparameter
from pipeline_pb2 import StepOutput
from pipeline_pb2 import ContainerArgument
from pipeline_pb2 import DataArgument
from pipeline_pb2 import PrimitiveArgument
from pipeline_pb2 import ValueArgument
from pipeline_pb2 import PrimitiveArguments

from problem_pb2 import ProblemPerformanceMetric
from problem_pb2 import PerformanceMetric
from problem_pb2 import ProblemTarget

from primitive_pb2 import Primitive

from value_pb2 import Value
from value_pb2 import ValueRaw
from value_pb2 import ValueList
from value_pb2 import ValueDict

# from value_pb2 import Value
# from value_pb2 import ValueError
# from value_pb2 import DoubleList
# from value_pb2 import Int64List
# from value_pb2 import BoolList
# from value_pb2 import StringList
# from value_pb2 import BytesList

from dsbox.controller.controller import Controller
from dsbox.pipeline.fitted_pipeline import FittedPipeline

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s -- %(message)s')
_logger = logging.getLogger(__name__)

# problem.proto and d3m.metadata.problem have different schemes for metrics
# Mapping needed for v2018.4.18, but not for later versions
# pb2_to_d3m_metric = {
#     0 : None,  # METRIC_UNDEFINED
#     1 : d3m_problem.PerformanceMetric.ACCURACY,
#     2 : None,  # PRECISION
#     3 : None,  # RECALL
#     4 : d3m_problem.PerformanceMetric.F1,
#     5 : d3m_problem.PerformanceMetric.F1_MICRO,
#     6 : d3m_problem.PerformanceMetric.F1_MACRO,
#     7 : d3m_problem.PerformanceMetric.ROC_AUC,
#     8 : d3m_problem.PerformanceMetric.ROC_AUC_MICRO,
#     9 : d3m_problem.PerformanceMetric.ROC_AUC_MACRO,
#     10 : d3m_problem.PerformanceMetric.MEAN_SQUARED_ERROR,
#     11 : d3m_problem.PerformanceMetric.ROOT_MEAN_SQUARED_ERROR,
#     12 : d3m_problem.PerformanceMetric.ROOT_MEAN_SQUARED_ERROR_AVG,
#     13 : d3m_problem.PerformanceMetric.MEAN_ABSOLUTE_ERROR,
#     14 : d3m_problem.PerformanceMetric.R_SQUARED,
#     15 : d3m_problem.PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION,
#     16 : d3m_problem.PerformanceMetric.JACCARD_SIMILARITY_SCORE,
#     17 : d3m_problem.PerformanceMetric.PRECISION_AT_TOP_K,
# #    18 : d3m_problem.PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION
# }


# The output of this function should be the same as the output for
# d3m/metadata/problem.py:parse_problem_description

def problem_to_dict(problem) -> typing.Dict:
    performance_metrics = []
    for metrics in problem.problem.performance_metrics:
        if metrics.metric==0:
            d3m_metric = None
        else:
            d3m_metric = d3m_problem.PerformanceMetric(metrics.metric)
        params = {}
        if d3m_metric == d3m_problem.PerformanceMetric.F1:
            if metrics.pos_label is None or metrics.pos_label=='':
                params['pos_label'] = '1'
            else:
                params['pos_label'] = metrics.pos_label
        if metrics.k > 0:
            params['k'] = metrics.k
        performance_metrics.append ({
            'metric' : d3m_metric,
            'params' : params
        })

    description: typing.Dict[str, typing.ANY] = {
        'schema': d3m_problem.PROBLEM_SCHEMA_VERSION,
        'problem': {
            'id': problem.problem.id,
            # "problemVersion" is required by the schema, but we want to be compatible with problem
            # descriptions which do not adhere to the schema.
            'version': problem.problem.version,
            'name': problem.problem.name,
            'task_type': d3m_problem.TaskType(problem.problem.task_type),
            'task_subtype': d3m_problem.TaskSubtype(problem.problem.task_subtype),
            'performance_metrics': performance_metrics
        },
        # Not Needed
        # 'outputs': {
        #     'predictions_file': problem_doc['expectedOutputs']['predictionsFile'],
        # }
    }

    inputs = []
    for input in problem.inputs:
        dataset_id = input.dataset_id
        for target in input.targets:
            targets = []
            targets.append({
                'target_index': target.target_index,
                'resource_id': target.resource_id,
                'column_index': target.column_index,
                'column_name': target.column_name,
                'clusters_number': target.clusters_number
            })
        inputs.append({
            'dataset_id': dataset_id,
            'targets': targets
        })
    description['inputs'] = inputs

    return description

# The output of this function should be the same as the problemDoc.json
def problem_to_json(problem) -> typing.Dict:
    performance_metrics = []
    for metrics in problem.problem.performance_metrics:
        if metrics.metric==0:
            d3m_metric = None
        else:
            d3m_metric = d3m_problem.PerformanceMetric(metrics.metric)
        params = {}
        if d3m_metric == d3m_problem.PerformanceMetric.F1:
            if metrics.pos_label is None or metrics.pos_label=='':
                params['pos_label'] = '1'
            else:
                params['pos_label'] = metrics.pos_label
        if metrics.k > 0:
            params['k'] = metrics.k
        ametric = {
            'metric' : d3m_metric.unparse(),
        }
        if params:
            ametric['params'] = params
        performance_metrics.append (ametric)

    description: typing.Dict[str, typing.ANY] = {
        'about': {
            'problemSchemaVersion': problem.problem.version,
            'problemID': problem.problem.id,
            # "problemVersion" is required by the schema, but we want to be compatible with problem
            # descriptions which do not adhere to the schema.
            # 'version': problem.problem.version,
            'problemName': problem.problem.name,
            'taskType': d3m_problem.TaskType(problem.problem.task_type).unparse(),
            'taskSubtype': d3m_problem.TaskSubtype(problem.problem.task_subtype).unparse(),
        }
        # Not Needed
        # 'outputs': {
        #     'predictions_file': problem_doc['expectedOutputs']['predictionsFile'],
        # }
    }

    data = []
    for input in problem.inputs:
        dataset_id = input.dataset_id
        for target in input.targets:
            targets = []
            targets.append({
                'targetIndex': target.target_index,
                'resID': str(target.resource_id),
                'colIndex': target.column_index,
                'colName': target.column_name,
                'clustersNumber': target.clusters_number
            })
        data.append({
            'datasetID': dataset_id,
            'targets': targets
        })

    description['inputs'] = {
        'data': data,
        'performanceMetrics': performance_metrics
    }

    return description

def to_proto_value_raw(value):
    if value is None:
        return ValueRaw(NulValue=value_pb2.NULL_VALUE)
    elif isinstance(value, int):
        return ValueRaw(int64=value)
    elif isinstance(value, float):
        return ValueRaw(double=value)
    elif isinstance(value, bool):
        return ValueRaw(bool=value)
    elif isinstance(value, str):
        return ValueRaw(string=value)
    elif isinstance(value, bytes):
        return ValueRaw(bytes=value)
    elif isinstance(value, list) or isinstance(value, tuple):
        alist = []
        for x in value:
            alist.append(to_proto_value_raw(x))
        return ValueRaw(list=ValueList(items=alist))
    elif isinstance(value, dict):
        adict = {}
        for k, v in value.items():
            adict[k] = to_proto_value_raw(v)
        return ValueRaw(dict=ValueDict(items=adict))
    else:
        raise ValueError('to_proto_value: Unknown value type {}({})'.format(type(value), value))

# def to_proto_value(value):
#     is_list = isinstance(value, collections.Iterable)
#     if not is_list:
#         if isinstance(value, int):
#             return Value(int64=value)
#         elif isinstance(value, float):
#             return Value(double=value)
#         elif isinstance(value, bool):
#             return Value(bool=value)
#         elif isinstance(value, str):
#             return Value(string=value)
#         elif isinstance(value, bytes):
#             return Value(bytes=value)
#         else:
#             raise ValueError('to_proto_value: Unknown value type {}({})'.format(type(value), value))

#     if len(value) == 0:
#         # what would be an appropriate default for empty list?
#         return Value(string_list=StringList())

#     sample = value[0]
#     if isinstance(sample, int):
#         alist = Int64List()
#         for x in value:
#             alist.list.append(x)
#         proto_value = Value(int64_list=alist)
#     elif isinstance(sample, float):
#         alist = DoubleList()
#         for x in value:
#             alist.list.append(x)
#         proto_value = Value(double_list=alist)
#     elif isinstance(sample, bool):
#         alist = BoolList()
#         for x in value:
#             alist.list.append(x)
#         proto_value = Value(bool_list=alist)
#     elif isinstance(sample, str):
#         alist = StringList()
#         for x in value:
#             alist.list.append(x)
#         proto_value = Value(string_list=alist)
#     elif isinstance(sample, bytes):
#         alist = BytesList()
#         for x in value:
#             alist.list.append(x)
#         proto_value = Value(bytes_list=alist)
#     else:
#         raise ValueError('to_proto_value: Unknown value list type {}({})'.format(type(sample), sample))

#     return proto_value

# def to_proto_value_with_type(value, typing_instance):
#     if value is None:
#         types = list(typing_instance.__args__)
#         types.remove(type(None))
#         if int in types:
#             return Value(int64=value)
#         elif float in types:
#             return Value(double=value)
#         elif bool in types:
#             return Value(bool=value)
#         elif str in types:
#             return Value(string=value)
#         elif bytes in types:
#             return Value(bytes=value)
#         else:
#             raise ValueError('to_proto_value: Unknown value type {}({})'.format(type(value), value))
#     elif isinstance(value, collections.Iterable) and len(value)==0:
#         types = list(typing_instance.__args__)
#         types.remove(type(None))
#         if int in types:
#             return Value(int64_list=value)
#         elif float in types:
#             return Value(double_list=value)
#         elif bool in types:
#             return Value(bool_list=value)
#         elif str in types:
#             return Value(string_list=value)
#         elif bytes in types:
#             return Value(bytes_list=value)
#         else:
#             raise ValueError('to_proto_value: Unknown value type {}({})'.format(type(value), value))
#     else:
#         return to_proto_value(value)

def to_proto_primitive(primitive_base: PrimitiveBase) -> Primitive:
    """
    Convert d3m Primitive to protocol buffer Prmitive
    """
    metadata = primitive_base.metadata.query()
    return Primitive(
        id = metadata['id'],
        version = metadata['version'],
        python_path = metadata['python_path'],
        name = metadata['name'],
        digest = metadata['digest'] if 'digest' in metadata else None
    )

def to_proto_primitive_step(step : PrimitiveStep) -> PipelineDescriptionStep:
    """
    Convert d3m PrimitiveStep to protocol buffer PipelineDescriptionStep
    """
    arguments = {}
    for argument_name, argument_desc in step.arguments.items():
        if argument_desc['type']==ArgumentType.CONTAINER:
            # ArgumentType.CONTAINER
            arguments[argument_name] = PrimitiveStepArgument(
                container=ContainerArgument(data=argument_desc['data']))
        else:
            # ArgumentType.DATA
            arguments[argument_name] = PrimitiveStepArgument(
                data=DataArgument(data=argument_desc['data']))
    outputs = [StepOutput(id=output) for output in step.outputs]
    hyperparams = {}
    for name, hyperparam_dict in step.hyperparams.items():
        hyperparam_type = hyperparam_dict['type']
        hyperparam_data = hyperparam_dict['data']
        if hyperparam_type==ArgumentType.CONTAINER:
            hyperparam = PrimitiveStepHyperparameter(container=ContainerArgument(data=hyperparam_data))
        elif hyperparam_type==ArgumentType.DATA:
            hyperparam = PrimitiveStepHyperparameter(data=DataArgument(data=hyperparam_data))
        elif hyperparam_type==ArgumentType.PRIMITIVE:
            hyperparam = PrimitiveStepHyperparameter(primitive=PrimitiveArgument(data=hyperparam_data))
        elif hyperparam_type==ArgumentType.VALUE:
            hyperparam = PrimitiveStepHyperparameter(value=ValueArgument(data=Value(raw=to_proto_value_raw(hyperparam_data))))
        else:
            # Dataset is not a valid ArgumentType
            # Should never get here.
            raise ValueError('to_proto_primitive_step: invalid hyperparam type {}'.format(hyperparam_type))
        hyperparams[name] = hyperparam
    primitive_description = PrimitivePipelineDescriptionStep(
        primitive=to_proto_primitive(step.primitive),
        arguments=arguments,
        outputs=outputs,
        hyperparams=hyperparams,
        users=[PipelineDescriptionUser(id=user_description)
               for user_description in step.users]
    )
    return PipelineDescriptionStep(primitive=primitive_description)

def to_proto_pipeline(pipeline : Pipeline, id:str = None) -> PipelineDescription:
    """
    Convert d3m Pipeline to protocol buffer PipelineDescription
    """
    inputs = []
    outputs = []
    steps = []
    users = []
    for input_description in pipeline.inputs:
        if 'name' in input_description:
            inputs.append(PipelineDescriptionInput(name=input_description['name']))
    for output_description in pipeline.outputs:
        outputs.append(
            PipelineDescriptionOutput(
                name=output_description['name'] if 'name' in output_description else None,
                data=output_description['data']))
    for step in pipeline.steps:
        if isinstance(step, PrimitiveStep):
            step_description = to_proto_primitive_step(step)
        elif isinstance(step, SubpipelineStep):
            # TODO: Subpipeline not yet implemented
            # PipelineDescriptionStep(pipeline=pipeline_description)
            pass
        else:
            # TODO: PlaceholderStep not yet implemented
            #PipelineDescriptionStep(placeholder=placeholde_description)
            pass
        steps.append(step_description)
    for user in pipeline.users:
        users.append(PipelineDescriptionUser(
            id=user['id'],
            reason=user['reason'] if 'reason' in user else None,
            rationale=user['rationale'] if 'rationale' in user else None
        ))
    if id is None:
        id = pipeline.id,

    import pdb
    pdb.set_trace()

    pipeline_context = pipeline.context.value
    return PipelineDescription(
        id=id,
        source=pipeline.source,
        created=Timestamp().FromDatetime(pipeline.created.replace(tzinfo=None)),
        context=pipeline_context,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        name=pipeline.name,
        description=pipeline.description,
        users=users
    )

def to_proto_search_solution_requests(problem, fitted_pipeline_id, metrics_result) -> typing.List[GetSearchSolutionsResultsResponse]:

    search_solutions_results = []

    timestamp = Timestamp()

    # Todo: controller needs to remember the partition method
    scoring_config = ScoringConfiguration(
        method=core_pb2.HOLDOUT,
        train_test_ratio=5,
        random_seed=4676,
        stratified=True)
    targets = []
    problem_dict = problem
    for inputs_dict in problem_dict['inputs']:
        for target in inputs_dict['targets']:
            targets.append(ProblemTarget(
                target_index = target['target_index'],
                resource_id = target['resource_id'],
                column_index = target['column_index'],
                column_name = target['column_name'],
                clusters_number = target['clusters_number']))
    score_list = []
    for metric in metrics_result:
        ppm = ProblemPerformanceMetric(metric=d3m_problem.PerformanceMetric.parse(metric['metric']).name)
        if 'k' in metric:
            ppm = metric['k']
        if 'pos_label' in metric:
            ppm = metric['pos_label']
        score_list.append(Score(
            metric=ppm,
            fold=0,
            targets=targets,
            value=Value(raw=to_proto_value_raw(metric['value']))))
    scores = []
    scores.append(
        SolutionSearchScore(
            scoring_configuration=scoring_config,
            scores=score_list))
    search_solutions_results.append(GetSearchSolutionsResultsResponse(
        progress=Progress(state=core_pb2.COMPLETED,
        status="Done",
        start=timestamp.GetCurrentTime(),
        end=timestamp.GetCurrentTime()),
        done_ticks=0, # TODO: Figure out how we want to support this
        all_ticks=0, # TODO: Figure out how we want to support this
        solution_id=fitted_pipeline_id, # TODO: Populate this with the pipeline id
        internal_score=0,
        # scores=None # Optional so we will not tackle it until needed
        scores=scores
    ))

    return search_solutions_results


def to_proto_steps_description(pipeline : Pipeline) -> typing.List[StepDescription]:
    '''
    Convert free hyperparameters in d3m pipeline steps to protocol buffer StepDescription
    '''
    # Todo: To be implemented
    descriptions: typing.List[StepDescription] = []
    return descriptions

    # for step in pipeline.steps:
    #     print(step)
    #     if isinstance(step, PrimitiveStep):
    #         free = step.get_free_hyperparms()
    #         values = {}
    #         for name, hyperparam_class in free.items():
    #             default = hyperparam_class.get_default()
    #             values[name] = to_proto_value_with_type(default, hyperparam_class.structural_type)
    #         descriptions.append(StepDescription(
    #             primitive=PrimitiveStepDescription(hyperparams=values)))
    #     else:
    #         # TODO: Subpipeline not yet implemented
    #         pass
    # return descriptions

'''
This class implements the CoreServicer base class. The CoreServicer defines the methods that must be supported by a
TA2 server implementation. The CoreServicer class is generated by grpc using the core.proto descriptor file. See:
https://gitlab.com/datadrivendiscovery/ta3ta2-api.
'''
class TA2Servicer(core_pb2_grpc.CoreServicer):

    '''
    The __init__ method is used to establish the underlying TA2 libraries to service requests from the TA3 system.
    '''
    def __init__(self, libdir, output_dir:str = None):
        self.log_msg("Init invoked")
        self.libdir = libdir
        self.output_dir = output_dir
        self.controller = Controller(libdir)


    '''
    Hello call
    Non streaming call
    '''
    def Hello(self, request, context):
        self.log_msg(msg="Hello invoked")
        # TODO: Figure out what we should be sending back to TA3 here.
        return HelloResponse(user_agent="ISI",
                             version="2.0",
                             allowed_value_types="",
                             supported_extensions="")


    '''
    Search Solutions call
    Non streaming
    '''
    def SearchSolutions(self, request, context):
        self.log_msg(msg="SearchSolutions invoked")

        problem_json_dict = problem_to_json(request.problem)
        problem_parsed = problem_to_dict(request.problem)
        print('==json')
        pprint(problem_json_dict)
        print('==parsed')
        pprint(problem_parsed)


        # Although called uri, it's just a filepath to datasetDoc.json
        dataset_uri = request.inputs[0].dataset_uri

        config_dict = {
            'problem_json' : problem_json_dict,
            'problem_parsed' : problem_parsed,
            'dataset_schema': dataset_uri,
            'timeout' : request.time_bound
        }

        if self.output_dir is not None:
            config_dict['saving_folder_loc'] = self.output_dir

        pprint(config_dict)

        self.controller.initialize_from_ta3(config_dict)

        status = self.controller.train()

        return SearchSolutionsResponse(search_id=self.generateId())


    '''
    Get Search Solutions Results call
    Streams response to TA3
    '''
    def GetSearchSolutionsResults(self, request, context):
        self.log_msg(msg="GetSearchSolutionsResults invoked with search_id: " + request.search_id)

        problem = self.controller.problem
        fitted_pipeline: FittedPipeline = self.controller.candidate.data['fitted_pipeline']
        metrics_result = self.controller.candidate.data['test_metrics']
        search_solutions_results = to_proto_search_solution_requests(
            problem, fitted_pipeline.id, metrics_result)

        for solution in search_solutions_results:
            yield solution

        # timestamp = Timestamp()

        # problem = self.controller.problem

        # pprint(self.controller.candidate.data)

        # metrics_result = self.controller.candidate.data['test_metrics']
        # pipeline = self.controller.candidate.data['fitted_pipeline']
        # searchSolutionsResults = []

        # # Todo: controller needs to remember the partition method
        # scoring_config = ScoringConfiguration(
        #     method=core_pb2.HOLDOUT,
        #     train_test_ratio=5,
        #     random_seed=4676,
        #     stratified=True)
        # targets = []
        # problem_dict = problem
        # for inputs_dict in problem_dict['inputs']:
        #     for target in inputs_dict['targets']:
        #         targets.append(ProblemTarget(
        #             target_index = target['target_index'],
        #             resource_id = target['resource_id'],
        #             column_index = target['column_index'],
        #             column_name = target['column_name'],
        #             clusters_number = target['clusters_number']))
        # score_list = []
        # for metric in metrics_result:
        #     ppm = ProblemPerformanceMetric(metric=d3m_problem.PerformanceMetric.parse(metric['metric']).name)
        #     if 'k' in metric:
        #         ppm = metric['k']
        #     if 'pos_label' in metric:
        #         ppm = metric['pos_label']
        #     score_list.append(Score(
        #         metric=ppm,
        #         fold=0,
        #         targets=targets,
        #         value=Value(raw=to_proto_value_raw(metric['value']))))
        # scores = []
        # scores.append(
        #     SolutionSearchScore(
        #         scoring_configuration=scoring_config,
        #         scores=score_list))
        # searchSolutionsResults.append(GetSearchSolutionsResultsResponse(
        #     progress=Progress(state=core_pb2.COMPLETED,
        #     status="Done",
        #     start=timestamp.GetCurrentTime(),
        #     end=timestamp.GetCurrentTime()),
        #     done_ticks=0, # TODO: Figure out how we want to support this
        #     all_ticks=0, # TODO: Figure out how we want to support this
        #     solution_id=pipeline.id, # TODO: Populate this with the pipeline id
        #     internal_score=0,
        #     # scores=None # Optional so we will not tackle it until needed
        #     scores=scores
        # ))
        # # Add a second result to test streaming responses
        # # searchSolutionsResults.append(GetSearchSolutionsResultsResponse(
        # #     progress=Progress(state=core_pb2.RUNNING,
        # #     status="Done",
        # #     start=timestamp.GetCurrentTime(),
        # #     end=timestamp.GetCurrentTime()),
        # #     done_ticks=0,
        # #     all_ticks=0,
        # #     solution_id="JIOEPB343", # TODO: Populate this with the pipeline id
        # #     internal_score=0,
        # #     scores=None
        # # ))
        # for solution in searchSolutionsResults:
        #     yield solution


    '''
    Get the Score Solution request_id associated with the supplied solution_id
    Non streaming
    '''
    def ScoreSolution(self, request, context):
        self.log_msg(msg="ScoreSolution invoked with solution_id: " + request.solution_id)

        return ScoreSolutionResponse(
            # Generate valid request id 22 characters long for TA3 tracking
            request_id=self.generateId()
        )


    '''
    Get Score Solution Results call
    Streams response to TA3
    '''
    def GetScoreSolutionResults(self, request, context):
        self.log_msg(msg="GetScoreSolutionResults invoked with request_id: " + request.request_id)
        # For now just return the last scored fitted_pipeline

        problem = self.controller.problem
        fitted_pipeline: FittedPipeline = self.controller.candidate.data['fitted_pipeline']
        metrics_result = self.controller.candidate.data['test_metrics']
        search_solutions_results = to_proto_search_solution_requests(
            problem, fitted_pipeline.id, metrics_result)

        for solution in search_solutions_results:
            yield solution

        # scoreSolutionResults = []
        # timestamp = Timestamp()
        # scoreSolutionResults.append(
        #     GetScoreSolutionResultsResponse(
        #     progress=Progress(state=core_pb2.COMPLETED,
        #                       status="Good",
        #                       start=timestamp.GetCurrentTime(),
        #                       end=timestamp.GetCurrentTime()),
        #     scores=[Score(metric=ProblemPerformanceMetric(metric=problem_pb2.ACCURACY,
        #                                     k = 0,
        #                                     pos_label="0"),
        #                   fold=0,
        #                   targets=[ProblemTarget(target_index=0,
        #                                    resource_id="0",
        #                                    column_index=0,
        #                                    column_name="0",
        #                                    clusters_number=0)],
        #                   value=Value(double=0.8))]
        # ))
        # scoreSolutionResults.append(GetScoreSolutionResultsResponse(
        #     progress=Progress(state=core_pb2.PENDING,
        #                       status="Good",
        #                       start=timestamp.GetCurrentTime(),
        #                       end=timestamp.GetCurrentTime()
        # )))
        # for score in scoreSolutionResults:
        #     yield score


    '''
    End the solution search process with the supplied search_id
    Non streaming
    '''
    def EndSearchSolutions(self, request, context):
        self.log_msg(msg="EndSearchSolutions invoked with search_id: " + request.search_id)

        return EndSearchSolutionsResponse()


    def GetProduceSolutionResults(self, request, context):
        pass


    def SolutionExport(self, request, context):
        pass


    def GetFitSolutionResults(self, request, context):
        pass


    def StopSearchSolutions(self, request, context):
        pass


    def ListPrimitives(self, request, context):
        primitives = []
        for python_path in d3m.index.search():
            primitives.append(to_proto_primitive(d3m.index.get_primitive(python_path)))
        return ListPrimitivesResponse(primitives = primitives)



    def ProduceSolution(self, request, context):
        pass


    def FitSolution(self, request, context):
        pass


    def UpdateProblem(self, request, context):
        pass


    def DescribeSolution(self, request, context) -> DescribeSolutionResponse:
        # pipeline = controller.getFittedPipeline(request.solution_id)

        if self.controller.candidate is None:
            return DescribeSolutionResponse()

        if self.controller.candidate.data is None:
            return DescribeSolutionResponse()

        if not 'fitted_pipeline' in self.controller.candidate.data:
            return DescribeSolutionResponse()

        fitted_pipeline: FittedPipeline = self.controller.candidate.data['fitted_pipeline']
        pipeline = fitted_pipeline.pipeline

        return DescribeSolutionResponse(
            pipeline=to_proto_pipeline(pipeline, fitted_pipeline.id),
            steps=to_proto_steps_description(pipeline)
        )


    '''
    Handy method for generating pipeline trace logs
    '''
    def log_msg(self, msg):
        msg = str(msg)
        for line in msg.splitlines():
            _logger.info("    | %s" % line)
        _logger.info("    \\_____________")


    '''
    Convenience method for generating 22 character id's
    '''
    def generateId(self):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(22))
