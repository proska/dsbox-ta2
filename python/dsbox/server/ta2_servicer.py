import os
import pdb
import sys
import typing

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ta3ta2_api = os.path.abspath(os.path.join(
    CURRENT_DIR, '..', '..', '..', '..', 'ta3ta2-api'))
print(ta3ta2_api)
sys.path.append(ta3ta2_api)


import d3m.metadata.problem as d3m_problem

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, SubpipelineStep, ArgumentType
from d3m.primitive_interfaces.base import PrimitiveBase

import core_pb2
import core_pb2_grpc
import logging
from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore
import random
import string

from pprint import pprint

import problem_pb2
import value_pb2

# import autoflowconfig
from core_pb2 import EndSearchSolutionsResponse
from core_pb2 import EvaluationMethod
from core_pb2 import GetScoreSolutionResultsResponse
from core_pb2 import GetSearchSolutionsResultsResponse
from core_pb2 import HelloResponse
from core_pb2 import PrimitiveStepDescription
from core_pb2 import Progress
from core_pb2 import ProgressState
from core_pb2 import Score
from core_pb2 import ScoreSolutionResponse
from core_pb2 import ScoringConfiguration
from core_pb2 import SearchSolutionsResponse
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

from problem_pb2 import ProblemPerformanceMetric
from problem_pb2 import PerformanceMetric
from problem_pb2 import ProblemTarget

from primitive_pb2 import Primitive

from value_pb2 import Value

from dsbox.controller.controller import Controller

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


# The output of this function should be the same sas the output for
# d3m/metadata/problem.py:parse_problem_description

def problem_to_dict(problem) -> typing.Dict:
    description = {
        'schema': d3m_problem.PROBLEM_SCHEMA_VERSION,
        'problem': {
            'id': problem.problem.id,
            # "problemVersion" is required by the schema, but we want to be compatible with problem
            # descriptions which do not adhere to the schema.
            'version': problem.problem.version,
            'name': problem.problem.name,
            'task_type': d3m_problem.TaskType(problem.problem.task_type),
            'task_subtype': d3m_problem.TaskSubtype(problem.problem.task_subtype)
        },
        # 'outputs': {
        #     'predictions_file': problem_doc['expectedOutputs']['predictionsFile'],
        # }
    }

    performance_metrics = []
    for metrics in problem.problem.performance_metrics:
        if metrics.metric==0:
            d3m_metric = None
        else:
            d3m_metric = d3m_problem.PerformanceMetric(metrics.metric)
        params = {}
        if d3m_metric == d3m_problem.PerformanceMetric.F1:
            if metrics.pos_label is None:
                params['pos_label'] = '1'
            else:
                params['pos_label'] = metrics.pos_label
        if metrics.k is not None:
            params['k'] = metrics.k
        performance_metrics.append ({
            'metric' : d3m_metric,
            'params' : params
        })
    description['problem']['performance_metrics'] = performance_metrics

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
                container=argument_desc['data'])
        else:
            # ArgumentType.DATA
            arguments[argument_name] = PrimitiveStepArgument(
                data= argument_desc['data'])
    outputs = [StepOutput(id=output) for output in step.outputs]
    hyperparams = {}
    for name, hyperparam_dict in step.hyperparams:
        hyperparam_type = hyperparam_dict['type']
        hyperparam_data = hyperparam_dict['data']
        if hyperparam_type==ArgumentType.CONTAINER:
            hyperparam = PrimitiveStepHyperparameter(container=hyperparam_data)
        elif hyperparam_type==ArgumentType.DATA:
            hyperparam = PrimitiveStepHyperparameter(data=hyperparam_data)
        elif hyperparam_type==ArgumentType.PRIMITIVE:
            hyperparam = PrimitiveStepHyperparameter(primitive=hyperparam_data)
        elif hyperparam_type==ArgumentType.VALUE:
            hyperparam = PrimitiveStepHyperparameter(value=hyperparam_data)
        else:
            # Should never get here. Dataset is not a valid ArgumentType
            hyperparam = PrimitiveStepHyperparameter(data_set=hyperparam_data)
        hyperparams[name] = hyperparam
    primitive_description = PrimitivePipelineDescriptionStep(
        primitive=to_proto_primitive(step),
        arguments=arguments,
        outputs=outputs,
        hyperparams=hyperparams,
        users=[PipelineDescriptionUser(id=user_description)
               for user_description in step.users]
    )
    return PipelineDescriptionStep(primitive=primitive_description)

def to_proto_pipeline(pipeline : Pipeline) -> PipelineDescription:
    """
    Convert d3m Pipeline to protocol buffer PipelineDescription
    """
    inputs = []
    outputs = []
    steps = []
    users = []
    for input_description in pipeline.inputs:
        if 'name' in input_description:
            inputs.append(PipelineDescriptionInput(input_description['name']))
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
    return PipelineDescription(
        id=pipeline.id,
        source=pipeline.source,
        created=pipeline.created,
        context=pipeline.context,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        name=pipeline.name,
        description=pipeline.description,
        users=users
    )

def to_proto_step_descriptions(pipeline : Pipeline) -> typing.List[StepDescription]:
    '''
    Convert free hyperparameters in d3m pipeline steps to protocol buffer StepDescription
    '''
    decriptions = []
    for step in pipeline.steps:
        if isinstance(step, PrimitiveStep):
            free = step.get_free_hyperparms()
            decriptions.append(StepDescription(
                primitive=PrimitiveStepDescription(hyperparams=free)))
        else:
            # TODO: Subpipeline not yet implemented
            pass
    return decriptions

'''
This class implements the CoreServicer base class. The CoreServicer defines the methods that must be supported by a
TA2 server implementation. The CoreServicer class is generated by grpc using the core.proto descriptor file. See:
https://gitlab.com/datadrivendiscovery/ta3ta2-api.
'''
class TA2Servicer(core_pb2_grpc.CoreServicer):

    '''
    The __init__ method is used to establish the underlying TA2 libraries to service requests from the TA3 system.
    '''
    def __init__(self, libdir):
        self.log_msg("Init invoked")
        self.libdir = libdir
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

        problem_description = problem_to_dict(request.problem)

        # Although called uri, it's just a filepath to datasetDoc.json
        dataset_uri = request.inputs[0].dataset_uri

        config_dict = {
            'problem' : problem_description,
            'dataset_schema': dataset_uri,
            'timeout' : request.time_bound
        }

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
        # TODO: Read the pipelines we generated and munge them into the response for TA3
        timestamp = Timestamp()
        searchSolutionsResults = []

        score = self.controller.candidate_value['validation_metrics']['value']
        scoring_config = ScoringConfiguration(
            method=EvaluationMethod.HOLDOUT,
            train_test_ratio=5,
            random_seed=4676,
            stratified=True
        )
        targets = []
        problem_dict = self.controller.problem
        for target_dict in problem_dict['inputs']['targets']:
            targets.append(ProblemTarget(
                target_index = target_dict['target_index'],
                resource_id = target_dict['resource_id'],
                column_index = target_dict['column_index'],
                column_name = target_dict['column_name'],
                clusters_number = target_dict['clusters_number']
            )
        )

        score = Score(
            metric=ProblemPerformanceMetric(
                metric=PerformanceMetric(
                    self.controller.candidate_value['validation_metrics']['metric']),
                k=0,
                pos_label = ''),
            fold=0,
            targets=targets
        )
        scores = SearchSolutionScore(
            scoring_configuration=scoring_config,
            scores=[score]
        )
        searchSolutionsResults.append(GetSearchSolutionsResultsResponse(
            progress=Progress(state=core_pb2.COMPLETED,
            status="Done",
            start=timestamp.GetCurrentTime(),
            end=timestamp.GetCurrentTime()),
            done_ticks=0, # TODO: Figure out how we want to support this
            all_ticks=0, # TODO: Figure out how we want to support this
            solution_id="HIDOEI8973", # TODO: Populate this with the pipeline id
            internal_score=0,
            # scores=None # Optional so we will not tackle it until needed
            scores=scores
        ))
        # Add a second result to test streaming responses
        # searchSolutionsResults.append(GetSearchSolutionsResultsResponse(
        #     progress=Progress(state=core_pb2.RUNNING,
        #     status="Done",
        #     start=timestamp.GetCurrentTime(),
        #     end=timestamp.GetCurrentTime()),
        #     done_ticks=0,
        #     all_ticks=0,
        #     solution_id="JIOEPB343", # TODO: Populate this with the pipeline id
        #     internal_score=0,
        #     scores=None
        # ))
        for solution in searchSolutionsResults:
            yield solution


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

        scoreSolutionResults = []
        timestamp = Timestamp()
        scoreSolutionResults.append(
            GetScoreSolutionResultsResponse(
            progress=Progress(state=core_pb2.COMPLETED,
                              status="Good",
                              start=timestamp.GetCurrentTime(),
                              end=timestamp.GetCurrentTime()),
            scores=[Score(metric=ProblemPerformanceMetric(metric=problem_pb2.ACCURACY,
                                            k = 0,
                                            pos_label="0"),
                          fold=0,
                          targets=[ProblemTarget(target_index=0,
                                           resource_id="0",
                                           column_index=0,
                                           column_name="0",
                                           clusters_number=0)],
                          value=Value(double=0.8))]
        ))
        scoreSolutionResults.append(GetScoreSolutionResultsResponse(
            progress=Progress(state=core_pb2.PENDING,
                              status="Good",
                              start=timestamp.GetCurrentTime(),
                              end=timestamp.GetCurrentTime()
        )))
        for score in scoreSolutionResults:
            yield score


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
        for python_path in controller.index.search():
            primitives.append(to_proto_primitive(
                controller.index.get_primitive(python_path)))
        return ListPrimitivesResponse(primitives = primitives)



    def ProduceSolution(self, request, context):
        pass


    def FitSolution(self, request, context):
        pass


    def UpdateProblem(self, request, context):
        pass


    def DescribeSolution(self, request, context):
        pipeline = controller.getFittedPipeline(request.solution_id)
        return DescribeSolutionRespose(
            pipeline=to_proto_pipeline(pipeline),
            steps=to_proto_step_description(pipeline)
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
