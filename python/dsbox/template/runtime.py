import argparse
import json
import os
import typing

import networkx  # type: ignore
from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Metadata
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitive_interfaces import base


class Runtime:
    """
    Class to run the build and run a Pipeline.

    Attributes
    ----------
    pipeline_description : Pipeline
        A pipeline description to be executed.
    primitives_arguments: Dict[int, Dict[str, Dict]
        List of indexes reference to the arguments for each step.
    execution_order
        List of indexes that contains the execution order.
    pipeline
        List of different models generated by the primitives.
    outputs
        List of indexes reference how to build the the output.

    Parameters
    ----------
    pipeline_description : Pipeline
        A pipeline description to be executed.
    """

    def __init__(self, pipeline_description: Pipeline) -> None:
        self.pipeline_description = pipeline_description
        n_steps = len(self.pipeline_description.steps)

        self.primitives_arguments: typing.Dict[int, typing.Dict[str, typing.Dict]] = {}
        for i in range(0, n_steps):
            self.primitives_arguments[i] = {}

        self.execution_order: typing.List[int] = []

        self.pipeline: typing.List[typing.Optional[base.PrimitiveBase]] = [None] * n_steps
        self.outputs: typing.List[typing.Tuple[str, int]] = []

        # Getting the outputs
        for output in self.pipeline_description.outputs:
            origin = output['data'].split('.')[0]
            source = output['data'].split('.')[1]
            self.outputs.append((origin, int(source)))

        # Constructing DAG to determine the execution order
        execution_graph = networkx.DiGraph()
        for i in range(0, n_steps):
            primitive_step: PrimitiveStep = typing.cast(PrimitiveStep, self.pipeline_description.steps[i])
            for argument, data in primitive_step.arguments.items():
                argument_edge = data['data']
                origin = argument_edge.split('.')[0]
                source = argument_edge.split('.')[1]

                self.primitives_arguments[i][argument] = {'origin': origin, 'source': int(source)}

                if origin == 'steps':
                    execution_graph.add_edge(str(source), str(i))
                else:
                    execution_graph.add_edge(origin, str(i))

        execution_order = list(networkx.topological_sort(execution_graph))

        # Removing non-step inputs from the order
        execution_order = list(filter(lambda x: x.isdigit(), execution_order))
        self.execution_order = [int(x) for x in execution_order]

        # Creating set of steps to be call in produce
        self.produce_order: typing.Set[int] = set()
        for output in self.pipeline_description.outputs:
            origin = output['data'].split('.')[0]
            source = output['data'].split('.')[1]
            if origin != 'steps':
                continue
            else:
                current_step = int(source)
                self.produce_order.add(current_step)
                for i in range(0, len(execution_order)):
                    step_origin = self.primitives_arguments[current_step]['inputs']['origin']
                    step_source = self.primitives_arguments[current_step]['inputs']['source']
                    if step_origin != 'steps':
                        break
                    else:
                        self.produce_order.add(step_source)
                        current_step = step_source
        # kyao!!!!
        self.produce_order = set(self.execution_order)
        self.fit_outputs = []
        self.produce_outputs = []

    def fit(self, **arguments: typing.Any) -> None:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to train the Pipeline
        """

        primitives_outputs: typing.List[typing.Optional[base.CallResult]] = [None] * len(self.execution_order)

        for i in range(0, len(self.execution_order)):
            primitive_arguments: typing.Dict[str, typing.Any] = {}
            n_step = self.execution_order[i]
            for argument, value in self.primitives_arguments[n_step].items():
                if value['origin'] == 'steps':
                    primitive_arguments[argument] = primitives_outputs[value['source']]
                else:
                    primitive_arguments[argument] = arguments[argument][value['source']]

            if isinstance(self.pipeline_description.steps[n_step], PrimitiveStep):
                primitive_step: PrimitiveStep = typing.cast(PrimitiveStep, self.pipeline_description.steps[n_step])
                primitives_outputs[n_step] = self._primitive_step_fit(n_step, primitive_step, primitive_arguments)
        # kyao!!!!
        self.fit_outputs = primitives_outputs

    def _primitive_step_fit(self, n_step: int, step: PrimitiveStep, primitive_arguments: typing.Dict[str, typing.Any]) -> base.CallResult:
        """
        Execute a step and train it with primitive arguments.

        Paramters
        ---------
        n_step: int
            An integer of the actual step.
        step: PrimitiveStep
            A primitive step.
        primitive_arguments
            Arguments for set_training_data, fit, produce of the primitive for this step.

        """
        primitive: typing.Type[base.PrimitiveBase] = step.primitive
        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        custom_hyperparams = dict()

        if bool(step.hyperparams):
            for hyperparam, value in step.hyperparams.items():
                if isinstance(value, dict):
                    custom_hyperparams[hyperparam] = value['data']
                else:
                    custom_hyperparams[hyperparam] = value

        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments: typing.Dict[str, typing.Any] = {}
        produce_params_primitive = self._primitive_arguments(primitive, 'produce')
        produce_params: typing.Dict[str, typing.Any] = {}

        for param, value in primitive_arguments.items():
            if param in produce_params_primitive:
                produce_params[param] = value
            if param in training_arguments_primitive:
                training_arguments[param] = value
        try:
            model = primitive(hyperparams=primitive_hyperparams(
                        primitive_hyperparams.defaults(), **custom_hyperparams))
        except:
            print("******************\n[ERROR]Hyperparameters unsuccesfully set - using defaults")
            model = primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults()))
        model.set_training_data(**training_arguments)
        model.fit()
        self.pipeline[n_step] = model
        return model.produce(**produce_params).value

    def _primitive_arguments(self, primitive: typing.Type[base.PrimitiveBase], method: str) -> set:
        """
        Get the arguments of a primitive given a function.

        Paramters
        ---------
        primitive
            A primitive.
        method
            A method of the primitive.
        """
        return set(primitive.metadata.query()['primitive_code']['instance_methods'][method]['arguments'])

    def produce(self, **arguments: typing.Any) -> typing.List:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to execute the Pipeline
        """
        steps_outputs = [None] * len(self.execution_order)

        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
            primitive_step: PrimitiveStep = typing.cast(PrimitiveStep, self.pipeline_description.steps[n_step])
            produce_arguments_primitive = self._primitive_arguments(primitive_step.primitive, 'produce')
            produce_arguments: typing.Dict[str, typing.Any] = {}

            for argument, value in self.primitives_arguments[n_step].items():
                if argument in produce_arguments_primitive:
                    if value['origin'] == 'steps':
                        produce_arguments[argument] = steps_outputs[value['source']]
                    else:
                        produce_arguments[argument] = arguments[argument][value['source']]
                    if produce_arguments[argument] is None:
                        continue
            if isinstance(self.pipeline_description.steps[n_step], PrimitiveStep):
                if n_step in self.produce_order:
                    steps_outputs[n_step] = self.pipeline[n_step].produce(**produce_arguments).value
                else:
                    steps_outputs[n_step] = None
        # kyao!!!!
        self.produce_outputs = steps_outputs
        
        # Create output
        pipeline_output: typing.List = []
        for output in self.outputs:
            if output[0] == 'steps':
                pipeline_output.append(steps_outputs[output[1]])
            else:
                pipeline_output.append(arguments[output[0][output[1]]])
        return pipeline_output


def load_problem_doc(problem_doc_path: str) -> Metadata:
    """
    Load problem_doc from problem_doc_path

    Paramters
    ---------
    problem_doc_path
        Path where the problemDoc.json is located
    """

    with open(problem_doc_path) as file:
        problem_doc = json.load(file)
    return Metadata(problem_doc)


def add_target_columns_metadata(dataset: 'Dataset', problem_doc_metadata: 'Metadata') -> Dataset:
    """
    Add metadata to the dataset from problem_doc_metadata

    Paramters
    ---------
    dataset
        Dataset
    problem_doc_metadata:
        Metadata about the problemDoc
    """

    for data in problem_doc_metadata.query(())['inputs']['data']:
        targets = data['targets']
        for target in targets:
            semantic_types = list(dataset.metadata.query(
                (target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex'])).get('semantic_types', []))

            if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                dataset.metadata = dataset.metadata.update(
                    (target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex']), {'semantic_types': semantic_types})

            if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                dataset.metadata = dataset.metadata.update(
                    (target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex']), {'semantic_types': semantic_types})

    return dataset


def generate_pipeline(pipeline_path: str, dataset_path: str, problem_doc_path: str, resolver: Resolver = None) -> Runtime:
    """
    Simplified interface that fit a pipeline with a dataset

    Paramters
    ---------
    pipeline_path
        Path to the pipeline description
    dataset_path:
        Path to the datasetDoc.json
    problem_doc_path:
        Path to the problemDoc.json
    resolver : Resolver
        Resolver to use.
    """

    # Pipeline description
    pipeline_description = None
    if '.json' in pipeline_path:
        with open(pipeline_path) as pipeline_file:
            pipeline_description = Pipeline.from_json(string_or_file=pipeline_file, resolver=resolver)
    else:
        with open(pipeline_path) as pipeline_file:
            pipeline_description = Pipeline.from_yaml(string_or_file=pipeline_file, resolver=resolver)

    # Problem Doc
    problem_doc = load_problem_doc(problem_doc_path)

    # Dataset
    if 'file:' not in dataset_path:
        dataset_path = 'file://{dataset_path}'.format(dataset_path=os.path.abspath(dataset_path))

    dataset = D3MDatasetLoader().load(dataset_uri=dataset_path)
    # Adding Metadata to Dataset
    dataset = add_target_columns_metadata(dataset, problem_doc)

    # Pipeline
    pipeline_runtime = Runtime(pipeline_description)
    # Fitting Pipeline
    pipeline_runtime.fit(inputs=[dataset])
    return pipeline_runtime


def test_pipeline(pipeline_runtime: Runtime, dataset_path: str) -> typing.List:
    """
    Simplified interface test a pipeline with a dataset

    Paramters
    ---------
    pipeline_runtime
        Runtime object
    dataset_path:
        Path to the datasetDoc.json
    """

    # Dataset
    if 'file:' not in dataset_path:
        dataset_path = 'file://{dataset_path}'.format(dataset_path=os.path.abspath(dataset_path))
    dataset = D3MDatasetLoader().load(dataset_uri=dataset_path)

    return pipeline_runtime.produce(inputs=[dataset])


def load_args() -> typing.Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Run pipelines.")

    parser.add_argument(
        'pipeline', action='store', metavar='PIPELINE',
        help="path to a pipeline file (.json or .yml)",
    )

    parser.add_argument(
        'dataset', action='store', metavar='DATASET',
        help="path to the primary datasetDoc.json for the dataset you want to use.",
    )

    arguments = parser.parse_args()

    return os.path.abspath(arguments.pipeline), os.path.abspath(arguments.dataset)


def main() -> None:
    pipeline_path, dataset_path = load_args()

    base_dataset_dir = os.path.abspath(os.path.join(dataset_path, os.pardir, os.pardir))
    train_dataset_doc = os.path.join(base_dataset_dir, 'TRAIN', 'dataset_TRAIN', 'datasetDoc.json')
    train_problem_doc = os.path.join(base_dataset_dir, 'TRAIN', 'problem_TRAIN', 'problemDoc.json')
    test_dataset_doc = os.path.join(base_dataset_dir, 'TEST', 'dataset_TEST', 'datasetDoc.json')

    pipeline_runtime = generate_pipeline(
            pipeline_path=pipeline_path,
            dataset_path=train_dataset_doc,
            problem_doc_path=train_problem_doc)

    results = test_pipeline(pipeline_runtime, test_dataset_doc)
    print(results)


if __name__ == '__main__':
    main()
