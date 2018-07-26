import json
import glob
import typing

from d3m import index
from d3m.container.dataset import SEMANTIC_TYPES
from d3m.metadata.problem import TaskType, TaskSubtype
from d3m.container.list import List
from dsbox.template.template import TemplatePipeline, DSBoxTemplate


class TemplateDescription:
    """
    Description of templates in the template library.

    Attributes
    ----------
    task : TaskType
        The type task the template handles
    template: TemplatePipeline
        The actual template
    target_step: int
        The step of the template that extract the ground truth target from
        the dataset
    predicted_target_step: int
        The step of the template generates the predictions
    """

    def __init__(self, task: TaskType, template: TemplatePipeline,
                 target_step: int, predicted_target_step: int) -> None:
        self.task = task
        self.template = template

        # Instead of having these attributes here, probably should attach
        # attributes to the template steps
        self.target_step = target_step
        self.predicted_target_step = predicted_target_step


class TemplateLibrary:
    """
    Library of template pipelines
    """

    def __init__(self, library_dir: str = None, run_single_template: str = "") -> None:
        self.templates: typing.List[typing.Type[DSBoxTemplate]] = []
        self.primitive: typing.Dict = index.search()

        self.library_dir = library_dir
        if self.library_dir is None:
            self._load_library()

        self.all_templates = {
            # classification
            "random_forest_classification_template": RandomForestClassificationTemplate, 
            "extra_trees_classification_template": ExtraTreesClassificationTemplate,
            "gradient_boosting_classification_template": GradientBoostingClassificationTemplate,
            "svc_classification_template": SVCClassificationTemplate,

            #regression
            "random_forest_regression_template": RandomForestRegressionTemplate,
            "extra_trees_regression_template": ExtraTreesRegressionTemplate,
            "gradient_boosting_regression_template": GradientBoostingRegressionTemplate,
            "svr_regression_template": SVRRegressionTemplate,
        }

        if run_single_template:
            self._load_single_inline_templates(run_single_template)
        else:
            self._load_inline_templates()

    def get_templates(self, task: TaskType, subtype: TaskSubtype, taskSourceType: SEMANTIC_TYPES)\
            -> typing.List[DSBoxTemplate]:
        results = []
        results.append(SRIMeanBaselineTemplate())  # put the meanbaseline here so whatever dataset will have a result
        for template_class in self.templates:
            template = template_class()
            # sourceType refer to d3m/container/dataset.py ("SEMANTIC_TYPES" as line 40-70)
            # taskType and taskSubtype refer to d3m/
            if task.name in template.template['taskType'] and subtype.name in template.template[
                'taskSubtype']:
                # if there is only one task source type which is table, we don't need to check
                # other things
                if {"table"} == taskSourceType and template.template['inputType'] == "table":
                    results.append(template)
                else:
                    # otherwise, we need to process in another way because "table" source type
                    # exist nearly in every dataset
                    if "table" in taskSourceType:
                        taskSourceType.remove("table")

                    for each_source_type in taskSourceType:
                        if each_source_type in {template.template['inputType']}:
                            results.append(template)
        # if we finally did not find a proper template to use
        if results == []:
            print("Error: Can't find a suitable template type to fit the problem.")
        else:
            print("[INFO] Template choices:")
            # otherwise print the template list we added
            for each_template in results:
                print("Template '", each_template.template["name"],
                      "' has been added to template base.")

        return results

    def _load_library(self):
        # TODO
        # os.path.join(library_dir, 'template_library.yaml')
        pass

    def _load_inline_templates(self):
    	for template in self.all_templates.values()
    		self.templates.append(template)

    def _load_single_inline_templates(self, template_name):
        if template_name in self.all_templates:
            self.templates.append(self.all_templates[template_name])
        else:
            raise KeyError("Template not found, name: {}".format(template_name))


class SemanticTypeDict(object):
    def __init__(self, libdir):
        self.pos = libdir
        self.mapper = {}

    def read_primitives(self) -> None:

        # jsonPath = os.path.join(libdir, filename)
        # print(self.pos)
        user_Defined_Confs = glob.glob(
            self.pos + "/*_template_semantic_mapping.json")
        # print(user_Defined_Confs)
        for u in user_Defined_Confs:
            with open(u, "r") as cf:
                print("opened", u)
                for v in json.load(cf).items():
                    self.mapper[v[0]] = v[1]

    def create_configuration_space(self, template: TemplatePipeline):
        definition = {}
        # for t in TemplatePipeline:
        #     if isinstance(t, list):
        steps = template.template_nodes.keys()
        for s in steps:
            if template.template_nodes[s].semantic_type in self.mapper.keys():
                definition[s] = self.mapper[
                    template.template_nodes[s].semantic_type]

        # return SimpleConfigurationSpace(definition)
        return definition


class RandomForestClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "random_forest_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                    # 'n_hidden':[(10)],
                                    # 'threshold':[(0)],
                                    # # 'threshold':[(0), (500)],
                                    # 'n_grams':[(1), (5)],
                                    # 'max_df':[(.9)],
                                    # 'min_df':[(.02)],
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                        "hyperparameters":
                            {
                            'bootstrap': [True, False],
                            'max_depth': [15, 30, None],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'max_features': ['auto', 'sqrt'],
                            'n_estimators': [10, 50, 100]
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7

class ExtraTreesClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "extra_trees_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKExtraTreesClassifier",
                        "hyperparameters":
                            {
                            'bootstrap': [True, False],
                            'max_depth': [15, 30, None],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'max_features': ['auto', 'sqrt'],
                            'n_estimators': [10, 50, 100]
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7

class GradientBoostingClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "gradient_boosting_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier",
                        "hyperparameters":
                            {
                            'max_depth': [2,3,5],
                            'n_estimators': [50, 75, 100],
                            'learning_rate': [0.1, 0.3, 0.5],
                            'min_samples_split': [1, 2, 3],
                            'min_samples_leaf': [1, 2],
                            'max_features': [None, 'auto', 'sqrt']
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class SVCClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "svc_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKSVC",
                        "hyperparameters":
                            {
                                'C': [0.8, 1.0, 1.2],
                                'kernel': ['rbf', 'poly'],
                                'degree': [2, 3, 4],
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class SVRRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "svr_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKSVR",
                        "hyperparameters":
                            {
                                'C': [0.8, 1.0, 1.2],
                                'kernel': ['rbf', 'poly'],
                                'degree': [2, 3, 4, 5],
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7

class GradientBoostingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "gradient_boosting_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKGradientBoostingRegressor",
                        "hyperparameters":
                            {
                            'max_depth': [2,3,5],
                            'n_estimators': [100, 150, 200],
                            'learning_rate': [0.1, 0.3, 0.5],
                            'min_samples_split': [1, 2, 3],
                            'min_samples_leaf': [1, 2],
                            'max_features': [None, 'auto', 'sqrt']
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7

class ExtraTreesRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "extra_trees_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKExtraTreesRegressor",
                        "hyperparameters":
                            {
                            # 'bootstrap': [True, False],
                            'max_depth': [15, 30, None],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'max_features': ['auto', 'sqrt'],
                            'n_estimators': [10, 50, 100]
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


class RandomForestRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "random_forest_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
        # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.dsbox.Denormalize"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "profiler_step",
                    "primitives": ["d3m.primitives.dsbox.Profiler"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "clean_step",
                    "primitives": ["d3m.primitives.dsbox.CleaningFeaturizer"],
                    "inputs": ["profiler_step"]
                },
                {
                    "name": "corex_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.dsbox.CorexText",
                            "hyperparameters":
                                {
                                }
                        },
                    ],
                    "inputs": ["clean_step"]
                },
                {
                    "name": "encoder_step",
                    "primitives": ["d3m.primitives.dsbox.Encoder"],
                    "inputs": ["corex_step"]
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.sklearn_wrap.SKImputer"],
                    "inputs": ["encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/Target',
                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step",
                    "runtime": {
                        "cross_validation": 10,
                        "stratified": True
                    },
                    "primitives": [
                        {
                        "primitive":
                            "d3m.primitives.sklearn_wrap.SKRandomForestRegressor",
                        "hyperparameters":
                            {
                            'bootstrap': [True, False],
                            'max_depth': [15, 30, None],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'max_features': ['auto', 'sqrt'],
                            'n_estimators': [10, 50, 100]
                            }
                        },
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7
