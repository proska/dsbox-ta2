import copy
import glob
import json
import logging
import numpy as np
import typing

from d3m import index
from d3m.container.dataset import SEMANTIC_TYPES
from d3m.metadata.problem import TaskType, TaskSubtype
from dsbox.template.template import DSBoxTemplate
from .template_steps import TemplateSteps

_logger = logging.getLogger(__name__)

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
            "sample_classification_template": SampleClassificationTemplate,
            "no_cleaning_classification_template": NoCleaningClassificationTemplate
        } # only two templates: with/without cleaningfeaturizer


        if run_single_template:
            self._load_single_inline_templates(run_single_template)
        else:
            self._load_inline_templates()

    def get_templates(self, task: TaskType, subtype: TaskSubtype, taskSourceType: SEMANTIC_TYPES) \
            -> typing.List[DSBoxTemplate]:
        results = []
        for template_class in self.templates:
            template = template_class()
            # sourceType refer to d3m/container/dataset.py ("SEMANTIC_TYPES" as line 40-70)
            # taskType and taskSubtype refer to d3m/
            if task.name in template.template['taskType'] and subtype.name in template.template['taskSubtype']:
                # if there is only one task source type which is table, we don't need to check
                # other things
                taskSourceType_check = copy.copy(taskSourceType)
                if {"table"} == taskSourceType_check and "table" in template.template['inputType']:
                    results.append(template)
                else:
                    # otherwise, we need to process in another way because "table" source type
                    # exist nearly in every dataset
                    if "table" in taskSourceType_check:
                        taskSourceType_check.remove("table")

                    for each_source_type in taskSourceType_check:
                        if type(template.template['inputType']) is set:
                            if each_source_type in template.template['inputType']:
                                results.append(template)
                        else:
                            if each_source_type == template.template['inputType']:
                                results.append(template)

        # if we finally did not find a proper template to use
        if results == []:
            _logger.error(f"Cannot find a suitable template type to fit the problem: {task.name}")
        else:
            # otherwise print the template list we added
            for each_template in results:
                _logger.info(f"{each_template} has been added to template base.")

        return results

    def _load_library(self):
        # TODO
        # os.path.join(library_dir, 'template_library.yaml')
        pass

    def _load_inline_templates(self):
        self.templates.append(SampleClassificationTemplate)
        self.templates.append(NoCleaningClassificationTemplate)
        # do loading inline here, use self.templates.append()

    def _load_single_inline_templates(self, template_name):
        if template_name in self.all_templates:
            self.templates.append(self.all_templates[template_name])
        else:
            raise KeyError("Template not found, name: {}".format(template_name))

################################################################################################################


######################################            Templates            #########################################


################################################################################################################

class SampleClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "sample_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "runtime": {
                                 "cross_validation": 10,
                                 "stratified": True
                             },
                             "primitives": [
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.random_forest.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': [True, False],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.extra_trees.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': [True, False],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'max_depth': [2, 3, 4, 5],
                                            'n_estimators': [50, 60, 80, 100],
                                            'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                            'min_samples_split': [2, 3],
                                            'min_samples_leaf': [1, 2],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         }
                     ]
        }

class NoCleaningClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "no_cleaning_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_steps_no_cleaning() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "runtime": {
                                 "cross_validation": 10,
                                 "stratified": True
                             },
                             "primitives": [
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.random_forest.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': [True, False],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.extra_trees.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'bootstrap': [True, False],
                                            'max_depth': [15, 30, None],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'n_estimators': [10, 50, 100],
                                         }
                                 },
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.gradient_boosting.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            'max_depth': [2, 3, 4, 5],
                                            'n_estimators': [50, 60, 80, 100],
                                            'learning_rate': [0.1, 0.2, 0.4, 0.5],
                                            'min_samples_split': [2, 3],
                                            'min_samples_leaf': [1, 2],
                                         }
                                 },
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         }
                     ]
        }