import traceback
import logging
import time
import typing
import random
from multiprocessing import Pool
from pprint import pprint

import pandas as pd
import numpy as np
from d3m.container.dataset import Dataset
from d3m.metadata.base import Metadata
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import ConfigurationSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceParallelBaseSearch import \
    TemplateSpaceParallelBaseSearch
from dsbox.combinatorial_search.search_utils import random_choices_without_replacement
from dsbox.combinatorial_search.ExecutionHistory import ExecutionHistory
from dsbox.template.configuration_space import ConfigurationPoint
from dsbox.template.template import DSBoxTemplate
from dsbox.template.template import HyperparamDirective
import itertools

T = typing.TypeVar("T")
# python path of primitive, i.e. 'd3m.primitives.common_primitives.RandomForestClassifier'
PythonPath = typing.NewType('PythonPath', str)

PrimitiveDescription = typing.NewType('PrimitiveDescription', dict)

_logger = logging.getLogger(__name__)


class MultiBanditSearch(TemplateSpaceParallelBaseSearch[T]):
    max_trials_per_template: int = 3
    """
    Use multi armed bandit to select the templates then search through the template randomly for
    a while
    """

    def __init__(self, num_proc):
        super().__init__(num_proc=num_proc)

    def _select_next_template(self, num_iter=2) -> \
            typing.Tuple[ConfigurationSpaceBaseSearch, str]:

        yield from self._bandit_select_next_template(num_iter)

    def _bandit_select_next_template(self, num_iter: int) -> ConfigurationSpaceBaseSearch:
        # initial evaluation
        for search in self.confSpaceBaseSearch:
            # yield search, "random"
            yield search
        _logger.info(f'Finished random phase, start sampling based on UCT score')
        # UCT based evaluation
        for _ in range(num_iter):
            _choices, _weights = self._update_UCT_score()
            selected = random_choices_without_replacement(_choices, _weights, 1)
            yield selected[0]

    def _update_UCT_score(self) -> typing.Tuple[ConfigurationSpaceBaseSearch, typing.Dict]:

        _logger.debug(f'execution history table for templates:\n{self.history}')

        try:
            normalize = self.history.normalize()
            total_time = normalize['total_runtime'].sum()
            total_run = normalize['sim_count'].sum()
        except TypeError:
            traceback.print_exc()
            _logger.exception(traceback.format_exc())
            print(self.history)
            exit(1)

        # print(f"normalize names: {list(normalize.iterrows())}")
        uct_score = {}
        # compute all the uct scores
        for t_name, row in normalize.iterrows():
            uct_score[t_name] = MultiBanditSearch.compute_uct(history=row,
                                                              total_run=total_run,
                                                              total_time=total_time)

        snames = [s.template.template["name"] for s in self.confSpaceBaseSearch]
        assert all([s in uct_score for s in snames]), \
            f"sname:{snames}, uct:{uct_score}"

        _logger.info(f"UCT updated: {uct_score}")
        choice_weight = [(s, uct_score[s.template.template["name"]])
                         for s in self.confSpaceBaseSearch]
        # choice_weight = [(s, uct_score[s]) for s in snames]

        _choices = [c for c, w in choice_weight]
        _weights = [w for c, w in choice_weight]

        return _choices, _weights

    def _push_random_candidates(self, num_iter: int):
        super()._push_random_candidates(num_iter)

    # def _search_templates(self, num_iter: int = 2) -> None:
    #     """
    #     runs the random search for each compatible template and returns the report of the best
    #     template evaluated. In each iteration the method randomly sample one of the templates
    #     from the template list based on their UCT score and runs random search on the template.
    #     Args:
    #         num_iter:
    #
    #     Returns:
    #
    #     """
    #     max_stalled_rounds = 3
    #     template_iter = self._select_next_template(num_iter=num_iter)
    #     for i, (search, mode) in enumerate(template_iter):
    #         templ_name = search.template.template['name']
    #         _logger.info(f"Using mode:({mode}), Selected Template: {templ_name}")
    #         self._random_pipeline_evaluation_push(search=search,
    #                                               num_iter=self.job_manager.proc_num)
    #         if self._stall_pushing(i, max_stalled_rounds):
    #             self._get_evaluation_results(max_num=self.job_manager.proc_num *
    #                                                  (max_stalled_rounds-1))
    #
    #         assert (self.job_manager.ongoing_jobs <=
    #                 self.job_manager.proc_num * max_stalled_rounds), \
    #             f"a lot of jobs in the queue, ongoing:{self.job_manager.ongoing_jobs}"
    #
    #     self._get_evaluation_results()

    # def _stall_pushing(self, i, max_stalled_rounds):
    #     proc_num = self.job_manager.proc_num
    #     ongoing = self.job_manager.ongoing_jobs
    #     return (ongoing >= proc_num*max_stalled_rounds)
        # return ((i > 0) and (i % max_stalled_rounds == 0)) or \
        #        (self.job_manager.ongoing_jobs >= self.job_manager.proc_num * max_stalled_rounds)

    @staticmethod
    def compute_uct(history: typing.Union[pd.Series, pd.DataFrame, typing.Dict],
                    total_time: float, total_run: float) -> float:
        beta = 10
        gamma = 1
        delta = 4

        try:
            return (beta * (history['cross_validation_metrics']) *
                    max(np.log(10 * history['sim_count']), 1) +
                    gamma * history['sim_count'] / np.sqrt(2 * np.log(total_run)) +
                    delta * np.sqrt(2 * np.log(total_time) / history['total_runtime']))
        except (KeyError, ZeroDivisionError):
            # _logger.error('Failed to compute UCT. Defaulting to None')
            # # print(STYLE+"[WARN] compute UCT failed:", history.tolist())
            # return None
            traceback.print_exc()
