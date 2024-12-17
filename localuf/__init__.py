"""
localuf (Local Union--Find)
========

A Python package to simulate and visualise decoders for CSS codes.

Available code classes:
* Surface
* Repetition

Available modules:
* decoders
* plot
* sim

Available functions:
* get_failure_data
* get_log_runtime_data
* get_stats
* get_failure_data_from_subset_sample
* add_ignored_timesteps
"""


# print(f'Invoking __init__.py for {__name__}')
# below correct (see https://github.com/psf/requests/blob/v2.23.0/requests/__init__.py#L112)
from localuf.codes import Repetition, Surface
from localuf import decoders
from localuf import plot
from localuf import sim
from localuf.data_processors import get_failure_data, get_log_runtime_data, \
    get_stats, get_failure_data_from_subset_sample, add_ignored_timesteps