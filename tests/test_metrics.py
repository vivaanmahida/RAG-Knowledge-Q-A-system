import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math

from src.metrics import (is_relevant, first_hit_only, recall_at_k,
                         reciprocal_rank, ndcg_at_k, average)


def test_is_relevant_ignores_case_and_spaces():
    assert is_relevant("The  CSS display property sets", "css display property")
    assert not is_relevant("The margin property", "css display property")


def test_recall_found_inside_k():
    assert recall_at_k([0, 0, 1, 0, 0], 3) == 1.0


def test_recall_found_outside_k():
    assert recall_at_k([0, 0, 0, 1, 0], 3) == 0.0


def test_reciprocal_rank_values():
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 0, 1]) == 1 / 3
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg_rank_one_is_perfect():
    assert ndcg_at_k([1, 0, 0, 0, 0], 5) == 1.0


def test_ndcg_rank_three():
    assert math.isclose(ndcg_at_k([0, 0, 1, 0, 0], 5), 1 / math.log2(4))


def test_ndcg_nothing_found_is_zero():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_second_hit_is_ignored():
    assert first_hit_only([0, 1, 1]) == [0, 1, 0]
    assert ndcg_at_k([1, 1, 0], 3) == 1.0


def test_average():
    assert average([1, 0, 1, 0]) == 0.5
    assert average([]) == 0.0
