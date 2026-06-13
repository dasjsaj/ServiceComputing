import numpy as np
import torch

from ServiceComputing.scripts.analyze_semantic_contribution import attribute_semantic_source
from ServiceComputing.scripts.train_slg_sage_mappo import set_global_seed


def test_set_global_seed_makes_numpy_and_torch_initialization_reproducible():
    set_global_seed(17)
    numpy_a = np.random.rand(3)
    torch_a = torch.rand(3)

    set_global_seed(17)
    numpy_b = np.random.rand(3)
    torch_b = torch.rand(3)

    assert np.allclose(numpy_a, numpy_b)
    assert torch.allclose(torch_a, torch_b)


def test_attribute_semantic_source_detects_incremental_semantic_contribution():
    summaries = {
        "Control-NoSemantic": {"tail_eval_return": 8.0, "tail_completion_ratio": 0.40},
        "Semantic-StateOnly": {"tail_eval_return": 9.0, "tail_completion_ratio": 0.42},
        "SLG-SAGE-Full": {"tail_eval_return": 11.0, "tail_completion_ratio": 0.44},
    }

    decision = attribute_semantic_source(summaries)

    assert decision["classification"] == "semantic_state_and_loss_supported"
    assert decision["full_gain_over_control_return"] == 3.0
    assert decision["full_gain_over_stateonly_return"] == 2.0
