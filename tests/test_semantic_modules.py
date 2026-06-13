import torch

from ServiceComputing.algorithms.slg_sage_mappo import SLGSAGEActorCritic


def test_semantic_prior_and_auxiliary_losses_backpropagate():
    model = SLGSAGEActorCritic(obs_dim=43, semantic_dim=10, action_dim=6, n_agents=8, hidden_dim=32)
    obs = torch.rand(16, 43)
    sem = obs[:, -10:]
    mean, std, prior, aux = model.policy(obs, sem)
    assert mean.shape == (16, 6)
    assert prior.shape == (16, 6)
    prior_loss = torch.nn.functional.mse_loss(mean, prior.detach())
    aux_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(aux["success_logit"], torch.ones(16))
        + torch.nn.functional.binary_cross_entropy_with_logits(aux["deadline_logit"], torch.zeros(16))
        + torch.nn.functional.mse_loss(aux["delay"], torch.rand(16))
    )
    loss = prior_loss + aux_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.semantic_encoder.parameters())

