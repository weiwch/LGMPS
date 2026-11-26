from __future__ import annotations

import torch

from lgmps.training.checkpoint import ResumeState, load_checkpoint, save_checkpoint


def test_full_checkpoint_restores_training_state(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = tmp_path / "resume.pt"
    state = ResumeState(2, 7, 11, 1.25, 1)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        config={"name": "test"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mode="full",
    )
    assert restored == state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
