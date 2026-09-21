"""Verify the shipped numerical results without training or network access."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'results/core/nonlinear_representations'))
import canonical_analysis as analysis


def main():
    torch.set_num_threads(2)
    frame = analysis.load_frame()
    expected = json.loads((ROOT / 'figures/canonical_analysis.json').read_text())
    assert len(frame) == 216
    assert frame[['n', 'm', 'l', 'S']].drop_duplicates().shape[0] == 216
    for row in frame.itertuples():
        model = analysis.load_model(row.n, row.m, row.l, row.S)
        with torch.no_grad():
            reconstructed, code = model(torch.zeros(2, row.n))
        assert reconstructed.shape == (2, row.n) and code.shape == (2, row.m)
        assert all(torch.isfinite(p).all() for p in model.parameters())
    fits = {str(depth): analysis.fit_depth(frame[frame.l == depth]) for depth in range(1, 5)}
    for depth, fit in fits.items():
        for key in ('A', 'log_mse_r2'):
            assert np.isclose(fit[key], expected['fits'][depth][key], rtol=1e-6, atol=1e-8)
        for sparsity, coefficient in fit['c'].items():
            assert np.isclose(coefficient, expected['fits'][depth]['c'][sparsity], rtol=1e-6, atol=1e-8)

    picture = json.loads((ROOT / 'figures/untied_read_write_example.json').read_text())
    generator = torch.Generator().manual_seed(picture['test_seed'])
    count, n = picture['test_samples'], picture['n']
    inputs = (torch.rand(count, n, generator=generator) > picture['sparsity']).float() * torch.rand(count, n, generator=generator)
    losses = {}
    with torch.no_grad():
        for name in ('trained_tied', 'trained_untied'):
            saved = picture['models'][name]
            encoder, decoder, bias = (torch.tensor(saved[key]) for key in ('encoder', 'decoder', 'bias'))
            prediction = torch.relu((inputs @ encoder.T) @ decoder.T + bias)
            measured = float((prediction - inputs).square().mean())
            assert np.isclose(measured, saved['mse'], rtol=5e-5, atol=1e-8), (name, measured, saved['mse'])
            losses[name] = measured
    assert losses['trained_untied'] < losses['trained_tied']
    print(json.dumps({'sweep_configurations': len(frame), 'compatible_checkpoints': len(frame),
                      'scaling_fits': fits, 'pictured_model_mse': losses,
                      'untied_improvement_percent': 100 * (1 - losses['trained_untied'] / losses['trained_tied']),
                      'source_status': analysis.SOURCE['status']}, indent=2))


if __name__ == '__main__':
    main()
