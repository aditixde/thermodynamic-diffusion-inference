# Thermodynamic Diffusion Inference with Minimal Digital Conditioning

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)  
**Author:** Aditi De

> *We build analog chips that turn waste heat into AI inference.*

Diffusion-model inference and overdamped Langevin dynamics are formally identical. A physical substrate encoding the score function equilibrates to the correct output by thermodynamics alone — no digital arithmetic during inference, with a theoretical **10⁷× energy reduction** over GPU inference.

This repository reproduces all experiments and figures from the paper.

## Key Results

| Configuration | Decoder Cosine Similarity | Parameters | Energy Gain |
|---|---|---|---|
| Oracle (ceiling) | 1.0000 | — | ~10⁷× |
| Skip only (trained weights) | 0.9924 | 256 | — |
| **Full pipeline (ours)** | **0.9906** | **2,560** | **~10⁷×** |

## What This Paper Solves

Two fundamental barriers blocked thermodynamic inference at production scale ([Jelinčič et al., 2025](https://arxiv.org/abs/2510.23972)):

1. **Non-local skip connections** — U-Net skip connections require O(D²) wiring in analog substrates. Our *hierarchical bilinear coupling* reduces this to O(Dk) using the low-rank singular structure of trained Gram matrices.

2. **Input conditioning signal deficit** — Coupling constants carry ~2,600× too little signal to distinguish inputs. Our *minimal digital interface* (2,560 parameters = 0.032% of U-Net) overcomes this barrier.

## Reproduction

### Quick Start (GPU recommended, CPU works)

```bash
git clone https://github.com/YOUR_USERNAME/thermodynamic-diffusion-inference.git
cd thermodynamic-diffusion-inference
pip install -r requirements.txt
python run_experiments.py
```

This runs all three experiments and saves figures to `figures/`.

### Individual Steps

```bash
# Run experiments only (prints results to console)
python run_experiments.py --no-figures

# Generate figures from saved results
python run_experiments.py --figures-only

# Train on real MNIST data (default uses random activations for speed)
python run_experiments.py --train-mnist --epochs 2

# Nonlinear equilibration validation (Appendix)
python run_experiments.py --nonlinear
```

### Google Colab

Open `notebook.ipynb` in Colab — runs everything in sequence with a T4 GPU.

## Repository Structure

```
├── run_experiments.py       # Complete reproduction script
├── notebook.ipynb           # Colab notebook (self-contained)
├── requirements.txt
├── LICENSE
├── figures/                 # Generated figures (PDF + PNG)
│   ├── fig1_architecture.*
│   ├── fig2_skip_trained.*
│   ├── fig3_conditioning_sweep.*
│   ├── fig4_production_test.*
│   └── figA_conditioning_failure.*
└── paper/
    └── main.tex             # Paper source
```

## Experiments

- **Experiment A (Section 3):** Skip coupling effect — measures decoder shift ρ_skip across ranks 2–64
- **Experiment B (Section 4):** Conditioning interface sweep — linear and MLP encoders across bottleneck dimensions k = 4–128
- **Experiment C (Section 5):** Full production test — oracle, learned encoder, skip-only, and full pipeline compared

## Citation

```bibtex
@article{de2026thermodynamic,
  title={Thermodynamic Diffusion Inference with Minimal Digital Conditioning},
  author={De, Aditi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
