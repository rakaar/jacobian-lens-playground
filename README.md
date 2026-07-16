# Jacobian Lens Playground

This notebook builds and validates a residual-stream Jacobian for
`Qwen/Qwen2.5-1.5B` in two independent ways:

1. row-by-row with PyTorch autograd and TransformerLens hooks;
2. with Anthropic's reference [`jlens`](https://github.com/anthropics/jacobian-lens)
   implementation.

The comparison uses the same prompt, model weights, source block, target block,
tokenization, and penultimate token position. It also checks the handwritten
Jacobian with a central finite difference along a random direction.

## What the matrix means

For the included run, each matrix has shape `[1536, 1536]`:

```text
row i, column j = d(target feature i) / d(source feature j)
```

The source is the penultimate-token residual after block 14, and the target is
the penultimate-token residual after block 27.

## Verified results

The saved run produced:

```text
Finite-difference cosine:       0.9999995
Handwritten Jacobian norm:      61.596313
Anthropic Jacobian norm:        61.596310
Matrix cosine (float64):        0.999999999991
Matrix relative L2 error:       4.17e-6
Matrix maximum absolute error:  6.20e-6
```

The notebook recomputes cosine similarity, norm ratio, relative L2 error, and
maximum absolute error using float64 reductions for a stable comparison.

## Run it

Create an environment with Python 3.10 or newer, then install the dependencies:

```bash
python -m pip install -r requirements.txt
jupyter lab tf_lens_play.ipynb
```

The notebook uses float32 deliberately. Finite differences through bfloat16
forward passes can be dominated by activation quantization. A CUDA GPU is
strongly recommended; the full Jacobian requires many retained-graph backward
passes.

## Relation to a fitted J-lens

This notebook configures Anthropic's estimator to include only the penultimate
position so it can be compared directly with the handwritten matrix. A normal
fitted J-lens averages its transport matrices over multiple valid sequence
positions and many prompts, then uses the model's final norm and unembedding to
produce vocabulary readouts.
