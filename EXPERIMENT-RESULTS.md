# Tutorial 12 Experiment Results

All runs used the Tiny Shakespeare corpus, the seed `ROMEO:`, CPU execution, and the same model code. The loss is training cross-entropy; lower values mean the model fit the training next-character predictions better. Generated snippets are stochastic and are examples, not a quality benchmark.

## Controlled parameter comparisons

| Run | Epochs | Sequence length | Hidden units | Loss by epoch | Final loss |
|---|---:|---:|---:|---|---:|
| Baseline | 2 | 100 | 128 | 2.4001 → 1.8805 | 1.8805 |
| Shorter sequence | 2 | 60 | 128 | 2.2333 → 1.7856 | 1.7856 |
| Fewer hidden units | 2 | 100 | 64 | 2.7051 → 2.1525 | 2.1525 |
| More epochs | 4 | 100 | 128 | 2.4001 → 1.8805 → 1.7344 → 1.6544 | 1.6544 |

The 10-epoch baseline run completed separately with loss decreasing from **2.4001** to **1.4869**. Its full epoch losses were 2.4001, 1.8805, 1.7344, 1.6544, 1.6035, 1.5678, 1.5413, 1.5198, 1.5016, and 1.4869.

## Generated text samples

These short excerpts come from the controlled two- and four-epoch runs:

**Baseline, 2 epochs:**

> My dit he seefforck tie as him a curpse; my doost in we cour'd the so kings to him;

**Sequence length 60:**

> Word my hist the good begema confies of the a graines leaves and have me cannot your hand firth'd

**64 hidden units:**

> Whe lult seace thet, and bunt: I cham to henow and what chat ind your oft blengh one

**Baseline, 4 epochs:**

> Good to my fight, That many being deal day him piople.

## Interpretation

- Increasing training from 2 to 4 and then 10 epochs steadily reduced the training loss in these runs. The generated text also began to contain more recognizable Shakespeare-like structure, though it remains noisy.
- With 64 hidden units, the two-epoch loss was higher than the 128-unit baseline. The smaller model has less capacity in this comparison.
- Sequence length 60 produced a lower two-epoch loss than length 100 in this run. This one result does not establish that 60 is generally better; changing sequence length also changes how training text is divided into examples.
- Compare loss and samples together. Training loss alone does not measure how coherent or original the generated passage is.

## Reproduce

Use the commands in `run_experiments.ps1`. It saves a terminal log per configuration under `experiment-results/`.
