# Predictive Maintenance: Remaining Useful Life (RUL) Prediction

Predicting the **Remaining Useful Life** of aircraft engines from multivariate sensor
time series (NASA C-MAPSS FD001), and — the part that matters most for a factory —
evaluating the model against the **cost structure** of predictive maintenance rather than
raw error alone.

> **English** · [한국어 README](./README.ko.md)

---

## 1. Problem

Given a stream of engine sensor readings, predict how many operating cycles remain before
failure. This is a **regression** problem (predict a number), not classification. It
matters because catching a failure *before* it happens lets you schedule maintenance and
avoid unplanned downtime, secondary damage, and safety incidents — the core promise of
predictive maintenance.

The twist that runs through the whole project: **the two kinds of mistake do not cost the
same.** Predicting *too much* life left (a *late* warning) is far more dangerous than
predicting *too little* (an *early* warning). RMSE cannot see that difference; a factory
can. Handling that asymmetry is the heart of this project.

## 2. Data

- **Source:** NASA C-MAPSS Turbofan Engine Degradation dataset, subset **FD001** (single
  operating condition, single fault mode).
- **Structure:** 100 run-to-failure engines in the training set (20,631 rows). Each row is
  one engine at one cycle, with 3 operating settings and 21 sensor channels. Training
  engines run **to failure**; test engines are **truncated** before failure, with the true
  remaining life given separately in `RUL_FD001.txt`.
- **Engine life** ranges from 128 to 362 cycles (mean ≈ 206).
- **Preprocessing:** dropped 6 constant, non-informative sensors (`s1, s5, s10, s16, s18,
  s19`), leaving 15; normalized to [0, 1] with statistics fit on **train only**; clipped
  RUL at **125**.

**Why clip RUL at 125?** Early in an engine's life there is no wear signal in the sensors,
so forcing the model to output "190 cycles left" from healthy readings is impossible and
meaningless. Clipping treats "plenty of life remaining" as a single healthy ceiling. It is
a modeling decision grounded in the physics, not a trick.

## 3. Time-series preprocessing — sliding windows

A single cycle's snapshot cannot reveal a *trend*. The signal in a time series is in *how
values move*, so the last **N = 30** cycles are grouped into one input, turning the data
into a 3-D tensor of shape `[samples, window length, sensors]`.

The one rule that governs correctness: **windows never cross an engine boundary.** Mixing
the tail of one engine with the head of another would fabricate a trajectory that never
happened and corrupt the series, so windows slide *within each engine only*. Individual
windows can then be shuffled freely for training, because the time order *inside* each
window is already fixed.

For the test set, each engine's **last** window is its single sample, and engines are
iterated in sorted order so predictions line up with the rows of `RUL_FD001.txt`.

## 4. Model

A **1-D CNN** (`Conv1d`) slides small filters along the **time axis** to pick up local
degradation patterns (e.g. a temperature ramp). One subtlety drives the implementation:
PyTorch's `Conv1d` expects `[batch, channels, length]` and treats **each sensor as a
channel** and **time as the length**, so the forward pass swaps the last two axes with
`permute` before convolving. `AdaptiveAvgPool1d` then collapses time to one value per
channel, and a linear layer outputs a single RUL value. The loss is `MSELoss` — this is
regression, so there is no sigmoid/softmax; the network outputs a number directly.

The training loop is the same `zero_grad → forward → loss → backward → step` used across
every supervised project; only the model (sequence) and the loss (MSE) change.

## 5. Evaluation — RMSE + asymmetric cost

Two numbers, read **together**:

- **RMSE — 22.24 cycles.** On average the baseline is off by about 22 cycles. Against an
  engine life of ~200 cycles this is a usable first model, not a strong one. RMSE is
  symmetric: a +15 error and a −15 error count the same.
- **C-MAPSS standard score — 4566.** The field-standard score that penalizes
  **over-prediction (late)** harder than under-prediction (early), via `exp(d/10)−1` for
  late errors versus `exp(−d/13)−1` for early ones. Lower is better.

**Error-direction analysis (the key insight).** Defining `error = predicted − true`:

- `error > 0` → over-prediction → maintenance delayed → risk of in-service failure.
  **Dangerous.**
- `error < 0` → under-prediction → early maintenance on a healthy part → wasted part, but
  **safe.**

The baseline over-predicts on **66 of 100** engines (mean +18 cycles) versus 34
under-predictions (mean −12). In other words the model leans toward the **risky, late**
side — exactly the direction a maintenance engineer would least want, and the motivation
for the next step.

## 6. Improvement experiments

Changing one thing at a time (evaluated on the test set for a compact comparison; strictly,
selection should happen on validation):

| Experiment | RMSE | C-MAPSS score |
|---|---:|---:|
| Baseline (window = 30, MSE) | 22.24 | 4566 |
| Window = 20 | 20.77 | 2798 |
| Window = 50 | 24.82 | 6160 |
| Dropout = 0.3 | 21.17 | 2764 |
| **Asymmetric loss (2×)** | **20.74** | **2192** |

A **shorter window (20)** and **dropout** both help modestly; a longer window (50) hurts,
so more context is not automatically better. The largest gain comes from the asymmetric
loss.

### Asymmetric loss — trading a metric for real-world cost

Plain MSE is symmetric, but the loss itself can be made asymmetric so the model learns to
avoid the dangerous *late* prediction from the start. A custom loss weights over-prediction
errors **2×**. The effect:

| | Late / risky | Early / safe | RMSE | C-MAPSS |
|---|---:|---:|---:|---:|
| Baseline (MSE) | 66 | 34 | 22.24 | 4566 |
| Asymmetric loss | 47 | 53 | 20.74 | 2192 |

The error distribution shifts toward the **safe** side (risky predictions drop from 66 to
47), and the C-MAPSS score more than halves. Here RMSE happened to improve too, but the
decision would be justified even if RMSE had risen slightly: in predictive maintenance,
cutting the count of dangerous late warnings is worth a small penalty on a symmetric error
metric. **Optimizing the objective that matches the cost — instead of chasing the headline
metric — is the point.**

## 7. What I learned

- How to turn a raw time series into a model-ready tensor with **engine-boundary-aware
  sliding windows**, and why that boundary rule is non-negotiable.
- How a **1-D CNN** reads a sequence, including the channel/time axis swap that trips up
  most first implementations.
- How to evaluate regression with a **domain cost function** (C-MAPSS asymmetric score)
  alongside RMSE, and to read the **direction** of a model's errors, not just their size.
- What to do when the **metric and the real cost disagree**: a small RMSE sacrifice can be
  the correct engineering choice when it removes dangerous late predictions.

**Limitations / next steps.** FD001 is the easiest subset (one operating condition);
FD002/FD004 add regime complexity. Natural extensions: an LSTM comparison, per-engine early
stopping, and tuning the over-penalty on the validation set rather than the test set.

---

## Repository

```
.
├── predictive_maintenance_rul.ipynb        # full pipeline, executed with plots (English)
├── README.md         # this file
├── README.ko.md      # Korean version
└── CMAPSSData/       # place the three FD001 .txt files here (not committed)
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

## Reproduce

1. Download the C-MAPSS dataset (NASA Prognostics Data Repository, or Kaggle "NASA
   CMAPSS") and put `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in `CMAPSSData/`.
2. `pip install torch numpy pandas scikit-learn matplotlib`
3. Run `코드.ipynb` top to bottom. All results and figures are reproducible with a fixed
   seed (42).
