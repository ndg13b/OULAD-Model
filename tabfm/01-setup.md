# Setup, and the three things that will block you

Everything here was checked against the packages actually installed in this
repo on **22 November 2026**. These projects move fast — re-check the version
numbers if you are reading this much later.

```
tabpfn   10.2.0     default model TabPFN-3.5
tabicl    2.2.0     default checkpoint tabicl-classifier-v2-20260212
torch     2.15.0
```

---

## Install

```bash
pip install tabpfn tabicl
```

That is genuinely all, but there is a trap worth knowing about.

**On a machine with no GPU, this installs ~5 GB of CUDA libraries you cannot
use.** `pip install torch` on Linux pulls the CUDA build by default — sixteen
`nvidia-*` packages. It works (CPU inference is fine), it is just enormous. To
avoid it, install CPU-only PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tabpfn tabicl
```

Check what you ended up with:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

---

## Blocker 1: the weights are gated

Neither package ships its model weights. They download from Hugging Face on
first `fit()`, and **TabPFN's repository is gated** — you must have an account
and have accepted the terms before the download will work.

If you have not, `fit()` raises `TabPFNHuggingFaceGatedRepoError`:

```
HuggingFace authentication error downloading from 'Prior-Labs/tabpfn_3_5'.
This model is gated and requires you to accept its terms.
```

To fix:

1. Make a Hugging Face account if you do not have one.
2. Visit <https://huggingface.co/Prior-Labs/tabpfn_3_5> and accept the terms.
3. Create a **read** token at <https://huggingface.co/settings/tokens>.
4. Either run `hf auth login`, or set the token in your environment:

```bash
export HF_TOKEN=hf_...
```

In a notebook, set it before the first `fit()`:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."   # do not commit this
```

**TabICL appears not to be gated.** Its checkpoint comes from `jingang/TabICL`
and the download starts without asking for credentials. So if you want to try a
tabular foundation model with the least friction, start there.

Weights are cached after the first download — `~/.cache/tabpfn/` and
`~/.cache/huggingface/` — so this is a one-time cost per machine.

## Blocker 2: it really wants a GPU

TabPFN enforces a hard cap on training rows when running on CPU. Read straight
from the installed package:

```python
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import cpu_sample_limit
{m.value: cpu_sample_limit(m) for m in ModelVersion}
```

| Model version | CPU row cap |
|---|---:|
| v2, v2.5, v2.6 | **1,000** |
| v3, v3.5, v3.5-fast | **5,000** |

With a GPU the documented ceiling for TabPFN-3.5 is **1,000,000 rows and 20,000
features** — three orders of magnitude more. The CPU cap is the binding
constraint for most people trying this on a laptop.

Exceeding it raises rather than silently degrading. Three ways out, in order of
preference:

1. **Subsample.** Usually fine, and honest. `tabfm_bench.sample_oulad` does it
   with the grouping this project requires.
2. **`ignore_pretraining_limits=True`.** Pushes past the guardrail. It will be
   slow and the authors do not promise the quality holds.
3. **`TABPFN_ALLOW_CPU_LARGE_DATASET=true`.** The environment-variable version
   of the same override.

For OULAD at day 28 — 27,538 rows — option 1 is the only sane choice on CPU.

## Blocker 3: the good weights are non-commercial

| Component | Licence |
|---|---|
| TabPFN **code** | Apache 2.0 |
| TabPFN-2 **weights** | Prior Labs License (Apache 2.0 + attribution) |
| TabPFN-2.5, 2.6, 3, 3.5 **weights** | **Non-commercial** |

So the default model you get from `TabPFNClassifier()` cannot be used
commercially without a licence from Prior Labs. For learning, coursework or
research this is irrelevant. For anything that ships, it is a conversation to
have before you build on it — and `ModelVersion.V2` is the fallback with
permissive weights.

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
```

Note the CPU cap drops to 1,000 rows for v2.

---

## Running it in a cloud session

If you are working in Claude Code on the web rather than locally, there is a
fourth blocker: **the default environment's network policy blocks
huggingface.co**, so the weights cannot be downloaded at all. The failure looks
like this:

```
ProxyError: 403 Forbidden (agent proxy: no rule or network setting allows this host)
```

An environment named **`OULAD-Model`** has been created for this repo with the
needed hosts allowed:

```
huggingface.co          cdn-lfs.huggingface.co
cdn-lfs-us-1.hf.co      transfer.xethub.hf.co
docs.priorlabs.ai       ux.priorlabs.ai
christophmolnar.com
```

Pick it from the environment menu when starting a session. You will also want
to add `download.pytorch.org` to that list if you want the CPU-only torch
install, and set `HF_TOKEN` as an environment variable or network secret.

---

## The API, in one screen

Both packages are scikit-learn compatible, so they drop into a `Pipeline` or
`cross_val_score` unchanged.

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(
    device="cpu",                        # or "cuda", or "auto"
    categorical_features_indices=[43],   # column positions, not names
    ignore_pretraining_limits=False,
    n_estimators="auto",
    random_state=0,
)
clf.fit(X_train, y_train)                # stores context; near-instant
proba = clf.predict_proba(X_test)[:, 1]  # this is where the time goes
```

```python
from tabicl import TabICLClassifier

clf = TabICLClassifier(device="cpu", random_state=0)
```

**Three rules that are the opposite of normal practice:**

1. **Do not scale.** No `StandardScaler`, no `MinMaxScaler`. The model does its
   own normalisation and expects the distributions it was pretrained on.
2. **Do not one-hot encode.** Pass categorical columns as they are and tell the
   model which they are via `categorical_features_indices`.
3. **Do not impute.** NaN is handled natively and carries information. Filling
   it throws that away.

**One more:** predict on the whole test set in a single call. Looping row by row
re-runs the context encoding every time and is dramatically slower.

### Choosing a model version

```python
import os
os.environ["TABPFN_MODEL_VERSION"] = "v3.5-fast"   # before import

# or explicitly:
from tabpfn.constants import ModelVersion
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)
```

Accepted: `v2`, `v2.5`, `v2.6`, `v3`, `v3.5`, `v3.5-fast`.

---

## Checking it works

```bash
python tabfm/10-first-contact.py
```

Runs in seconds on a small built-in dataset. If the weights are unavailable it
reports exactly why and still runs the baselines, so you can tell a setup
problem from a model problem.
