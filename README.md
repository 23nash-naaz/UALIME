# UALIME

UALIME (Uncertainty-Aware LIME) is an explanation framework for black-box text models that improves on standard LIME by making perturbations more in-distribution and by reporting uncertainty for each feature.

It combines ideas from:
- M-LIME: BERT-based marginalization to avoid out-of-distribution perturbations
- tree-ALIME: decision-tree surrogate explanations for human readability
- BayLIME: Bayesian regression with credible intervals

## What this project contains

- [ualime_core.py](ualime_core.py): core implementation of the UALIME pipeline
- [ualime_demo.py](ualime_demo.py): small demo that runs a mock sentiment classifier and visualizes the explanation
- [ualime_imdb_full.py](ualime_imdb_full.py): fuller research-style experiment using the IMDB dataset
- [requirements.txt](requirements.txt): Python dependencies
- Generated outputs such as PNG charts and JSON results

## Features

- Adaptive perturbation generation using masked language model marginalization
- Uncertainty-aware feedback loops for focused resampling
- Bayesian Ridge regression with credible intervals
- Decision-tree surrogate explanations for interpretable outputs
- Stability metrics and visualization dashboards

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

For the full IMDB research script, also install:

```bash
pip install datasets tqdm
```

## Quick start demo

Run the built-in demo:

```bash
python ualime_demo.py
```

This script uses a small mock text classifier and produces:
- a terminal explanation report
- a stability map figure saved as [ualime_stability_map.png](ualime_stability_map.png)

## Full research experiment

To run the IMDB-based experiment:

```bash
python ualime_imdb_full.py
```

This script:
- downloads the IMDB dataset from Hugging Face
- trains several black-box classifiers
- compares LIME, M-LIME, and UALIME-style explanations
- saves results to files such as [ualime_results.json](ualime_results.json)

## Using UALIME with your own model

The main class in [ualime_core.py](ualime_core.py) accepts a callable prediction function that returns probabilities for a list of texts.

Example structure:

```python
from ualime_core import UALIME


def llm_predict_fn(texts):
    # Replace this with your own model or API call
    return [0.82, 0.21, 0.91]

explainer = UALIME(llm_predict_fn=llm_predict_fn)
result = explainer.explain("This movie is fantastic")
print(result["feature_weights"])
```

## Notes

- The demo is intended as a lightweight smoke test.
- The full IMDB script is more computationally expensive and may take longer depending on your hardware and internet connection.
- For best results, use a real text model or API endpoint instead of the mock example.

## License

This project is provided as a research/demo implementation and is intended for educational and experimental use.
