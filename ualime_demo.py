"""
UALIME Dashboard + Demo
Shows the stability map, credible intervals, and tree importances.
Run: python ualime_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ualime_core import UALIME


# ─────────────────────────────────────────────────────────────
# Stability Map Dashboard (the "Final State" in your flowchart)
# ─────────────────────────────────────────────────────────────

def plot_stability_map(result: dict, title: str = "UALIME Stability Map"):
    """
    Three-panel dashboard:
      Left:   Feature weights with 95% credible intervals (error bars)
      Center: Uncertainty per token (std of posterior weight)
      Right:  Decision tree feature importances (tree-ALIME)
    Tokens with stable CI (doesn't cross zero) are highlighted in teal.
    Unstable tokens (CI crosses zero) are shown in coral — signal vs noise.
    """
    stability_map = result["stability_map"]
    tree_imp = result["tree_importances"]
    esi = result["esi_score"]

    tokens = list(stability_map.keys())
    weights = [stability_map[t]["weight"] for t in tokens]
    uncertainties = [stability_map[t]["uncertainty"] for t in tokens]
    ci_lowers = [stability_map[t]["ci_lower"] for t in tokens]
    ci_uppers = [stability_map[t]["ci_upper"] for t in tokens]
    stables = [stability_map[t]["stable"] for t in tokens]
    errors = [[w - l for w, l in zip(weights, ci_lowers)],
              [u - w for w, u in zip(weights, ci_uppers)]]
    colors = ["#1D9E75" if s else "#D85A30" for s in stables]

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(tokens) * 0.35 + 1)))
    fig.suptitle(f"{title}\nESI Score: {esi:.4f}  |  Samples: {result['n_total_samples']}  |  Feedback loops: {result['n_feedback_loops']}",
                 fontsize=12, fontweight="bold")

    # Panel 1: Weights + Credible Intervals
    ax = axes[0]
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, weights, xerr=errors, color=colors, alpha=0.8,
            error_kw={"ecolor": "gray", "capsize": 3, "elinewidth": 1})
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Bayesian weight (mean ± 95% CI)")
    ax.set_title("Feature weights")
    stable_patch = mpatches.Patch(color="#1D9E75", alpha=0.8, label="Stable signal")
    noise_patch = mpatches.Patch(color="#D85A30", alpha=0.8, label="Noise (CI crosses 0)")
    ax.legend(handles=[stable_patch, noise_patch], fontsize=8)

    # Panel 2: Uncertainty (posterior std)
    ax2 = axes[1]
    ax2.barh(y_pos, uncertainties, color="#534AB7", alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tokens, fontsize=9)
    ax2.set_xlabel("Posterior uncertainty (std)")
    ax2.set_title("Token uncertainty")

    # Panel 3: Decision tree importances (tree-ALIME)
    ax3 = axes[2]
    tree_tokens = list(tree_imp.keys())
    tree_vals = list(tree_imp.values())
    t_y = np.arange(len(tree_tokens))
    ax3.barh(t_y, tree_vals, color="#BA7517", alpha=0.8)
    ax3.set_yticks(t_y)
    ax3.set_yticklabels(tree_tokens, fontsize=9)
    ax3.set_xlabel("Decision tree importance")
    ax3.set_title("Tree surrogate (tree-ALIME)")

    plt.tight_layout()
    plt.savefig("ualime_stability_map.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Stability map saved to ualime_stability_map.png")


def print_stability_report(result: dict):
    """Terminal-friendly report."""
    print("\n" + "═" * 60)
    print("  UALIME EXPLANATION REPORT")
    print("═" * 60)
    print(f"  Samples used     : {result['n_total_samples']}")
    print(f"  Feedback loops   : {result['n_feedback_loops']}")
    print(f"  ESI score        : {result['esi_score']:.4f}  (1.0 = perfectly stable)")
    print("─" * 60)
    print(f"  {'Token':<20}  {'Weight':>8}  {'Uncert':>8}  {'95% CI':>22}  {'Signal?'}")
    print("─" * 60)
    sm = result["stability_map"]
    sorted_tokens = sorted(sm.keys(), key=lambda t: abs(sm[t]["weight"]), reverse=True)
    for t in sorted_tokens:
        v = sm[t]
        ci = f"[{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}]"
        flag = "✓ stable" if v["stable"] else "✗ noisy"
        print(f"  {t:<20}  {v['weight']:>+8.4f}  {v['uncertainty']:>8.4f}  {ci:>22}  {flag}")
    print("─" * 60)
    print("\n  Top tree-ALIME features:")
    ti = result["tree_importances"]
    for t, imp in sorted(ti.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {t:<20}  {imp:.4f}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# Demo: Sentiment classification with a mock LLM
# ─────────────────────────────────────────────────────────────

def demo_mock_llm():
    """
    Quick smoke-test using a bag-of-words sentiment mock.
    Replace `mock_sentiment_predict` with your actual LLM call
    (e.g., OpenAI API, HuggingFace pipeline, local model).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    # Train a tiny sentiment classifier as a stand-in for the LLM
    train_texts = [
        "I love this movie it is fantastic",
        "great film wonderful acting",
        "absolutely terrible awful waste",
        "boring dull horrible film",
        "amazing brilliant superb",
        "worst movie ever so bad",
    ]
    train_labels = [1, 1, 0, 0, 1, 0]
    vec = CountVectorizer()
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=500).fit(X_train, train_labels)

    def mock_llm_predict(texts: list) -> list:
        X = vec.transform(texts)
        return list(clf.predict_proba(X)[:, 1])

    # ── Run UALIME ────────────────────────────────────────────
    test_text = "this movie is absolutely wonderful and fantastic"

    explainer = UALIME(
        llm_predict_fn=mock_llm_predict,
        mlm_model="bert-base-uncased",
        sentence_encoder=None,           # set to SentenceTransformer("all-MiniLM-L6-v2") for autoencoder weighting
        n_initial_samples=80,            # reduce for demo speed; use 300+ in production
        n_focused_samples=40,
        variance_threshold=0.03,
        max_feedback_loops=2,
        top_k_features=8,
    )

    print(f"\nExplaining: '{test_text}'")
    result = explainer.explain(test_text, verbose=True)
    print_stability_report(result)
    plot_stability_map(result, title=f"UALIME — \"{test_text}\"")
    return result


if __name__ == "__main__":
    result = demo_mock_llm()