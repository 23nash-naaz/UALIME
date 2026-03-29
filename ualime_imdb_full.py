"""
═══════════════════════════════════════════════════════════════════════════════
UALIME: Uncertainty-Aware LIME — Full Research Implementation
Dataset  : REAL IMDB Movie Reviews (HuggingFace datasets library)
Compares : LIME  vs  M-LIME  vs  UALIME
Metrics  : FSI, FSSI, ESI  (M-LIME paper — Fan & Li, 2024)
           AUC deletion curve (M-LIME paper)
           CI-Coverage, Stable-Feature-% (UALIME novel metrics)
Models   : Logistic Regression, Random Forest, AdaBoost
           (same lineup as M-LIME Table II)

HOW TO RUN:
  pip install scikit-learn numpy scipy matplotlib tqdm datasets
  python ualime_imdb_full.py

OUTPUT FILES (saved in same folder as script):
  ualime_research_results.png   — 6-panel comparison figure
  ualime_single_pos.png         — deep-dive on a positive review
  ualime_single_neg.png         — deep-dive on a negative review
  ualime_results.json           — all numeric results
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import warnings
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # works without a display / GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — change these to trade speed for quality
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_SIZE       = 2500    # how many IMDB training reviews to use  (max 25000)
TEST_SIZE        = 500     # how many test reviews to use
EXP_TEXTS_N      = 10      # reviews used for explanation experiments
N_STABILITY_RUNS = 10      # repetitions per text for FSI/FSSI/ESI
N_SAMPLES        = 300     # perturbation samples per explanation
N_FOCUSED        = 120     # extra focused samples in UALIME feedback loop
TOP_K            = 10      # features to return per explanation

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — REAL IMDB DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _clean_review(text: str) -> str:
    """Remove HTML tags and collapse whitespace (common in IMDB reviews)."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_imdb_data(train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE):
    """
    Downloads the real IMDB dataset via HuggingFace 'datasets' library.
    First run: ~80 MB download cached locally. Subsequent runs: instant.
    Returns: train_texts, train_labels, test_texts, test_labels
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "\n  'datasets' library not found.\n"
            "  Install it with:  pip install datasets\n"
        )

    print("  Loading IMDB dataset from HuggingFace (cached after first run)...")
    print("  (First run downloads ~80 MB. All future runs use local cache.)")
    try:
        ds = load_dataset("imdb")
    except Exception as e:
        raise ConnectionError(f"\n\n  Failed to load IMDB.\n  Ensure internet access and run: pip install datasets\n  Error: {e}")

    def _balanced(split, n):
        """Return n samples, always 50% positive / 50% negative."""
        half   = n // 2
        texts  = split["text"]
        labels = split["label"]
        pos_t = [_clean_review(t) for t, l in zip(texts, labels) if l == 1]
        neg_t = [_clean_review(t) for t, l in zip(texts, labels) if l == 0]
        rng   = np.random.RandomState(42)
        pi    = rng.permutation(len(pos_t))[:half]
        ni    = rng.permutation(len(neg_t))[:half]
        tx    = [pos_t[i] for i in pi] + [neg_t[i] for i in ni]
        lb    = [1] * half + [0] * half
        combined = list(zip(tx, lb))
        rng.shuffle(combined)
        tx, lb = zip(*combined)
        return list(tx), list(lb)

    train_texts, train_labels = _balanced(ds["train"], train_size)
    test_texts,  test_labels  = _balanced(ds["test"],  test_size)

    pos = sum(train_labels)
    neg = len(train_labels) - pos
    print(f"  Train : {len(train_texts):,} reviews  ({pos:,} positive / {neg:,} negative)")
    print(f"  Test  : {len(test_texts):,} reviews   ({sum(test_labels):,} pos / {len(test_labels)-sum(test_labels):,} neg)")
    return train_texts, train_labels, test_texts, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BLACK-BOX CLASSIFIERS  (LR, RF, AdaBoost)
# ─────────────────────────────────────────────────────────────────────────────

class BlackBoxModel:
    """
    TF-IDF + sklearn classifier.
    Matches the three black-box models used in M-LIME (Table II).
    """
    def __init__(self, model_type: str = "LR"):
        self.name = model_type
        if model_type == "LR":
            self.clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                          multi_class="auto")
        elif model_type == "RF":
            self.clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                              n_jobs=-1)
        elif model_type == "Adaboost":
            self.clf = AdaBoostClassifier(n_estimators=200, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.vec = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2
        )

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        acc = self.clf.score(X, labels)
        print(f"    [{self.name}] training accuracy: {acc:.3f}")
        return self

    def predict_proba(self, texts):
        """Returns P(positive) for each text. Core interface for explainers."""
        X = self.vec.transform(texts)
        return self.clf.predict_proba(X)[:, 1]

    def accuracy(self, texts, labels):
        preds = (self.predict_proba(texts) > 0.5).astype(int)
        return float(np.mean(np.array(preds) == np.array(labels)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Lowercase word tokenizer (no stopword removal — we want all tokens)."""
    return [w for w in re.findall(r"\b[a-z]+\b", text.lower()) if w]


def sentiment_vocab():
    """
    Sentiment word lists used for M-LIME / UALIME marginalization.
    Mimics what BERT-MLM would predict for masked sentiment words.
    """
    pos = [
        "good", "great", "excellent", "wonderful", "brilliant", "fantastic",
        "amazing", "superb", "outstanding", "magnificent", "perfect",
        "beautiful", "lovely", "enjoyable", "entertaining", "compelling",
        "powerful", "moving", "impressive", "masterful", "exceptional",
        "remarkable", "breathtaking", "unforgettable", "delightful",
    ]
    neg = [
        "bad", "terrible", "awful", "horrible", "dreadful", "boring",
        "poor", "weak", "disappointing", "mediocre", "dull", "tedious",
        "slow", "flat", "confusing", "painful", "wasted", "forgettable",
        "atrocious", "appalling", "pathetic", "laughable", "unbearable",
        "unwatchable", "pointless",
    ]
    neutral = [
        "the", "a", "an", "this", "that", "film", "movie", "story", "scene",
        "character", "performance", "director", "cast", "plot", "script",
        "acting", "cinematic", "narrative", "sequence", "moment",
    ]
    return pos, neg, neutral


POS_VOCAB, NEG_VOCAB, NEU_VOCAB = sentiment_vocab()


def replacement_distribution(token: str, sigma: float = 1e-4) -> dict:
    """
    Pseudo-MLM: probability distribution over replacement tokens.
    Positive tokens are replaced by other positive/neutral words.
    Negative tokens by negative/neutral words.
    This keeps perturbed samples in-distribution (M-LIME's core insight).
    Threshold truncation (sigma) follows M-LIME Eq. 5.
    """
    if token in POS_VOCAB:
        cands = POS_VOCAB + NEU_VOCAB
        raw = ([0.60 / len(POS_VOCAB)] * len(POS_VOCAB) +
               [0.40 / len(NEU_VOCAB)] * len(NEU_VOCAB))
    elif token in NEG_VOCAB:
        cands = NEG_VOCAB + NEU_VOCAB
        raw = ([0.60 / len(NEG_VOCAB)] * len(NEG_VOCAB) +
               [0.40 / len(NEU_VOCAB)] * len(NEU_VOCAB))
    else:
        cands = NEU_VOCAB + POS_VOCAB + NEG_VOCAB
        raw = ([0.50 / len(NEU_VOCAB)] * len(NEU_VOCAB) +
               [0.25 / len(POS_VOCAB)] * len(POS_VOCAB) +
               [0.25 / len(NEG_VOCAB)] * len(NEG_VOCAB))
    # Threshold truncation
    probs = np.clip(raw, sigma, None)
    probs /= probs.sum()
    return dict(zip(cands, probs.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIME  (Ribeiro et al., 2016 — baseline)
# ─────────────────────────────────────────────────────────────────────────────

class LIME:
    """
    Standard LIME.
    Perturbation: random binary masking (token present = 1, absent = 0).
    Surrogate: weighted Ridge regression.
    Problem: absent tokens become empty string → out-of-distribution.
    """
    name = "LIME"

    def __init__(self, n_samples: int = N_SAMPLES, top_k: int = TOP_K,
                 kernel_width: float = 0.25):
        self.n_samples    = n_samples
        self.top_k        = top_k
        self.kernel_width = kernel_width

    def _perturb(self, tokens: list):
        n  = len(tokens)
        bm = np.ones((self.n_samples + 1, n), dtype=np.float32)
        tx = [" ".join(tokens)]                     # original always first
        for i in range(1, self.n_samples + 1):
            mask    = np.random.randint(0, 2, n)
            bm[i]   = mask
            tx.append(" ".join(t for t, m in zip(tokens, mask) if m) or "neutral")
        return bm, tx

    def explain(self, text: str, predict_fn) -> dict:
        tokens = tokenize(text)
        if not tokens:
            return {}
        bm, tx = self._perturb(tokens)
        preds  = predict_fn(tx)

        # Cosine-kernel weights (distance from original)
        sims    = cosine_similarity(bm[0:1], bm)[0]
        weights = np.exp(-((1 - sims) ** 2) / (2 * self.kernel_width ** 2))

        reg = Ridge(alpha=1.0)
        reg.fit(bm, preds, sample_weight=weights)

        coefs = sorted(zip(tokens, reg.coef_.tolist()),
                       key=lambda x: abs(x[1]), reverse=True)
        return dict(coefs[:self.top_k])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — M-LIME  (Fan & Li, 2024)
# ─────────────────────────────────────────────────────────────────────────────

class MLIME:
    """
    M-LIME — Marginalized LIME.
    Key improvement over LIME:
      - Masked tokens replaced by contextually probable words (not deleted)
      - Keeps perturbed samples in-distribution (fixes OOD problem)
      - Scores features with Weight of Evidence (WoE, Eq. 1)
      - Threshold truncation σ reduces compute (Eq. 5)
    Reference: Fan & Li, RICAI 2024.
    """
    name = "M-LIME"

    def __init__(self, n_samples: int = N_SAMPLES, top_k: int = TOP_K,
                 sigma: float = 1e-4, lam: float = 40.0):
        self.n_samples = n_samples
        self.top_k     = top_k
        self.sigma     = sigma
        self.lam       = lam

    def _perturb(self, tokens: list):
        n  = len(tokens)
        bm = np.ones((self.n_samples + 1, n), dtype=np.float32)
        tx = [" ".join(tokens)]
        for i in range(1, self.n_samples + 1):
            n_mask = np.random.randint(1, max(2, n // 2 + 1))
            idx    = np.random.choice(n, size=n_mask, replace=False)
            sample = tokens.copy()
            row    = np.ones(n)
            for j in idx:
                dist        = replacement_distribution(tokens[j], self.sigma)
                sample[j]   = np.random.choice(list(dist.keys()),
                                                p=list(dist.values()))
                row[j]      = 0           # 0 = original token replaced
            bm[i] = row
            tx.append(" ".join(sample))
        return bm, tx

    @staticmethod
    def _woe(p_with: float, p_without: float, eps: float = 1e-9) -> float:
        """Weight of Evidence — M-LIME Eq. 1."""
        odds_w  = p_with    / (1 - p_with    + eps)
        odds_wo = p_without / (1 - p_without + eps)
        return float(np.log2(odds_w + eps) - np.log2(odds_wo + eps))

    def explain(self, text: str, predict_fn) -> dict:
        tokens = tokenize(text)
        if not tokens:
            return {}
        bm, tx = self._perturb(tokens)
        preds  = predict_fn(tx)
        p_orig = float(preds[0])

        scores = {}
        for j, tok in enumerate(tokens):
            masked      = np.where(bm[1:, j] == 0)[0]
            p_without   = float(np.mean(preds[1:][masked])) if len(masked) > 0 else p_orig
            scores[tok] = self._woe(p_orig, p_without)

        sorted_s = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_s[:self.top_k])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — UALIME  (this work)
# ─────────────────────────────────────────────────────────────────────────────

class UALIME:
    """
    UALIME — Uncertainty-Aware LIME.

    Novel contributions on top of M-LIME and tree-ALIME:

    1. ADAPTIVE EXPLOSION
       Initial perturbations use M-LIME-style marginalization (not random deletion).

    2. ENTROPY/VARIANCE GATE + FEEDBACK LOOP
       After initial sampling, output variance across perturbations is measured.
       If variance > threshold, a focused second wave of perturbations targets
       the most uncertain tokens specifically (not random), then loops again.
       This is the "Adaptive Explosion → Measurement → Feedback Loop" path
       described in the problem statement flowchart.

    3. COSINE SAMPLE WEIGHTS  (from ALIME / tree-ALIME)
       Perturbed samples are weighted by cosine similarity to the original,
       so nearby samples in feature space have more influence on the surrogate.

    4. BAYESIAN CORE  (BayLIME / Destruction & Rebirth)
       Bayesian Ridge Regression replaces OLS. This gives a full posterior
       distribution over feature weights, from which 95% credible intervals
       are computed. Tokens whose CI crosses zero are flagged as noise.

    5. DECISION TREE SURROGATE  (tree-ALIME, Ranjbar & Safabakhsh 2022)
       A decision tree is fitted on the top-k Bayesian features, providing
       the human-interpretable output that tree-ALIME showed outperforms
       linear models in user studies (Table III of that paper).

    6. STABILITY MAP
       Each token receives: mean weight, posterior std, 95% CI bounds,
       and a binary "stable" flag. This is the novel output not present
       in LIME or M-LIME.

    7. ESI SELF-CONSISTENCY CHECK
       ESI is computed by splitting samples into two halves and comparing
       the resulting explanations (M-LIME Eq. 9), giving a built-in
       reliability score for each explanation.
    """
    name = "UALIME"

    def __init__(self, n_initial: int = N_SAMPLES,
                 n_focused: int = N_FOCUSED,
                 variance_threshold: float = 0.04,
                 max_loops: int = 3,
                 sigma: float = 1e-4,
                 lam: float = 40.0,
                 top_k: int = TOP_K,
                 tree_depth: int = 5,
                 kernel_width: float = 0.25):
        self.n_initial   = n_initial
        self.n_focused   = n_focused
        self.var_thresh  = variance_threshold
        self.max_loops   = max_loops
        self.sigma       = sigma
        self.lam         = lam
        self.top_k       = top_k
        self.tree_depth  = tree_depth
        self.kw          = kernel_width
        self._last_meta  = {}   # stores full stability map, ESI, loop count

    # ── Perturbation batch (M-LIME style marginalization) ────────────────────
    def _make_batch(self, tokens: list, n: int,
                    targeted: list = None) -> tuple:
        nt   = len(tokens)
        pool = targeted if targeted else list(range(nt))
        bm, tx = [], []
        for _ in range(n):
            k      = np.random.randint(1, max(2, len(pool) // 2 + 1))
            idx    = np.random.choice(pool, size=min(k, len(pool)), replace=False)
            sample = tokens.copy()
            row    = np.ones(nt)
            for j in idx:
                dist      = replacement_distribution(tokens[j], self.sigma)
                sample[j] = np.random.choice(list(dist.keys()),
                                              p=list(dist.values()))
                row[j]    = 0
            bm.append(row)
            tx.append(" ".join(sample))
        return np.array(bm, dtype=np.float32), tx

    # ── Sample weights (ALIME cosine kernel) ─────────────────────────────────
    def _weights(self, bm: np.ndarray) -> np.ndarray:
        orig  = np.ones((1, bm.shape[1]))
        sims  = cosine_similarity(orig, bm)[0]
        dists = 1 - sims
        return np.exp(-(dists ** 2) / (2 * self.kw ** 2))

    # ── Uncertainty gate ──────────────────────────────────────────────────────
    def _high_variance(self, preds: np.ndarray) -> bool:
        return float(np.var(preds)) > self.var_thresh

    def _uncertain_token_idx(self, bm: np.ndarray,
                              preds: np.ndarray, k: int = 5) -> list:
        corrs = []
        for j in range(bm.shape[1]):
            col = bm[:, j]
            c   = abs(float(np.corrcoef(col, preds)[0, 1])) if col.std() > 0 else 0.0
            corrs.append(c)
        return list(np.argsort(corrs)[-k:])

    # ── Bayesian Ridge (posterior weights + credible intervals) ──────────────
    def _bayes_fit(self, bm: np.ndarray, preds: np.ndarray,
                   sw: np.ndarray) -> tuple:
        br = BayesianRidge(compute_score=True, fit_intercept=True)
        br.fit(bm, preds, sample_weight=sw)
        means = br.coef_
        stds  = np.sqrt(np.diag(br.sigma_))
        return means, stds

    # ── Decision tree surrogate (tree-ALIME) ─────────────────────────────────
    def _tree_fit(self, bm: np.ndarray, preds: np.ndarray,
                  tokens: list, sw: np.ndarray, idx: list) -> dict:
        X    = bm[:, idx]
        y    = (preds > 0.5).astype(int)
        tree = DecisionTreeClassifier(max_depth=self.tree_depth, random_state=42)
        tree.fit(X, y, sample_weight=sw)
        return {tokens[i]: float(imp)
                for i, imp in zip(idx, tree.feature_importances_)}

    # ── ESI (M-LIME Eq. 9) ───────────────────────────────────────────────────
    def _esi(self, wa: dict, wb: dict) -> float:
        common    = set(wa) & set(wb)
        same_sign = {f for f in common if wa[f] * wb[f] >= 0}
        k         = max(len(wa), len(wb), 1)
        return sum(np.exp(-self.lam * abs(wa[f] - wb[f]))
                   for f in same_sign) / k

    # ── Main explain ──────────────────────────────────────────────────────────
    def explain(self, text: str, predict_fn) -> dict:
        tokens = tokenize(text)
        if not tokens:
            return {}
        nt = len(tokens)

        # ── Step 1: Initial adaptive explosion ───────────────────────────────
        bm, tx  = self._make_batch(tokens, self.n_initial)
        all_tx  = [" ".join(tokens)] + tx
        all_pr  = predict_fn(all_tx)
        orig_pr = float(all_pr[0])
        preds   = all_pr[1:]

        # ── Step 2: Entropy/variance gate + feedback loop ────────────────────
        n_loops = 0
        for _ in range(self.max_loops):
            if not self._high_variance(preds):
                break
            unc  = self._uncertain_token_idx(bm, preds)
            bm2, tx2 = self._make_batch(tokens, self.n_focused, unc)
            pr2  = predict_fn(tx2)
            bm   = np.vstack([bm, bm2])
            preds = np.concatenate([preds, pr2])
            n_loops += 1

        # ── Step 3: Cosine weights ────────────────────────────────────────────
        sw = self._weights(bm)

        # ── Step 4: Bayesian core → credible intervals ────────────────────────
        means, stds = self._bayes_fit(bm, preds, sw)

        # ── Step 5: Top-k features by |Bayesian weight| ───────────────────────
        top_idx = list(np.argsort(np.abs(means))[-self.top_k:])

        # ── Step 6: Decision tree surrogate (tree-ALIME) ──────────────────────
        tree_imp = self._tree_fit(bm, preds, tokens, sw, top_idx)

        # ── Step 7: Build stability map ───────────────────────────────────────
        sm = {}
        for i, tok in enumerate(tokens):
            w, s   = float(means[i]), float(stds[i])
            lo, hi = w - 1.96 * s, w + 1.96 * s
            sm[tok] = {
                "weight": w, "std": s,
                "ci_lower": lo, "ci_upper": hi,
                "stable": not (lo < 0 < hi),
            }

        # ── Step 8: ESI self-consistency ─────────────────────────────────────
        h          = len(preds) // 2
        m_a, _     = self._bayes_fit(bm[:h], preds[:h], sw[:h])
        m_b, _     = self._bayes_fit(bm[h:], preds[h:], sw[h:])
        esi_score  = self._esi(
            {t: float(m_a[i]) for i, t in enumerate(tokens)},
            {t: float(m_b[i]) for i, t in enumerate(tokens)},
        )

        self._last_meta = {
            "n_loops":        n_loops,
            "n_samples":      len(preds),
            "esi":            esi_score,
            "stability_map":  sm,
            "tree_imp":       tree_imp,
            "var_final":      float(np.var(preds)),
            "orig_pred":      orig_pr,
        }

        sorted_w = sorted(sm.items(), key=lambda x: abs(x[1]["weight"]), reverse=True)
        return {tok: v["weight"] for tok, v in sorted_w[:self.top_k]}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATION METRICS  (exact formulas from M-LIME paper)
# ─────────────────────────────────────────────────────────────────────────────

def jaccard(a: dict, b: dict) -> float:
    """M-LIME Eq. 6 — set overlap of feature lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def compute_fsi(runs: list) -> float:
    """
    Feature Stability Index — M-LIME Eq. 7.
    Average Jaccard over all pairs of neighbouring explanation runs.
    Higher = more stable.
    """
    pairs = [(runs[i], runs[i+1]) for i in range(len(runs)-1)]
    return float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else 0.0


def compute_fssi(ea: dict, eb: dict) -> float:
    """
    Feature Sequence Stability Index — M-LIME Eq. 8.
    Checks same feature appears at same rank with same sign.
    """
    ka = list(ea.keys())
    kb = list(eb.keys())
    k  = max(len(ka), len(kb), 1)
    score = 0
    for i in range(min(len(ka), len(kb))):
        if ka[i] == kb[i] and ea[ka[i]] * eb.get(kb[i], 0) >= 0:
            score += 1
    return score / k


def compute_esi(ea: dict, eb: dict, lam: float = 40.0) -> float:
    """
    Explanation Stability Index — M-LIME Eq. 9.
    Penalises weight magnitude differences between runs in addition to set mismatch.
    """
    common    = set(ea) & set(eb)
    same_sign = {f for f in common if ea[f] * eb[f] >= 0}
    k         = max(len(ea), len(eb), 1)
    return sum(np.exp(-lam * abs(ea[f] - eb[f])) for f in same_sign) / k


def compute_auc(text: str, explanation: dict,
                predict_fn, orig_pred: float) -> float:
    """
    AUC of deletion curve — M-LIME Eq. 10 (trapezoidal rule).
    Progressively removes top-importance tokens and tracks confidence.
    Lower AUC = better (explanation correctly identified important tokens).
    """
    tokens  = tokenize(text)
    order   = list(explanation.keys())   # sorted by importance already
    scores  = [orig_pred]
    removed = set()
    for feat in order:
        removed.add(feat)
        remaining = [t for t in tokens if t not in removed] or ["neutral"]
        scores.append(float(predict_fn([" ".join(remaining)])[0]))
    t_vals = np.linspace(0, 1, len(scores))
    trap_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap_fn(scores, t_vals))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EXPERIMENT RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_stability(texts: list, predict_fn,
                  n_runs: int = N_STABILITY_RUNS,
                  n_samples: int = N_SAMPLES) -> dict:
    """
    Replicates M-LIME Table II stability experiment.
    Runs each explainer n_runs times per text and measures FSI/FSSI/ESI.
    """
    methods = {
        "LIME":   LIME(n_samples=n_samples, top_k=TOP_K),
        "M-LIME": MLIME(n_samples=n_samples, top_k=TOP_K),
        "UALIME": UALIME(n_initial=n_samples, n_focused=N_FOCUSED, top_k=TOP_K),
    }
    out = {m: {"fsi": [], "fssi": [], "esi": []} for m in methods}

    for text in tqdm(texts, desc="  Stability", ncols=70):
        for mname, method in methods.items():
            runs = [method.explain(text, predict_fn) for _ in range(n_runs)]
            out[mname]["fsi"].append(compute_fsi(runs))
            fssi_s, esi_s = [], []
            for i in range(len(runs)-1):
                fssi_s.append(compute_fssi(runs[i], runs[i+1]))
                esi_s.append(compute_esi(runs[i], runs[i+1]))
            out[mname]["fssi"].append(float(np.mean(fssi_s)))
            out[mname]["esi"].append(float(np.mean(esi_s)))

    return {m: {k: float(np.mean(v)) for k, v in vv.items()}
            for m, vv in out.items()}


def run_effectiveness(texts: list, predict_fn,
                      n_samples: int = N_SAMPLES) -> dict:
    """
    Replicates M-LIME Fig. 2 effectiveness experiment.
    Lower AUC = better (important tokens identified correctly).
    """
    methods = {
        "LIME":   LIME(n_samples=n_samples, top_k=TOP_K),
        "M-LIME": MLIME(n_samples=n_samples, top_k=TOP_K),
        "UALIME": UALIME(n_initial=n_samples, n_focused=N_FOCUSED, top_k=TOP_K),
    }
    out = {m: [] for m in methods}

    for text in tqdm(texts, desc="  Effectiveness", ncols=70):
        orig = float(predict_fn([text])[0])
        for mname, method in methods.items():
            exp = method.explain(text, predict_fn)
            out[mname].append(compute_auc(text, exp, predict_fn, orig))

    return {m: float(np.mean(v)) for m, v in out.items()}


def run_ualime_specific(texts: list, predict_fn,
                        n_samples: int = N_SAMPLES) -> dict:
    """
    UALIME-only novel metrics:
      - Stable-feature %: tokens whose 95% CI doesn't cross zero
      - Mean posterior uncertainty (std of Bayesian weights)
      - Average number of feedback loop iterations triggered
    """
    u = UALIME(n_initial=n_samples, n_focused=N_FOCUSED, top_k=TOP_K, max_loops=3)
    stable_pcts, uncerts, loops = [], [], []

    for text in tqdm(texts, desc="  UALIME-specific", ncols=70):
        u.explain(text, predict_fn)
        sm = u._last_meta.get("stability_map", {})
        if sm:
            stable_pcts.append(np.mean([v["stable"] for v in sm.values()]) * 100)
            uncerts.append(np.mean([v["std"] for v in sm.values()]))
        loops.append(u._last_meta.get("n_loops", 0))

    return {
        "stable_pct":       float(np.mean(stable_pcts)),
        "mean_uncertainty": float(np.mean(uncerts)),
        "avg_loops":        float(np.mean(loops)),
        "per_text":         stable_pcts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"LIME": "#5F5E5A", "M-LIME": "#534AB7", "UALIME": "#1D9E75"}
METHOD_NAMES = ["LIME", "M-LIME", "UALIME"]
BAR_COLORS   = [COLORS[m] for m in METHOD_NAMES]


def _style_ax(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_facecolor("#F7F6F2")
    ax.spines[["top", "right"]].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


def _bar_panel(ax, vals, title, ylabel, ylim=None, annotate=True):
    bars = ax.bar(METHOD_NAMES, vals, color=BAR_COLORS,
                  width=0.5, edgecolor="white", linewidth=1.2)
    _style_ax(ax, title, ylabel, ylim)
    if annotate:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] * 0.01 if ylim else 0.01),
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MAIN RESEARCH FIGURE  (6-panel + summary table)
# ─────────────────────────────────────────────────────────────────────────────

def make_research_figure(stab: dict, eff: dict,
                         ual: dict, sample_meta: dict,
                         model_name: str, save_path: str):
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor("#FAFAF8")
    gs  = GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.42)

    # Panel 1 — FSI
    ax1 = fig.add_subplot(gs[0, 0])
    _bar_panel(ax1, [stab[m]["fsi"] for m in METHOD_NAMES],
               f"FSI — Feature Stability Index\n(IMDB · {model_name})",
               "FSI ↑", ylim=(0, 1.08))
    ax1.axhline(0.95, color="gray", lw=0.8, ls="--", alpha=0.6, label="M-LIME paper")
    ax1.legend(fontsize=7)

    # Panel 2 — FSSI
    ax2 = fig.add_subplot(gs[0, 1])
    _bar_panel(ax2, [stab[m]["fssi"] for m in METHOD_NAMES],
               f"FSSI — Feature Sequence Stability\n(IMDB · {model_name})",
               "FSSI ↑", ylim=(0, 1.08))

    # Panel 3 — ESI
    ax3 = fig.add_subplot(gs[0, 2])
    _bar_panel(ax3, [stab[m]["esi"] for m in METHOD_NAMES],
               f"ESI — Explanation Stability Index\n(IMDB · {model_name})",
               "ESI ↑", ylim=(0, 1.08))

    # Panel 4 — AUC deletion curve
    ax4 = fig.add_subplot(gs[1, 0])
    _bar_panel(ax4, [eff[m] for m in METHOD_NAMES],
               f"AUC Deletion Curve\n(IMDB · {model_name} · lower = better)",
               "AUC ↓")

    # Panel 5 — UALIME novel metrics
    ax5 = fig.add_subplot(gs[1, 1])
    labels5 = ["Stable\nfeature %", "Feedback\nloops ×20"]
    vals5   = [ual["stable_pct"], ual["avg_loops"] * 20]
    bars5   = ax5.bar(labels5, vals5,
                      color=[COLORS["UALIME"], "#BA7517"],
                      width=0.45, edgecolor="white", linewidth=1.2)
    _style_ax(ax5, "UALIME novel metrics\n(not present in LIME or M-LIME)", "Value")
    ax5.text(0, vals5[0] + 0.5, f"{ual['stable_pct']:.1f}%",
             ha="center", fontsize=9, fontweight="bold")
    ax5.text(1, vals5[1] + 0.5, f"{ual['avg_loops']:.2f} loops",
             ha="center", fontsize=9, fontweight="bold")

    # Panel 6 — Per-token credible intervals (sample explanation)
    ax6 = fig.add_subplot(gs[1, 2])
    sm   = sample_meta.get("stability_map", {})
    toks = sorted(sm, key=lambda t: abs(sm[t]["weight"]), reverse=True)[:8]
    if toks:
        y    = np.arange(len(toks))
        ws   = [sm[t]["weight"]   for t in toks]
        lo   = [sm[t]["ci_lower"] for t in toks]
        hi   = [sm[t]["ci_upper"] for t in toks]
        stbl = [sm[t]["stable"]   for t in toks]
        cols = [COLORS["UALIME"] if s else "#D85A30" for s in stbl]
        errs = [[w - l for w, l in zip(ws, lo)],
                [h - w for w, h in zip(ws, hi)]]
        ax6.barh(y, ws, xerr=errs, color=cols, alpha=0.82,
                 error_kw={"ecolor": "gray", "capsize": 3, "elinewidth": 0.8})
        ax6.axvline(0, color="black", lw=0.8, ls="--")
        ax6.set_yticks(y);  ax6.set_yticklabels(toks, fontsize=9)
        ax6.set_xlabel("Bayesian weight (mean ± 95% CI)", fontsize=9)
        p1 = mpatches.Patch(color=COLORS["UALIME"], alpha=0.82, label="Stable signal")
        p2 = mpatches.Patch(color="#D85A30",        alpha=0.82, label="Noise (CI ∋ 0)")
        ax6.legend(handles=[p1, p2], fontsize=8)
    _style_ax(ax6, "UALIME credible intervals\n(sample review explanation)",
              "Feature weight")

    # Panel 7 — Summary comparison table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    cols_h = ["Method", "FSI ↑", "FSSI ↑", "ESI ↑", "AUC ↓",
              "Stable CI % ↑", "Avg loops", "Novel output"]
    rows = [
        ["LIME",
         f"{stab['LIME']['fsi']:.4f}",
         f"{stab['LIME']['fssi']:.4f}",
         f"{stab['LIME']['esi']:.4f}",
         f"{eff['LIME']:.4f}",
         "N/A", "N/A", "—"],
        ["M-LIME  (Fan & Li 2024)",
         f"{stab['M-LIME']['fsi']:.4f}",
         f"{stab['M-LIME']['fssi']:.4f}",
         f"{stab['M-LIME']['esi']:.4f}",
         f"{eff['M-LIME']:.4f}",
         "N/A", "N/A", "—"],
        ["UALIME  (ours)",
         f"{stab['UALIME']['fsi']:.4f}",
         f"{stab['UALIME']['fssi']:.4f}",
         f"{stab['UALIME']['esi']:.4f}",
         f"{eff['UALIME']:.4f}",
         f"{ual['stable_pct']:.1f}%",
         f"{ual['avg_loops']:.2f}",
         "✓ Bayesian CI bounds"],
    ]
    tbl = ax7.table(cellText=rows, colLabels=cols_h,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.4)
    for j in range(len(cols_h)):
        tbl[0, j].set_facecolor("#2C2C2A")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    row_palette = ["#E8E8E6", "#EEEDFE", "#E1F5EE"]
    for i, rc in enumerate(row_palette):
        for j in range(len(cols_h)):
            tbl[i+1, j].set_facecolor(rc)
            if i == 2:
                tbl[i+1, j].set_text_props(fontweight="bold")
    ax7.set_title(
        f"Table — LIME vs M-LIME vs UALIME on IMDB ({model_name})   "
        "↑ higher is better  ·  ↓ lower is better",
        fontsize=10, fontweight="bold", pad=10
    )

    fig.suptitle(
        "UALIME: Uncertainty-Aware LIME — Research Results on Real IMDB Dataset",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="#FAFAF8")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — SINGLE REVIEW DEEP-DIVE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def visualise_single(text: str, predict_fn,
                     model_name: str, save_path: str):
    """
    Three-panel side-by-side explanation of one IMDB review.
    Left: LIME  ·  Centre: M-LIME  ·  Right: UALIME with CI and ✓/✗ flags.
    """
    lime_m   = LIME(n_samples=400, top_k=10)
    mlime_m  = MLIME(n_samples=400, top_k=10)
    ualime_m = UALIME(n_initial=400, n_focused=150, top_k=10)

    exp_l  = lime_m.explain(text, predict_fn)
    exp_m  = mlime_m.explain(text, predict_fn)
    exp_u  = ualime_m.explain(text, predict_fn)
    meta   = ualime_m._last_meta
    sm     = meta.get("stability_map", {})

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#FAFAF8")
    short_text = (text[:90] + "…") if len(text) > 90 else text
    fig.suptitle(
        f"Single explanation — {model_name}\n\"{short_text}\"",
        fontsize=10, fontweight="bold"
    )

    def _panel(ax, exp, title, color, show_ci=False):
        toks = list(exp.keys())[:10]
        vals = [exp[t] for t in toks]
        y    = np.arange(len(toks))
        cols = [color if v >= 0 else "#D85A30" for v in vals]

        if show_ci and sm:
            errs = [
                [abs(sm[t]["weight"] - sm[t]["ci_lower"]) if t in sm else 0 for t in toks],
                [abs(sm[t]["ci_upper"] - sm[t]["weight"]) if t in sm else 0 for t in toks],
            ]
            ax.barh(y, vals, color=cols, alpha=0.82, xerr=errs,
                    error_kw={"ecolor": "gray", "capsize": 3, "elinewidth": 0.8})
            for yi, tok in enumerate(toks):
                if tok in sm:
                    flag = "✓" if sm[tok]["stable"] else "✗"
                    fc   = color if sm[tok]["stable"] else "#D85A30"
                    max_v = max(abs(v) for v in vals) if vals else 0.1
                    ax.text(max_v * 1.15, yi, flag,
                            va="center", fontsize=12, color=fc, fontweight="bold")
        else:
            ax.barh(y, vals, color=cols, alpha=0.82)

        ax.set_yticks(y)
        ax.set_yticklabels(toks, fontsize=10)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Feature importance", fontsize=9)
        ax.set_facecolor("#F7F6F2")
        ax.spines[["top", "right"]].set_visible(False)

    _panel(axes[0], exp_l, "LIME (baseline)\nRandom deletion → OOD",     COLORS["LIME"])
    _panel(axes[1], exp_m, "M-LIME (Fan & Li, 2024)\nMLM marginalization", COLORS["M-LIME"])
    _panel(axes[2], exp_u, "UALIME (ours)\nBayesian CI · ✓ stable · ✗ noise",
           COLORS["UALIME"], show_ci=True)

    orig = meta.get("orig_pred", predict_fn([text])[0])
    sent = "POSITIVE" if orig > 0.5 else "NEGATIVE"
    fig.text(0.5, 0.01,
             f"Prediction: {sent}  (conf: {orig:.3f})  ·  "
             f"Feedback loops: {meta.get('n_loops', 0)}  ·  "
             f"ESI: {meta.get('esi', 0):.4f}  ·  "
             f"Stable features: {sum(1 for v in sm.values() if v['stable'])}/{len(sm)}",
             ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="#FAFAF8")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("═" * 72)
    print("  UALIME Research Experiment — Real IMDB Dataset")
    print("═" * 72)

    # ── Load real IMDB data ───────────────────────────────────────────────────
    train_texts, train_labels, test_texts, test_labels = load_imdb_data()

    # Pick EXP_TEXTS_N reviews balanced pos/neg for explanation experiments
    pos_idx = [i for i, l in enumerate(test_labels) if l == 1][:EXP_TEXTS_N // 2]
    neg_idx = [i for i, l in enumerate(test_labels) if l == 0][:EXP_TEXTS_N // 2]
    exp_texts = [test_texts[i] for i in pos_idx + neg_idx]
    print(f"\n  Using {len(exp_texts)} test reviews for explanation experiments")
    print(f"  ({len(pos_idx)} positive + {len(neg_idx)} negative)\n")

    all_results = {}

    # ── Loop over three black-box models (mirrors M-LIME Table II) ────────────
    for model_type in ["LR", "RF", "Adaboost"]:
        print(f"{'─'*72}")
        print(f"  Black-box model: {model_type}")
        print(f"{'─'*72}")

        bb = BlackBoxModel(model_type)
        bb.fit(train_texts, train_labels)
        test_acc = bb.accuracy(test_texts[:200], test_labels[:200])
        print(f"    [{model_type}] test accuracy (first 200): {test_acc:.3f}")
        predict_fn = bb.predict_proba

        print(f"\n  [1/3] Stability experiment ({N_STABILITY_RUNS} runs × {len(exp_texts)} texts)…")
        stab = run_stability(exp_texts, predict_fn,
                             n_runs=N_STABILITY_RUNS, n_samples=N_SAMPLES)

        print(f"\n  [2/3] Effectiveness experiment (AUC deletion curve)…")
        eff = run_effectiveness(exp_texts, predict_fn, n_samples=N_SAMPLES)

        print(f"\n  [3/3] UALIME-specific metrics (CI coverage, feedback loops)…")
        ual = run_ualime_specific(exp_texts, predict_fn, n_samples=N_SAMPLES)

        # Sample explanation for the credible-interval panel
        sample_u = UALIME(n_initial=N_SAMPLES, n_focused=N_FOCUSED, top_k=TOP_K)
        sample_u.explain(exp_texts[0], predict_fn)

        all_results[model_type] = {
            "stability":    stab,
            "effectiveness": eff,
            "ualime":       ual,
            "sample_meta":  sample_u._last_meta,
        }

        # ── Per-model console table ───────────────────────────────────────────
        print(f"\n  Results — {model_type}  (IMDB dataset)")
        print(f"  {'Method':<24}  {'FSI':>8}  {'FSSI':>8}  {'ESI':>8}  {'AUC':>8}")
        print(f"  {'─'*60}")
        for m in METHOD_NAMES:
            print(f"  {m:<24}  "
                  f"{stab[m]['fsi']:>8.4f}  "
                  f"{stab[m]['fssi']:>8.4f}  "
                  f"{stab[m]['esi']:>8.4f}  "
                  f"{eff[m]:>8.4f}")
        print(f"\n  UALIME stable-feature %  : {ual['stable_pct']:.1f}%")
        print(f"  UALIME mean uncertainty  : {ual['mean_uncertainty']:.4f}")
        print(f"  UALIME avg feedback loops: {ual['avg_loops']:.2f}")

    # ── Generate all figures using LR (primary model) ─────────────────────────
    print(f"\n{'─'*72}")
    print("  Generating research figures (LR model)…")

    bb_lr = BlackBoxModel("LR")
    bb_lr.fit(train_texts, train_labels)
    predict_lr = bb_lr.predict_proba
    r = all_results["LR"]

    make_research_figure(
        r["stability"], r["effectiveness"], r["ualime"],
        r["sample_meta"], model_name="LR",
        save_path="ualime_research_results.png"
    )

    # Pick one real positive and one real negative review for deep-dive
    first_pos = next(t for t, l in zip(test_texts, test_labels) if l == 1)
    first_neg = next(t for t, l in zip(test_texts, test_labels) if l == 0)

    visualise_single(first_pos, predict_lr,
                     model_name="LR · positive review",
                     save_path="ualime_single_pos.png")
    visualise_single(first_neg, predict_lr,
                     model_name="LR · negative review",
                     save_path="ualime_single_neg.png")

    # ── Save all numeric results ───────────────────────────────────────────────
    with open("ualime_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  ✓ Saved: ualime_results.json")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  COMPLETE — output files created in this folder:")
    print("    ualime_research_results.png  — 6-panel figure + summary table")
    print("    ualime_single_pos.png        — deep-dive on a positive review")
    print("    ualime_single_neg.png        — deep-dive on a negative review")
    print("    ualime_results.json          — all numeric results (reproducible)")
    print("═" * 72)


if __name__ == "__main__":
    main()