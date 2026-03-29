"""
UALIME: Uncertainty-Aware LIME
Combines insights from:
  - M-LIME (Fan & Li, 2024): BERT marginalization to fix OOD & instability
  - tree-ALIME (Ranjbar & Safabakhsh, 2022): Decision tree for human-interpretable output
  - BayLIME (Zhao et al., 2021): Bayesian core for credible intervals
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import logit
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. Adaptive Explosion (Uncertainty-Guided Engine)
#    Based on M-LIME: BERT marginalization replaces random deletion
# ─────────────────────────────────────────────────────────────

class AdaptiveExplosion:
    """
    Generates perturbed text samples via BERT-based marginalization.
    Instead of blindly zeroing tokens (which creates OOD samples),
    we replace masked tokens with their contextually probable substitutes.
    This is the core fix from M-LIME.
    """

    def __init__(self, mlm_model_name: str = "bert-base-uncased", sigma: float = 1e-4):
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
        self.mlm = AutoModelForMaskedLM.from_pretrained(mlm_model_name)
        self.mlm.eval()
        self.sigma = sigma  # M-LIME threshold truncation: skip candidates p < sigma

    def marginalize_token(self, tokens: list[str], mask_idx: int) -> dict[str, float]:
        """
        For a single masked position, return a probability distribution over
        candidate replacement tokens (truncated at self.sigma).
        Returns {token_str: probability}
        """
        masked = tokens.copy()
        masked[mask_idx] = self.tokenizer.mask_token
        input_ids = self.tokenizer(
            " ".join(masked), return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.mlm(**input_ids).logits
        mask_pos = (input_ids["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]
        probs = torch.softmax(logits[0, mask_pos], dim=-1).numpy()

        # Threshold truncation (M-LIME Eq. 5): only keep candidates above sigma
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        filtered = {inv_vocab[i]: float(probs[i]) for i in np.where(probs > self.sigma)[0]}
        total = sum(filtered.values())
        return {k: v / total for k, v in filtered.items()}  # renormalize

    def generate_samples(
        self,
        text: str,
        n_samples: int = 150,
        targeted_indices: list[int] = None
    ) -> tuple[np.ndarray, list[str]]:
        """
        Returns:
            binary_matrix: (n_samples, n_tokens) — 1 = token present, 0 = replaced
            perturbed_texts: list of perturbed strings
        """
        tokens = text.split()
        n_tokens = len(tokens)
        mask_set = list(range(n_tokens)) if targeted_indices is None else targeted_indices

        binary_matrix = np.ones((n_samples, n_tokens), dtype=np.float32)
        perturbed_texts = []

        for i in range(n_samples):
            sample_tokens = tokens.copy()
            n_mask = np.random.randint(1, max(2, n_tokens // 2))
            masked_indices = np.random.choice(mask_set, size=min(n_mask, len(mask_set)), replace=False)
            for idx in masked_indices:
                cands = self.marginalize_token(tokens, idx)
                if cands:
                    replacement = np.random.choice(list(cands.keys()), p=list(cands.values()))
                    sample_tokens[idx] = replacement
                    binary_matrix[i, idx] = 0  # mark as "replaced / absent"
            perturbed_texts.append(" ".join(sample_tokens))

        return binary_matrix, perturbed_texts

    def compute_token_entropy(self, text: str) -> np.ndarray:
        """
        Compute per-token entropy from BERT marginalization distributions.
        High entropy = BERT is uncertain about what goes there = the token is "noisy".
        This feeds the uncertainty gate.
        """
        tokens = text.split()
        entropies = []
        for i in range(len(tokens)):
            cands = self.marginalize_token(tokens, i)
            probs = list(cands.values())
            entropies.append(entropy(probs))
        return np.array(entropies)


# ─────────────────────────────────────────────────────────────
# 2. Uncertainty Measurement & Feedback Loop
# ─────────────────────────────────────────────────────────────

class UncertaintyGate:
    """
    Measures the LLM's output variance across perturbed samples.
    If variance is above threshold, triggers a focused re-explosion
    targeting the most uncertain token positions.
    """

    def __init__(self, variance_threshold: float = 0.05, max_iterations: int = 3):
        self.variance_threshold = variance_threshold
        self.max_iterations = max_iterations

    def compute_output_variance(self, predictions: np.ndarray) -> float:
        """Variance of predicted probabilities across perturbations."""
        return float(np.var(predictions))

    def identify_uncertain_tokens(
        self,
        binary_matrix: np.ndarray,
        predictions: np.ndarray,
        top_k: int = 5
    ) -> list[int]:
        """
        Tokens whose presence/absence correlates most with high output variance.
        These become the 'focused explosion' targets.
        """
        n_tokens = binary_matrix.shape[1]
        correlations = []
        for j in range(n_tokens):
            col = binary_matrix[:, j]
            if col.std() > 0:
                corr = float(np.abs(np.corrcoef(col, predictions)[0, 1]))
            else:
                corr = 0.0
            correlations.append(corr)
        return list(np.argsort(correlations)[-top_k:])

    def should_resample(self, predictions: np.ndarray) -> bool:
        return self.compute_output_variance(predictions) > self.variance_threshold


# ─────────────────────────────────────────────────────────────
# 3. Bayesian Core (Destruction & Rebirth)
#    Based on BayLIME: replace OLS with Bayesian Ridge regression
# ─────────────────────────────────────────────────────────────

class BayesianCore:
    """
    Fuses the perturbed samples through Bayesian Ridge Regression.
    Outputs credible intervals (mean ± std) for each feature weight.
    Unstable noisy samples are naturally down-weighted via the prior.
    """

    def __init__(self, alpha_1: float = 1e-6, alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6, lambda_2: float = 1e-6):
        self.model = BayesianRidge(
            alpha_1=alpha_1, alpha_2=alpha_2,
            lambda_1=lambda_1, lambda_2=lambda_2,
            compute_score=True,
            fit_intercept=True,
        )

    def fit(
        self,
        binary_matrix: np.ndarray,
        predictions: np.ndarray,
        sample_weights: np.ndarray = None
    ):
        """
        binary_matrix: (n_samples, n_tokens)
        predictions:   (n_samples,) — LLM output probabilities
        sample_weights: cosine-distance weights from the autoencoder (tree-ALIME)
        """
        self.model.fit(binary_matrix, predictions, sample_weight=sample_weights)
        return self

    def get_weights_with_uncertainty(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_weights, std_weights) — the credible interval bounds.
        Large std = UALIME is signalling this feature is noisy/uncertain.
        """
        # BayesianRidge stores sigma_ (posterior covariance diagonal)
        means = self.model.coef_
        # sigma_ is the full covariance matrix
        stds = np.sqrt(np.diag(self.model.sigma_))
        return means, stds

    def compute_woe(
        self,
        pred_with: float,
        pred_without: float,
        eps: float = 1e-9
    ) -> float:
        """
        Weight of Evidence (M-LIME Eq. 1).
        Measures how much a token shifts the model's log-odds.
        """
        odds_with = pred_with / (1 - pred_with + eps)
        odds_without = pred_without / (1 - pred_without + eps)
        return np.log2(odds_with + eps) - np.log2(odds_without + eps)


# ─────────────────────────────────────────────────────────────
# 4. Interpretable Surrogate (tree-ALIME style)
#    Decision tree as the local model for human-readable output
# ─────────────────────────────────────────────────────────────

class InterpretableSurrogate:
    """
    Fits a decision tree (tree-ALIME) as the final local surrogate.
    Linear models have higher local fidelity; decision trees are far
    more interpretable to humans (tree-ALIME Table III).
    We expose both and let the stability map decide which to show.
    """

    def __init__(self, max_depth: int = 5, n_features_to_select: int = 10):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.n_features = n_features_to_select
        self.selected_feature_indices = None

    def fit(
        self,
        binary_matrix: np.ndarray,
        predictions: np.ndarray,
        feature_names: list[str],
        sample_weights: np.ndarray = None,
        pre_selected_indices: list[int] = None
    ):
        """
        pre_selected_indices: from Bayesian core's top-k by |mean_weight|
        """
        if pre_selected_indices is not None:
            self.selected_feature_indices = pre_selected_indices
        else:
            self.selected_feature_indices = list(range(min(self.n_features, binary_matrix.shape[1])))

        X_sub = binary_matrix[:, self.selected_feature_indices]
        y = (predictions > 0.5).astype(int)
        self.tree.fit(X_sub, y, sample_weight=sample_weights)
        self.feature_names = [feature_names[i] for i in self.selected_feature_indices]
        return self

    def get_feature_importances(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.tree.feature_importances_))

    def compute_stability_score(
        self,
        importances_a: dict[str, float],
        importances_b: dict[str, float]
    ) -> float:
        """
        tree-ALIME Eq. 4: Jaccard-based stability over features with importance > 0.
        """
        set_a = set(k for k, v in importances_a.items() if v > 0)
        set_b = set(k for k, v in importances_b.items() if v > 0)
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)


# ─────────────────────────────────────────────────────────────
# 5. ESI Metric (M-LIME)
# ─────────────────────────────────────────────────────────────

def compute_esi(
    weights_a: dict[str, float],
    weights_b: dict[str, float],
    lam: float = 40.0
) -> float:
    """
    Explanation Stability Index (M-LIME Eq. 9).
    Penalises not just feature set mismatch but also weight magnitude differences.
    lam=40 is the M-LIME paper's recommended value.
    """
    common_features = set(weights_a.keys()) & set(weights_b.keys())
    same_sign = {f for f in common_features if weights_a[f] * weights_b[f] >= 0}
    k = max(len(weights_a), len(weights_b), 1)

    score = sum(
        np.exp(-lam * abs(weights_a[f] - weights_b[f]))
        for f in same_sign
    )
    return score / k


# ─────────────────────────────────────────────────────────────
# 6. Autoencoder Weighting (tree-ALIME / ALIME)
#    Weights perturbed samples by similarity in latent space
# ─────────────────────────────────────────────────────────────

def compute_cosine_weights(
    original_text: str,
    perturbed_texts: list[str],
    encoder,          # sentence-transformers SentenceTransformer
    kernel_width: float = 0.25
) -> np.ndarray:
    """
    Encode all texts → cosine similarity → exponential kernel weights.
    This is the ALIME/tree-ALIME weighting: nearby samples in embedding
    space get higher weight, improving local fidelity.
    """
    all_texts = [original_text] + perturbed_texts
    embeddings = encoder.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    orig_emb = embeddings[0:1]
    pert_emb = embeddings[1:]
    sims = cosine_similarity(orig_emb, pert_emb)[0]
    distances = 1.0 - sims
    weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))
    return weights


# ─────────────────────────────────────────────────────────────
# 7. UALIME Orchestrator
# ─────────────────────────────────────────────────────────────

class UALIME:
    """
    Uncertainty-Aware LIME.

    Full pipeline:
      1. Adaptive Explosion      → BERT-marginalized perturbations
      2. Async LLM Batching      → collect predictions in parallel
      3. Uncertainty Gate        → measure entropy/variance, resample if needed
      4. Cosine Weighting        → ALIME-style sample weighting
      5. Bayesian Rebirth        → Bayesian Ridge → credible intervals
      6. Interpretable Surrogate → decision tree for human output
      7. Stability Map           → ESI score, error bounds per feature
    """

    def __init__(
        self,
        llm_predict_fn,               # callable: list[str] → list[float]
        mlm_model: str = "bert-base-uncased",
        sentence_encoder=None,        # SentenceTransformer or None
        n_initial_samples: int = 150,
        n_focused_samples: int = 100,
        variance_threshold: float = 0.05,
        max_feedback_loops: int = 3,
        sigma: float = 1e-4,
        esi_lambda: float = 40.0,
        tree_max_depth: int = 5,
        top_k_features: int = 10,
    ):
        self.predict = llm_predict_fn
        self.explosion = AdaptiveExplosion(mlm_model, sigma)
        self.gate = UncertaintyGate(variance_threshold, max_feedback_loops)
        self.bayesian = BayesianCore()
        self.surrogate = InterpretableSurrogate(tree_max_depth, top_k_features)
        self.sentence_encoder = sentence_encoder
        self.n_initial = n_initial_samples
        self.n_focused = n_focused_samples
        self.esi_lambda = esi_lambda

    def explain(self, text: str, verbose: bool = True) -> dict:
        """
        Returns:
          {
            "feature_weights":    {token: mean_weight},
            "credible_intervals": {token: (lower, upper)},
            "esi_score":          float,
            "stability_map":      {token: {"weight": float, "uncertainty": float, "stable": bool}},
            "tree_importances":   {token: float},
            "n_feedback_loops":   int,
          }
        """
        tokens = text.split()
        feature_names = tokens

        # ── Step 1: Initial perturbation batch ──────────────────────
        if verbose: print("[1/6] Adaptive explosion (BERT marginalization)...")
        binary_matrix, perturbed_texts = self.explosion.generate_samples(
            text, n_samples=self.n_initial
        )

        # ── Step 2: LLM predictions (batched) ───────────────────────
        if verbose: print("[2/6] Querying LLM...")
        predictions = np.array(self.predict(perturbed_texts), dtype=np.float32)

        # ── Step 3: Feedback loop if uncertainty is high ─────────────
        n_loops = 0
        for loop_i in range(self.gate.max_iterations):
            if not self.gate.should_resample(predictions):
                break
            if verbose:
                var = self.gate.compute_output_variance(predictions)
                print(f"[3/6] Uncertainty high (var={var:.4f}), focused explosion #{loop_i+1}...")
            uncertain_tokens = self.gate.identify_uncertain_tokens(binary_matrix, predictions)
            bm_extra, pt_extra = self.explosion.generate_samples(
                text, n_samples=self.n_focused, targeted_indices=uncertain_tokens
            )
            pred_extra = np.array(self.predict(pt_extra), dtype=np.float32)
            binary_matrix = np.vstack([binary_matrix, bm_extra])
            perturbed_texts += pt_extra
            predictions = np.concatenate([predictions, pred_extra])
            n_loops += 1

        if verbose and n_loops == 0:
            print("[3/6] Uncertainty acceptable, no focused resampling needed.")

        # ── Step 4: Cosine sample weights (ALIME) ────────────────────
        if self.sentence_encoder is not None:
            if verbose: print("[4/6] Computing autoencoder-style cosine weights...")
            sample_weights = compute_cosine_weights(text, perturbed_texts, self.sentence_encoder)
        else:
            if verbose: print("[4/6] No sentence encoder, using uniform weights.")
            sample_weights = np.ones(len(predictions))

        # ── Step 5: Bayesian rebirth ──────────────────────────────────
        if verbose: print("[5/6] Bayesian regression (credible intervals)...")
        self.bayesian.fit(binary_matrix, predictions, sample_weights=sample_weights)
        mean_weights, std_weights = self.bayesian.get_weights_with_uncertainty()

        # Select top-k features by |weight| for surrogate tree
        top_k_idx = np.argsort(np.abs(mean_weights))[-self.surrogate.n_features:]

        # ── Step 6: Decision tree surrogate (tree-ALIME) ──────────────
        if verbose: print("[6/6] Fitting decision tree surrogate (tree-ALIME)...")
        self.surrogate.fit(
            binary_matrix, predictions, feature_names,
            sample_weights=sample_weights,
            pre_selected_indices=list(top_k_idx)
        )
        tree_importances = self.surrogate.get_feature_importances()

        # ── Step 7: Build stability map ───────────────────────────────
        feature_weights = {}
        credible_intervals = {}
        stability_map = {}

        for i, token in enumerate(feature_names):
            w = float(mean_weights[i])
            s = float(std_weights[i])
            lower = w - 1.96 * s
            upper = w + 1.96 * s
            # "stable" = confidence interval does not cross zero
            stable = not (lower < 0 < upper)
            feature_weights[token] = w
            credible_intervals[token] = (lower, upper)
            stability_map[token] = {
                "weight": w,
                "uncertainty": s,
                "ci_lower": lower,
                "ci_upper": upper,
                "stable": stable,
            }

        # ESI between two halves of the sample (self-consistency check)
        half = len(predictions) // 2
        bm_a, bm_b = binary_matrix[:half], binary_matrix[half:]
        pred_a, pred_b = predictions[:half], predictions[half:]
        sw_a, sw_b = sample_weights[:half], sample_weights[half:]

        self.bayesian.fit(bm_a, pred_a, sw_a)
        w_a, _ = self.bayesian.get_weights_with_uncertainty()
        self.bayesian.fit(bm_b, pred_b, sw_b)
        w_b, _ = self.bayesian.get_weights_with_uncertainty()
        # Re-fit full model
        self.bayesian.fit(binary_matrix, predictions, sample_weights)

        weights_dict_a = {t: float(w_a[i]) for i, t in enumerate(feature_names)}
        weights_dict_b = {t: float(w_b[i]) for i, t in enumerate(feature_names)}
        esi = compute_esi(weights_dict_a, weights_dict_b, self.esi_lambda)

        return {
            "feature_weights": feature_weights,
            "credible_intervals": credible_intervals,
            "stability_map": stability_map,
            "tree_importances": tree_importances,
            "esi_score": esi,
            "n_feedback_loops": n_loops,
            "n_total_samples": len(predictions),
        }