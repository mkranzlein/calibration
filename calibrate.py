"""Implementation of three calibration methods: histogram binning, isotonic regression, and temperature scaling.

Definitions:
    n: number of samples with predictions
    t: size of labelset
"""

import math

# Preempt Flake8 linting error on annotations
from typing import Optional

import numpy as np
import torch

from sklearn.isotonic import IsotonicRegression


class Bin:
    """"Functionality for binning.

    Not used for some calibration tuning methods, but always needed for evaluation.
    """

    def __init__(self, min_score: float, max_score: float):
        self.scores = []
        self.calibrated_scores = []
        self.labels = []
        self.score_ids = []
        self.min_score = min_score
        self.max_score = max_score

    def size(self) -> int:
        return len(self.scores)

    def average_label(self) -> float:
        return np.mean(self.labels)

    def average_score(self, use_calibrated_scores=False) -> float:
        if use_calibrated_scores:
            return np.mean(self.calibrated_scores)
        else:
            return np.mean(self.scores)

    def weighted_square_error(self, use_calibrated_scores=False) -> float:
        error = self.average_score(use_calibrated_scores) - self.average_label()
        square_error = error ** 2
        return self.size() * square_error

    def add(self, score: float, label: int, score_id: int, calibrated_score: Optional[int] = None) -> None:
        """Adds a confidence score to the bin. Include unique score_id for easier analysis later."""
        self.scores.append(score)
        self.labels.append(label)
        self.score_ids.append(score_id)

        if calibrated_score is not None:
            self.calibrated_scores.append(calibrated_score)


def fixed_width_binning(num_bins: int) -> list:
    """Creates a set of empty fixed-width bins with boundaries set based on number of bins."""

    bins = []
    bin_interval = 1.0 / num_bins

    # Set up first bin
    bins.append(Bin(min_score=0, max_score=bin_interval))

    for i in range(1, num_bins):
        # Min score of current bin is max score of previous bin
        # (ensures no scores are left outside of bin ranges due to rounding errors)
        bins.append(Bin(min_score=bins[-1].max_score, max_score=bin_interval * (i + 1)))

    # Last bin gets impossible max score for rounding errors that may push some confidence scores above 1 (e.g. 1.00004)
    bins[-1].max_score = 1.1

    return bins

# TODO: fixed_size_binning


def place_scores_in_bins(scores: list, labels: list, bins: list,
                         ids: Optional[list] = None, calibrated_scores: Optional[list] = None) -> list:
    """Place confidence scores in empty bins with boundaries already set.

    Allows for inclusion of pre-calibrated scores (e.g. with a method like Temperature Scaling,
    which doesn't require binning).
    """

    # We start with bin 0, but as scores get higher, we need to check fewer and fewer bins since we sort the scores
    start_bin = 0

    if ids is None:
        # Sort scores for improved placement efficiency
        score_ids = list(range(len(scores)))
    else:
        score_ids = ids

    print(f"Placing {len(scores)} scores in {len(bins)} bins...")
    if calibrated_scores is None:
        sorted_data = sorted(zip(scores, labels, score_ids), key=lambda x: x[0])
        for j, (score, label, score_id) in enumerate(sorted_data):
            if j % 1_000_000 == 0:
                print(j)
            for i, b in enumerate(bins[start_bin:]):
                if score >= b.min_score and score < b.max_score:
                    b.add(score, label, score_id)
                    # Only increase start bin if we placed something in bin after it (i.e. i > 0)
                    start_bin = start_bin + i
                    break
    else:
        sorted_data = sorted(zip(scores, labels, score_ids, calibrated_scores), key=lambda x: x[0])
        for j, (score, label, score_id, calibrated_score) in enumerate(sorted_data):
            if j % 1_000_000 == 0:
                print(j)
            for i, b in enumerate(bins[start_bin:]):
                if score >= b.min_score and score < b.max_score:
                    b.add(score, label, score_id, calibrated_score=calibrated_score)
                    # Only increase start bin if we placed something in bin after it (i.e. i > 0)
                    start_bin = start_bin + i
                    break

    assert len(scores) == sum([b.size() for b in bins])
    return bins


def estimate_calibration_error(bins: list, use_calibrated_scores=False):
    """Calculates estimate of calibration error using RMSE-based method from Nguyen and O'Connor (2015)."""
    weighted_square_errors = []

    for b in bins:
        weighted_square_errors.append(b.weighted_square_error(use_calibrated_scores))

    calibration_error = math.sqrt(np.sum(weighted_square_errors) / sum([b.size() for b in bins]))
    return calibration_error


# ---------------------------------------------------------------------------- #
#                               Histogram Binning                              #
# ---------------------------------------------------------------------------- #
def histogram_binning(scores_dev: list, labels_dev: list,
                      scores_test: list, labels_test: list,
                      use_fixed_width_bins=True, num_bins: int = 10, bin_size: int = 200):
    """Calibrates using histogram binning.

    Args:
        scores_dev: A list of n * t of confidence scores (i.e. softmaxed logits) for dev set.
        labels_dev: A list of n * t binary labels for dev set (i.e. flattened one-hot label matrix)
        scores_test: A list of n * t of confidence scores (i.e. softmaxed logits) for dev set.
        labels_test: A list of n * t binary labels for dev set (i.e. flattened one-hot label matrix)
        use_fixed_width_bins: If true, use fixed-width bins and num_bins. Otherwise, use fixed-size bins and bin_size.
        num_bins: The number of evenly sized intervals to use for binning.
        bin_size: The number of scores to put in each bin.
    """

    if use_fixed_width_bins:
        bins_empty_dev = fixed_width_binning(num_bins=num_bins)

    print("Binning dev scores...")
    bins_dev = place_scores_in_bins(scores_dev, labels_dev, bins_empty_dev)

    # Create empty bins for the data targeted for calibration with the same bin boundaries as from the dev set
    bins_empty_test = []

    for b in bins_dev:
        bins_empty_test.append(Bin(b.min_score, b.max_score))
    print("Binning test scores...")
    bins_test = place_scores_in_bins(scores_test, labels_test, bins_empty_test)

    for dev_bin, test_bin in zip(bins_dev, bins_test):
        test_bin.calibrated_scores = [dev_bin.average_label()] * test_bin.size()

    print("Estimating calibration error pre-tuning")
    pre_tuning_error = estimate_calibration_error(bins_test, use_calibrated_scores=False)
    print("Estimating calibration error post-tuning")
    post_tuning_error = estimate_calibration_error(bins_test, use_calibrated_scores=True)

    print(f"Pre-tuning error: {pre_tuning_error}")
    print(f"Post-tuning error: {post_tuning_error}")
    print(f"Relative change: {(post_tuning_error - pre_tuning_error) / pre_tuning_error}")
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                              Isotonic Regression                             #
# ---------------------------------------------------------------------------- #
def isotonic_regression(scores_dev: list, labels_dev: list,
                        scores_test: list, labels_test: list,
                        use_fixed_width_bins=True, num_bins: int = 10, bin_size: int = 200):
    """Calibrates using scikit-learn implementation of isotonic regression.

        Args:
        scores_dev: A list of n * t of confidence scores (i.e. softmaxed logits) for dev set.
        labels_dev: A list of n * t binary labels for dev set (i.e. flattened one-hot label matrix)
        scores_test: A list of n * t of confidence scores (i.e. softmaxed logits) for dev set.
        labels_test: A list of n * t binary labels for dev set (i.e. flattened one-hot label matrix)
        use_fixed_width_bins: If true, use fixed-width bins and num_bins. Otherwise, use fixed-size bins and bin_size.
        num_bins: The number of evenly sized intervals to use for binning.
        bin_size: The number of scores to put in each bin.
    """

    print("Starting isotonic regression...")
    if use_fixed_width_bins:
        empty_bins = fixed_width_binning(num_bins)
    target_bins = place_scores_in_bins(scores_test, labels_test, empty_bins)

    print("Building regression model based on dev scores and labels...")
    iso_reg = IsotonicRegression(y_min=0, y_max=1).fit(X=scores_dev, y=labels_dev)

    print("Tuning scores...")
    for b in target_bins:
        b.calibrated_scores = iso_reg.predict(b.scores)

    print("Estimating calibration error pre-tuning")
    pre_tuning_error = estimate_calibration_error(target_bins, use_calibrated_scores=False)
    print("Estimating calibration error post-tuning")
    post_tuning_error = estimate_calibration_error(target_bins, use_calibrated_scores=True)
    # Measure calibration error of uncalibrated scores and calibrated scores

    print(f"Pre-tuning error: {pre_tuning_error}")
    print(f"Post-tuning error: {post_tuning_error}")
    print(f"Relative change: {(post_tuning_error - pre_tuning_error) / pre_tuning_error}")
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                              Temperature Scaling                             #
# ---------------------------------------------------------------------------- #
def temperature_scaling(logits_dev: np.ndarray, label_indexes_dev: list,
                        logits_test: np.ndarray, label_indexes_test: list,
                        labelset_size: int, use_fixed_width_bins=True, num_bins=10, bin_size=200,
                        learning_rate: int = .00001, iterations: int = 100_000):
    """Calibrates using temperature scaling.

    Implementation adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py.

    Args:
        logits_dev: An [n, t] array of (pre-softmax) logits for dev set.
        label_indexes_dev: A list of n label indexes for dev set.
        logits_dev: An [n, t] array of (pre-softmax) logits for test set.
        label_indexes_test: label_indexes_dev: A list of n label indexes for test set.
        labelset_size: The number of possible labels.
        use_fixed_width_bins: If true, evaluation uses fixed-width bins and num_bins, else fixed_size bins and bin_size.
        num_bins: The number of evenly sized intervals to use for binning.
        bin_size: The number of scores to put in each bin.
    """

    print("Starting Temperature Scaling...")
    logits_dev = torch.tensor(logits_dev)
    label_indexes_dev = torch.tensor(label_indexes_dev, dtype=torch.long)
    model = ModelWithTemperature()
    model.optimize_temperature(logits_dev, label_indexes_dev, learning_rate=learning_rate, iterations=iterations)
    print(f"Optimal temperature is {model.temperature.item():.3f}.")

    softmax = torch.nn.Softmax(dim=1)
    # Uncalibrated confidence scores
    scores_test = list((softmax(torch.Tensor(logits_test))).numpy().flatten())

    # Calibrated confidence scores
    scaled_logits_test = logits_test * (1 / model.temperature.item())
    calibrated_scores_test = list((softmax(torch.Tensor(scaled_logits_test))).numpy().flatten())

    labels_test = list(torch.nn.functional.one_hot(torch.tensor(label_indexes_test).long(),
                                                   labelset_size).numpy().flatten())

    if use_fixed_width_bins:
        bins_test_empty = fixed_width_binning(num_bins)

    bins_test = place_scores_in_bins(scores_test, labels_test, bins_test_empty,
                                     calibrated_scores=calibrated_scores_test)

    print("Estimating calibration error pre-tuning")
    pre_tuning_error = estimate_calibration_error(bins_test, use_calibrated_scores=False)
    print("Estimating calibration error post-tuning")
    post_tuning_error = estimate_calibration_error(bins_test, use_calibrated_scores=True)

    print(f"Pre-tuning error: {pre_tuning_error}")
    print(f"Post-tuning error: {post_tuning_error}")
    print(f"Relative change: {(post_tuning_error - pre_tuning_error) / pre_tuning_error}")


class ModelWithTemperature(torch.nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    """

    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    # def forward(self, logits):
    #     raise NotImplementedError()

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits.
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def optimize_temperature(self, logits_dev: torch.tensor, labels_dev: torch.tensor,
                             learning_rate: float, iterations: int) -> None:
        """
        Tunes the temperature of the model (using the dev set) by minimizing negative log likelihood.

        Args:
            logits_dev: An n x t tensor, where n is the number of samples and t is the size of the labelset.
            labels_dev: A 1-dimensional tensor of length n, where each entry is the integer index of the correct label.
        """

        self.cuda()
        nll_criterion = torch.nn.CrossEntropyLoss().cuda()

        with torch.no_grad():
            logits = logits_dev.cuda()
            labels = labels_dev.cuda()
        optimizer = torch.optim.LBFGS([self.temperature], lr=learning_rate, max_iter=iterations)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
# ---------------------------------------------------------------------------- #
