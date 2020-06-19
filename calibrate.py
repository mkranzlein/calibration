"""Implementation of three calibration methods: histogram binning, isotonic regression, and temperature scaling.

ECE measurement adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

Definitions:
    n: number of samples with predictions
    t: size of labelset
"""

import torch


def get_fixed_width_bin_boundaries(num_bins: int = 10):
    """Calculates bin boundaries for num_bins evenly-spaced bins.

    This can be done without having any confidence scores.
    
    Args:
        num_bins: Number of evenly-spaced bins to create.
    """

    boundaries = torch.linspace(start=0, end=1, steps=num_bins + 1)
    lower_boundaries = boundaries[:-1]
    upper_boundaries = boundaries[1:]
    return lower_boundaries, upper_boundaries


def get_fixed_size_bin_boundaries(scores: torch.Tensor, bin_size: int = 200):
    """Calculates bin boundaries for bins that each contain an equal number of items.

    Requires confidence scores.

    Args:
        scores: [n, t] Tensor of confidence scores.
        bin_size: Number of scores to put in each bin before
    """

    flattened_scores = scores.reshape(-1)
    if flattened_scores.shape[0] < bin_size:
        raise ValueError(f"Too few scores ({flattened_scores.shape[0]}) to fill a bin. Bin size {bin_size} too large.")

    binned_scores = flattened_scores.sort().values.reshape(-1, bin_size)
    lower_boundaries = torch.cat((torch.Tensor([0.0]), binned_scores[:-1, 0]))
    upper_boundaries = torch.cat((binned_scores[:-1, 0], torch.Tensor([1.1])))

    return lower_boundaries, upper_boundaries


def estimate_error(scores: torch.Tensor, labels: torch.Tensor,
                   lower_boundaries: list, upper_boundaries: list,
                   k: int = 999999, min_score: float = 0.0, max_score: float = 1.0):
    """Estimates calibration error using RMSE-based binning method from Nguyen & O'Connor (2015).

    Args:
        scores: [n, t] Tensor of confidence scores.
        labels: [n, t] One-hot tensor of labels.
        lower_boundaries: List of lower boundaries for bins.
        upper_boundaries: List of upper boundaries for bins.
        k: Top-k confidence scores to use when estimating error.
            If k is bigger than labelset_size, all scores will be used.
        min_score: Minimum uncalibrated confidence score to consider when estimating error.
        max_score: Maximum uncalibrated confidence_score to consider when estimating error.
    """

    if k > scores.shape[1]:
        k = scores.shape[1]

    top_scores, indices = scores.topk(k, 1)
    top_labels = labels.gather(1, indices)
    flattened_scores = top_scores.reshape(-1)
    flattened_labels = top_labels.reshape(-1)

    # TODO: Check ge/le vs. ge/lt
    range_mask = flattened_scores.ge(min_score) * flattened_scores.le(max_score)
    valid_scores = flattened_scores.masked_select(range_mask)
    valid_labels = flattened_labels.masked_select(range_mask)

    ece = 0
    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask = valid_scores.ge(lower) * valid_scores.lt(upper)
        proportion_in_bin = bin_mask.double().mean()
        if proportion_in_bin.item() > 0:
            average_label = valid_labels[bin_mask].double().mean()
            average_confidence = valid_scores[bin_mask].double().mean()
            squared_error = (average_confidence - average_label) ** 2
            weighted_square_error = proportion_in_bin * squared_error
            ece += weighted_square_error
    ece = torch.sqrt(ece).item()

    return ece


def histogram_binning(scores_dev: torch.Tensor, labels_dev: torch.Tensor,
                      scores_test: torch.Tensor, labels_test: torch.Tensor,
                      num_bins: int = 10, k: int = 999_999, min_score: float = 0.0, max_score: float = 1.0):
    """Calibrates confidence scores using histogram binning.

    Args:
        scores_dev: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for dev set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        scores_test: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for test set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        num_bins: The number of evenly sized intervals to use for binning.
        k: Top-k confidence scores to use when estimating error.
            If k is bigger than labelset_size, all scores will be used.
        min_score: Minimum uncalibrated confidence score to consider when estimating error.
        max_score: Maximum uncalibrated confidence_score to consider when estimating error.
    """

    print("Starting histogram binning...")
    lower_boundaries, upper_boundaries = get_fixed_width_bin_boundaries(num_bins)

    flattened_scores_dev = scores_dev.reshape(-1)
    flattened_labels_dev = labels_dev.reshape(-1)
    flattened_scores_test = scores_test.clone().reshape(-1)

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask_dev = flattened_scores_dev.ge(lower) * flattened_scores_dev.lt(upper)
        average_dev_label = flattened_labels_dev[bin_mask_dev].float().mean()
        bin_mask_test = flattened_scores_test.ge(lower) * flattened_scores_test.lt(upper)
        flattened_scores_test[bin_mask_test] = average_dev_label

    labelset_size = scores_dev.shape[1]
    calibrated_scores_test = flattened_scores_test.reshape(-1, labelset_size)

    pre_tuning_error = estimate_error(scores_test, labels_test,
                                      lower_boundaries, upper_boundaries,
                                      k=k, min_score=min_score, max_score=max_score)
    post_tuning_error = estimate_error(calibrated_scores_test, labels_test,
                                       lower_boundaries, upper_boundaries,
                                       k=k, min_score=min_score, max_score=max_score)

    print(f"Original calibration error: {pre_tuning_error:.4f}")
    print(f"Post-tuning calibration error: {post_tuning_error:.4}")
    print(f"Calibration error changed by {(((post_tuning_error - pre_tuning_error) / pre_tuning_error) * 100):.3f}%")
