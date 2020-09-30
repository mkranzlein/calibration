"""Implementation of four calibration methods:
    - histogram binning
    - isotonic regression
    - temperature scaling
    - scaling binning

Temperature scaling adapted from:
https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

Definitions:
    n: number of samples with predictions
    t: size of tagset
"""

import logging
import math

import torch

from sklearn.isotonic import IsotonicRegression

import util


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def histogram_binning(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                      labels_dev: torch.Tensor, labels_test: torch.Tensor,
                      num_bins: int = 10, bin_size: int = 10_000,
                      use_adaptive_binning: bool = True):
    """Calibrates confidence scores using histogram binning.

    Args:
        scores_dev: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for dev set.
        scores_test: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        num_bins: Number of even intervals to use for bins (fixed-width).
        bin_size: Number of scores to put in each bin (adaptive binning).
        use_adaptive_binning: Whether to use fixed-width or adaptive binning.
    """

    logger.info("Starting histogram binning...")

    if use_adaptive_binning:
        lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores_dev, bin_size)
    else:
        lower_boundaries, upper_boundaries = util.get_fixed_width_bin_boundaries(num_bins)

    flattened_scores_dev = scores_dev.reshape(-1)
    flattened_labels_dev = labels_dev.reshape(-1)
    flattened_scores_test = scores_test.clone().reshape(-1)

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask_dev = flattened_scores_dev.ge(lower) & flattened_scores_dev.lt(upper)
        average_dev_label = flattened_labels_dev[bin_mask_dev].float().mean()
        bin_mask_test = flattened_scores_test.ge(lower) & flattened_scores_test.lt(upper)
        flattened_scores_test[bin_mask_test] = average_dev_label

    labelset_size = scores_dev.shape[1]
    calibrated_scores_test = flattened_scores_test.reshape(-1, labelset_size)

    # TODO: normalize? Won't be able to do that with top-label
    # calibrated_scores_test = torch.nn.functional.softmax(calibrated_scores_test, dim=1)

    return calibrated_scores_test, lower_boundaries, upper_boundaries


def isotonic_regression(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                        labels_dev: torch.Tensor, labels_test: torch.Tensor,
                        num_bins: int = 10, bin_size: int = 10_000,
                        use_adaptive_binning: bool = True):

    """Calibrates confidence scores using scikit-learn implementation of isotonic regression.

        Isotonic regression does not require bins for recalibration.
        The bin-related parameters are just used to determine bin boundaries for evaluation.

    Args:
        scores_dev: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for dev set.
        scores_test: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        num_bins: Number of even intervals to use for bins (fixed-width).
        bin_size: Number of scores to put in each bin (adaptive binning).
        use_adaptive_binning: Whether to use fixed-width or adaptive binning.
    e"""

    logger.info("Starting isotonic regression...")

    if use_adaptive_binning:
        lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores_dev, bin_size)
    else:
        lower_boundaries, upper_boundaries = util.get_fixed_width_bin_boundaries(num_bins)

    # Scores need to be moved to CPU for sklearn model
    flattened_scores_dev = scores_dev.reshape(-1).cpu()
    flattened_labels_dev = labels_dev.reshape(-1).cpu()
    flattened_scores_test = (scores_test.reshape(-1)).cpu()

    model = IsotonicRegression(y_min=0, y_max=1)
    model.fit(X=flattened_scores_dev, y=flattened_labels_dev)
    predictions = torch.Tensor(model.predict(flattened_scores_test)).cuda()

    # Resize model predictions to get matrix of calibrated confidence scores
    labelset_size = scores_dev.shape[1]
    calibrated_scores_test = predictions.reshape(-1, labelset_size)

    return calibrated_scores_test, lower_boundaries, upper_boundaries


def temperature_scaling(logits_dev: torch.Tensor, logits_test: torch.Tensor,
                        labels_dev: torch.Tensor, labels_test: torch.Tensor,
                        learning_rate: int = .00001, iterations: int = 100_000,
                        num_bins: int = 10, bin_size: int = 10_000,
                        use_adaptive_binning: bool = True):
    """Calibrates confidence scores using temperature scaling.

        Temperature scaling does not require bins for recalibration.
        The bin-related parameters are just used to determine bin boundaries for evaluation.

    Args:
        logits_dev: [n, t] Tensor of pre-softmax logits for dev set.
        logits_test: [n, t] Tensor of pre-softmax logits for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        learning_rate: Learning rate for LBFGS optimizer.
        iterations: Iterations for LBFGS optimizer.
        num_bins: Number of even intervals to use for bins (fixed-width).
        bin_size: Number of scores to put in each bin (adaptive binning).
        use_adaptive_binning: Whether to use fixed-width or adaptive binning.

    """
    scores_dev = torch.nn.functional.softmax(logits_dev, dim=1)

    logger.info("Starting temperature scaling...")
    if use_adaptive_binning:
        lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores_dev, bin_size)
    else:
        lower_boundaries, upper_boundaries = util.get_fixed_width_bin_boundaries(num_bins)

    model = ModelWithTemperature()
    model.optimize_temperature(logits_dev, labels_dev.argmax(1), learning_rate=learning_rate, iterations=iterations)
    logger.info(f"Optimal temperature is {model.temperature.item():.3f}.")

    calibrated_logits_test = logits_test / model.temperature
    calibrated_scores_test = torch.nn.functional.softmax(calibrated_logits_test, dim=1).detach()

    return calibrated_scores_test, lower_boundaries, upper_boundaries


class ModelWithTemperature(torch.nn.Module):
    """A thin decorator, which wraps a model with temperature scaling."""

    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        # Initialize temperature to 1.5
        self.temperature = torch.nn.Parameter(torch.Tensor([1.5]))

    def optimize_temperature(self, logits_dev: torch.tensor, labels_dev: torch.tensor,
                             learning_rate: float, iterations: int) -> None:
        """Tunes the temperature of the model (using the dev set) by minimizing negative log likelihood.

        Args:
            logits_dev: An n x t tensor, where n is the number of samples and t is the size of the labelset.
            labels_dev: A 1-dimensional tensor of length n, where each entry is the integer index of the correct label.
            learning_rate: Learning rate for LBFGS optimizer.
            iterations: Iterations for LBFGS optimizer.
        """

        # Move model to GPU
        self.cuda()
        nll_criterion = torch.nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.LBFGS([self.temperature], lr=learning_rate, max_iter=iterations)

        def eval():
            loss = nll_criterion(logits_dev / self.temperature, labels_dev)
            loss.backward()
            return loss
        optimizer.step(eval)


def scaling_binning(logits_dev: torch.Tensor, logits_test: torch.Tensor,
                    labels_dev: torch.Tensor, labels_test: torch.Tensor,
                    learning_rate: int = .00001, iterations: int = 100_000,
                    bin_size: int = 10_000):
    """Calibrates confidence scores using scaling binning (Kumar et al., 2019).

        There's a lot going on here, so we use the same notation from Kumar et al. (2019) and defer to
        the algorithm explanation in the paper for details.

        Note: Kumar et al. advise against fixed-width binning for scaling binning,
        and instead recommend fixed-size bins.
        We do not test scaling binning with fixed-width binning.

    Args:
        logits_dev: [n, t] Tensor of pre-softmax logits for dev set.
        logits_test: [n, t] Tensor of pre-softmax logits for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
    """

    logger.info("Starting scaling binning...")
    # Split dev logits in half; t1-t3 notation comes from Kumar (2019)
    halfway = math.floor(logits_dev.shape[0] / 2)
    t1 = logits_dev[:halfway]
    y1 = labels_dev[:halfway]

    t2 = logits_dev[halfway:]
    # TODO: See if there's any way to make use of this data that we're essentially throwing away
    _ = labels_dev[halfway:]

    # Learn temperature scaling model
    model = ModelWithTemperature()
    model.optimize_temperature(t1, y1.argmax(1), learning_rate=learning_rate, iterations=iterations)
    logger.info(f"Optimal temperature is {model.temperature.item():.3f}.")

    # Bin t2 after the logits have been scaled based on the t1 model

    t2_scaled = torch.nn.functional.softmax(t2 / model.temperature, dim=1)

    lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(t2_scaled, bin_size=bin_size)

    t3_scaled = torch.nn.functional.softmax((logits_test / model.temperature), dim=1)

    labelset_size = logits_test.shape[1]

    # Detaching here is important; gradients hog VRAM otherwise
    # TODO: not sure that a clone is necessary here
    flattened_scores_test = t3_scaled.detach().clone().reshape(-1)

    assert (flattened_scores_test == 1.).sum(dim=0) == 0

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask_test = flattened_scores_test.ge(lower) & flattened_scores_test.lt(upper)
        average_temp_scaled_score = flattened_scores_test[bin_mask_test].float().mean()
        flattened_scores_test[bin_mask_test] = average_temp_scaled_score

    calibrated_scores_test = flattened_scores_test.reshape(-1, labelset_size)

    # Return the calibrated scores AND the bin boundaries (for evaluation)
    return calibrated_scores_test, lower_boundaries, upper_boundaries


def isotonic_scaling_binning(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                             labels_dev: torch.Tensor, labels_test: torch.Tensor,
                             bin_size: int = 10_000):

    logger.info("Starting isotonic scaling binning...")
    
    labelset_size = scores_dev.shape[1]
    # Split dev logits in half; t1-t3 notation comes from Kumar (2019)
    halfway = math.floor(scores_dev.shape[0] / 2)
    t1 = scores_dev[:halfway]
    y1 = labels_dev[:halfway]

    t2 = scores_dev[halfway:]
    # TODO: See if there's any way to make use of this data that we're essentially throwing away
    _ = labels_dev[halfway:]

    # Scores need to be moved to CPU for sklearn model
    flattened_t1 = t1.reshape(-1).cpu()
    flattened_y1 = y1.reshape(-1).cpu()
    flattened_t2 = t2.reshape(-1).cpu()
    flattened_t3 = (scores_test.reshape(-1)).cpu()

    model = IsotonicRegression(y_min=0, y_max=.99999)
    model.fit(X=flattened_t1, y=flattened_y1)
    t2_predictions = torch.Tensor(model.predict(flattened_t2)).cuda().reshape(-1, labelset_size)

    # Bin t2 after the scores have been adjusted based on t1 model

    lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(t2_predictions, bin_size=bin_size)

    t3_predictions = torch.Tensor(model.predict(flattened_t3)).cuda()

    assert (t3_predictions == 1.).sum(dim=0) == 0

    for lower, upper in zip(lower_boundaries, upper_boundaries):
        bin_mask_test = t3_predictions.ge(lower) & t3_predictions.lt(upper)
        average_iso_scaled_score = t3_predictions[bin_mask_test].float().mean()
        t3_predictions[bin_mask_test] = average_iso_scaled_score

    calibrated_scores_test = t3_predictions.reshape(-1, labelset_size)

    # Return the calibrated scores AND the bin boundaries (for evaluation)
    return calibrated_scores_test, lower_boundaries, upper_boundaries


def temperature_scaling_post_softmax(scores_dev: torch.Tensor, scores_test: torch.Tensor,
                                     labels_dev: torch.Tensor, labels_test: torch.Tensor,
                                     learning_rate: int = .00001, iterations: int = 100_000,
                                     num_bins: int = 10, bin_size: int = 10_000,
                                     use_adaptive_binning: bool = True):
    """Testing temperature scaling with post-softmaxed scores

        Temperature scaling does not require bins for recalibration.
        The bin-related parameters are just used to determine bin boundaries for evaluation.

    Args:
        scores_dev: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for dev set.
        scores_test: [n, t] Tensor of confidence scores (e.g. softmaxed logits) for test set.
        labels_dev: [n, t] One-hot tensor of labels for dev set.
        labels_test: [n, t] One-hot tensor of labels for test set.
        learning_rate: Learning rate for LBFGS optimizer.
        iterations: Iterations for LBFGS optimizer.
        num_bins: Number of even intervals to use for bins (fixed-width).
        bin_size: Number of scores to put in each bin (adaptive binning).
        use_adaptive_binning: Whether to use fixed-width or adaptive binning.

    """

    logger.info("Starting temperature scaling...")
    if use_adaptive_binning:
        lower_boundaries, upper_boundaries = util.get_adaptive_bin_boundaries(scores_dev, bin_size)
    else:
        lower_boundaries, upper_boundaries = util.get_fixed_width_bin_boundaries(num_bins)

    model = ModelWithTemperature()
    model.optimize_temperature(scores_dev, labels_dev.argmax(1), learning_rate=learning_rate, iterations=iterations)
    logger.info(f"Optimal temperature is {model.temperature.item():.3f}.")

    calibrated_scores_test = (scores_test / model.temperature).detach()

    return calibrated_scores_test, lower_boundaries, upper_boundaries
