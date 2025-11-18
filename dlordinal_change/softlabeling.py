from typing import Optional
import torch
from torch import Tensor
from dlordinal.losses.custom_targets_loss import CustomTargetsCrossEntropyLoss

import numpy as np
from scipy.stats import binom


def get_binomial_soft_labels(J):
    """Get soft labels for the binomial distribution for ``J`` classes or splits
    using the approach described in :footcite:t:`liu2020unimodal`.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the binomial
    probability function for the interval boundaries. The probability for the first
    interval is computed as the value of the binomial probability function for the first
    interval boundary.

    The binomial distributions employed are denoted as :math:`\\text{b}(k, n-1, p)` where
    :math:`k` is given by the order of the class for which the probability is computed,
    and :math:`p` is given by :math:`0.1 + (0.9-0.1) / (n-1) * j` where :math:`j` is
    is the order of the target class.

    Parameters
    ----------
    J : int
            Number of classes or splits.

    Raises
    ------
    ValueError
            If ``J`` is not a positive integer greater than 1.

    Returns
    -------
    probs : 2d array-like of shape (J, J)
            Matrix of probabilities where each row represents the true class
            and each column the probability for class ``j``.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_binomial_soft_labels
    >>> get_binomial_soft_labels(5)
    array([[6.561e-01, 2.916e-01, 4.860e-02, 3.600e-03, 1.000e-04],
            [2.401e-01, 4.116e-01, 2.646e-01, 7.560e-02, 8.100e-03],
            [6.250e-02, 2.500e-01, 3.750e-01, 2.500e-01, 6.250e-02],
            [8.100e-03, 7.560e-02, 2.646e-01, 4.116e-01, 2.401e-01],
            [1.000e-04, 3.600e-03, 4.860e-02, 2.916e-01, 6.561e-01]])
    """


    if J < 2 or not isinstance(J, int):
        raise ValueError(f"{J=} must be a positive integer greater than 1")

    params = {}
    # CHANGE: can be any J
    params[str(J)] = np.linspace(0.1, 0.9, J)

    probs = []

    for true_class in range(0, J):
        probs.append(binom.pmf(np.arange(0, J), J - 1, params[str(J)][true_class]))

    return np.array(probs)


class BinomialCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Binomial unimodal regularised cross entropy loss from :footcite:t:`liu2020unimodal`.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    """

    def __init__(
        self,
        num_classes: int,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_binomial_soft_labels(num_classes)).float()

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=0.0,
        )
