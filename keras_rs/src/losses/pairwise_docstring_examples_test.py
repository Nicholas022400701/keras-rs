import numpy as np
from absl.testing import parameterized

from keras_rs.src import testing
from keras_rs.src.losses.pairwise_hinge_loss import PairwiseHingeLoss
from keras_rs.src.losses.pairwise_logistic_loss import PairwiseLogisticLoss
from keras_rs.src.losses.pairwise_mean_squared_error import (
    PairwiseMeanSquaredError,
)
from keras_rs.src.losses.pairwise_soft_zero_one_loss import (
    PairwiseSoftZeroOneLoss,
)

Y_UNBATCHED = np.array([1.0, 0.0, 1.0, 3.0, 2.0])
S_UNBATCHED = np.array([1.0, 3.0, 2.0, 4.0, 0.8])
Y_BATCHED = np.array([[1.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
S_BATCHED = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 1.8, 2.0, 3.0]])
MASK = np.array([[True, True, True, True], [True, True, False, False]])
SAMPLE_WEIGHT = np.array([[2.0, 3.0, 1.0, 1.0], [2.0, 1.0, 0.0, 0.0]])

CASES = [
    (
        "hinge",
        PairwiseHingeLoss,
        2.32,
        0.75,
        0.65,
        1.025,
        [[3.0, 0.0, 2.0, 0.0], [0.0, 0.20000005, 0.79999995, 0.0]],
    ),
    (
        "logistic",
        PairwiseLogisticLoss,
        1.70708,
        0.73936,
        0.53751,
        0.80337,
        [
            [2.126928, 0.0, 1.3132616, 0.48877698],
            [0.0, 0.37110072, 0.9114005, 0.7034721],
        ],
    ),
    (
        "mse",
        PairwiseMeanSquaredError,
        19.104,
        5.58,
        4.76,
        11.05,
        [[11.0, 17.0, 5.0, 5.0], [2.04, 1.3199999, 1.6399999, 1.6399999]],
    ),
    (
        "soft_zero_one",
        PairwiseSoftZeroOneLoss,
        0.86103,
        0.46202,
        0.29468,
        0.40478,
        [
            [0.8807971, 0.0, 0.73105854, 0.43557024],
            [0.0, 0.31002545, 0.7191075, 0.61961967],
        ],
    ),
]


class PairwiseDocstringExamplesTest(testing.TestCase, parameterized.TestCase):
    """Scratch test that checks the numbers in the pairwise loss docstrings."""

    @parameterized.named_parameters(CASES)
    def test_examples(
        self, loss_cls, unbatched, batched, masked, sample_weight, none
    ):
        loss = loss_cls()
        self.assertAllClose(
            loss(y_true=Y_UNBATCHED, y_pred=S_UNBATCHED), unbatched, atol=1e-5
        )
        self.assertAllClose(
            loss(y_true=Y_BATCHED, y_pred=S_BATCHED), batched, atol=1e-5
        )
        self.assertAllClose(
            loss(y_true={"labels": Y_BATCHED, "mask": MASK}, y_pred=S_BATCHED),
            masked,
            atol=1e-5,
        )
        self.assertAllClose(
            loss(
                y_true=Y_BATCHED,
                y_pred=S_BATCHED,
                sample_weight=SAMPLE_WEIGHT,
            ),
            sample_weight,
            atol=1e-5,
        )
        self.assertAllClose(
            loss_cls(reduction="none")(y_true=Y_BATCHED, y_pred=S_BATCHED),
            none,
            atol=1e-6,
        )
