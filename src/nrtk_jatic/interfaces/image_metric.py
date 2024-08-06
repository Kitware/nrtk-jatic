import abc
from typing import Any, Optional, Protocol, Dict
from maite.protocols import ArrayLike


class ImageMetric(Protocol):

    @abc.abstractmethod
    def compute(
        self,
        img_1: ArrayLike,
        img_2: Optional[ArrayLike] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Given up to two images, and additional parameters, return some given metric about the image(s).

        :param img_1: An input image in the shape (height, width, channels) that conforms with
                      maite's ArrayLike protocol.
        :param img_2: An optional input image in the shape (height, width, channels) that conforms with
                      maite's ArrayLike protocol.
        :param additional_params: A dictionary containing implementation-specific input param-values pairs.

        :return: Returns a single scalar value representing an implementation's computed metric. Implementations
                 should impart no side effects upon either input image or the additional parameters.
        """

    def __call__(
        self,
        img_1: ArrayLike,
        img_2: Optional[ArrayLike] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calls compute() with the given input image(s) and additional parameters."""

        return self.compute(img_1, img_2, additional_params)
