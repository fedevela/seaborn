from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from seaborn._stats.base import Stat


@dataclass
class PolyFit(Stat):
    """
    Fit a polynomial of the given order and resample data onto predicted curve.
    """
    # This is a provisional class that is useful for building out functionality.
    # It may or may not change substantially in form or dissappear as we think
    # through the organization of the stats subpackage.

    order: int = 2
    gridsize: int = 100

    def _fit_predict(self, data):

        # Architecture seam (POLYFIT-001, POLYFIT-002, POLYFIT-003, POLYFIT-007):
        # This method owns complete-pair selection for one group. Keep the paired
        # tabular representation intact through that boundary; only then project
        # it into the separate numeric inputs consumed by numpy. The sufficiency
        # check, fit, and prediction-grid construction depend on that projection,
        # while grouping and missing-value policy remain outside numpy's boundary.
        # POLYFIT-001, POLYFIT-002, POLYFIT-003, POLYFIT-007:
        # INPUT: fitting observations containing corresponding x and y coordinates.
        # DERIVE one completeness mask that is true only where both coordinates
        # are non-null; do not fill, substitute, or otherwise impute either value.
        # SELECT x and y with that same mask so every retained pair comes from the
        # same original observation and every incomplete observation is discarded.
        # HAND OFF only the retained x and retained y to the existing unique-x
        # sufficiency decision; incomplete observations must not affect that branch.
        # IF sufficient complete pairs remain, fit using only those retained pairs,
        # derive the prediction grid from retained x bounds, and evaluate the fit.
        # IF they do not remain, follow the existing insufficient-data output path.
        # OUTPUT only predictions derived from complete fitting pairs; never pass a
        # missing coordinate or an imputed/incomplete observation to fit or grid input.
        x = data["x"]
        y = data["y"]
        if x.nunique() <= self.order:
            # TODO warn?
            xx = yy = []
        else:
            p = np.polyfit(x, y, self.order)
            xx = np.linspace(x.min(), x.max(), self.gridsize)
            yy = np.polyval(p, xx)

        return pd.DataFrame(dict(x=xx, y=yy))

    # TODO we should have a way of identifying the method that will be applied
    # and then only define __call__ on a base-class of stats with this pattern

    def __call__(self, data, groupby, orient, scales):

        return groupby.apply(data, self._fit_predict)


@dataclass
class OLSFit(Stat):

    ...
