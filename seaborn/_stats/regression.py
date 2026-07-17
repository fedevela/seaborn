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

        # Architecture seam (POLYFIT-004, POLYFIT-005): This method is the
        # single-group boundary. It owns pair completeness and post-filter
        # sufficiency, and its stable contract is a schema-compatible x/y
        # DataFrame that may contain zero rows. It must not depend on GroupBy or
        # reach outside the supplied frame, so group isolation remains the
        # caller's responsibility and an insufficient group is an ordinary result.
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
        # POLYFIT-004, POLYFIT-005 (single-group logic):
        # INPUT: only the observations belonging to the current group, as handed
        # off by the grouped caller; keep this group isolated from every other group.
        # FILTER this group's incomplete x/y pairs before evaluating sufficiency.
        # EVALUATE the requested-order sufficiency predicate using only this group's
        # retained complete coordinates; never borrow observations across groups.
        # IF the retained coordinates are insufficient, RETURN the schema-compatible
        # empty fitted result as a normal group outcome without attempting a fit.
        # IF no complete pair remains, TAKE the same empty-result transition.
        # OTHERWISE, FIT and RETURN points derived only from this group's retained pairs.
        data = data.dropna(subset=["x", "y"])
        x = np.asarray(data["x"].tolist())
        y = np.asarray(data["y"].tolist())
        if np.unique(x).size <= self.order:
            # TODO warn?
            empty = np.array([], dtype=float)
            return pd.DataFrame({"x": empty, "y": empty})

        p = np.polyfit(x, y, self.order)
        xx = np.linspace(x.min(), x.max(), self.gridsize)
        yy = np.polyval(p, xx)

        return pd.DataFrame(dict(x=xx, y=yy))

    # TODO we should have a way of identifying the method that will be applied
    # and then only define __call__ on a base-class of stats with this pattern

    def __call__(self, data, groupby, orient, scales):

        # Integration seam (POLYFIT-004, POLYFIT-005): GroupBy owns partitioning
        # and calls the single-group boundary; PolyFit owns only the transformation
        # of each partition. Dependency points from grouped orchestration to
        # _fit_predict, whose schema-compatible empty result lets aggregation
        # continue without a special failure channel or cross-group fallback.
        # POLYFIT-004, POLYFIT-005 (grouped orchestration logic):
        # FOR EACH group selected by groupby, HAND OFF that group's rows alone to
        # the single-group fit procedure; preserve group boundaries at every call.
        # ACCEPT either fitted points or the normal empty result from each group.
        # WHEN one group returns empty for insufficient or zero complete pairs,
        # CONTINUE processing all remaining groups and do not treat it as failure.
        # COMBINE only the points each group returned; an empty group contributes
        # no fitted points and cannot supply observations to any other group.
        return groupby.apply(data, self._fit_predict)


@dataclass
class OLSFit(Stat):

    ...
