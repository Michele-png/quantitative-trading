"""S&P 500 universe provider."""

from __future__ import annotations

from datetime import date

import pandas as pd


class SP500Universe:
    """Current S&P 500 constituent provider.

    The historical-membership implementation can be swapped in later; this
    minimal provider restores the public interface used by the dataset builder.
    """

    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self) -> None:
        self._members: set[str] | None = None

    def get_members(self, as_of: date) -> set[str]:
        """Return S&P 500 members for ``as_of``.

        The current implementation returns today's constituents and ignores
        ``as_of``; callers that require strict historical membership should use
        a persisted historical universe file.
        """
        _ = as_of
        if self._members is None:
            table = pd.read_html(self.WIKI_URL)[0]
            self._members = {
                str(symbol).replace(".", "-").upper()
                for symbol in table["Symbol"].tolist()
            }
        return set(self._members)
