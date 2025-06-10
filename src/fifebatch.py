import duckdb
import math
import numpy as np
import matplotlib.pyplot as plt

class FifebatchField:
    def __init__(self, field, pattern):
        self._field = field
        self._pattern = pattern

    def get_field_statistics(self):
        """
        Get the summary statistics of a numeric field.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        sql_string = f"""
        SELECT  COUNT(*) AS count,
                COUNT(DISTINCT("{self._field}")) AS distinct_count,
                AVG("{self._field}") AS avg,
                MIN("{self._field}") AS min,
                MAX("{self._field}") AS max,
                approx_quantile("{self._field}", 0.90) AS p90
        FROM '{self._pattern}'
        WHERE {self._field} IS NOT NULL AND {self._field} < 1e9;
        """
        self._summary_statistics = {k: v[0] for k,v in duckdb.query(sql_string).fetchnumpy().items()}
        self._summary_statistics['range'] = (0, 10**math.ceil(np.log10(self._summary_statistics['p90'])))
        self._summary_statistics['binwidth'] = self._summary_statistics['range'][1] / 1e4
        self._summary_statistics['fullcount'] = duckdb.query(f"SELECT COUNT(*) AS count FROM '{self._pattern}'").fetchnumpy()['count'][0]

    def load_data(self):
        """
        Load the histogram data for a numeric field.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        sql_string = f"""
        SELECT  FLOOR("{self._field}" / {self._summary_statistics['binwidth']}) * {self._summary_statistics['binwidth']} AS "{self._field}",
                COUNT(*) AS count
        FROM '{self._pattern}'
        WHERE {self._field} IS NOT NULL AND {self._field} < 1e9
        GROUP BY 1
        ORDER BY 1;
        """
        self._data = list(duckdb.query(sql_string).fetchnumpy().values())

    def plot(self, downsample=1, save=False):
        """
        Plot the histogram.
        
        Parameters
        ----------
        downsample : int
            The factor by which to downsample the histogram. Defaults to 1.
        save : bool
            Whether to save the plot or not. Defaults to False.

        Returns
        -------
        None.
        """
        figure = plt.figure()
        ax = figure.add_subplot()
        nbins = int((self._summary_statistics['range'][1] - self._summary_statistics['range'][0]) / self._summary_statistics['binwidth'])
        nbins_ds = int(nbins / downsample) if downsample < nbins else nbins
        ax.hist(self._data[0], weights=self._data[1], bins=nbins_ds, histtype='step', range=self._summary_statistics['range'])
        ax.set_xlabel(self._field)
        ax.set_ylabel('Entries')
        ax.set_title(self._field)
        ax.set_yscale('log')
        ax.set_ylim(0.1, None)
        textstr = f'Entries: {self._summary_statistics["count"]:.3e}'
        textstr += f'\nNull Fraction: {(self._summary_statistics["fullcount"] - self._summary_statistics["count"]) / self._summary_statistics["fullcount"]:.3f}'
        ax.text(0.75, 0.875, textstr, transform=ax.transAxes)
        return figure