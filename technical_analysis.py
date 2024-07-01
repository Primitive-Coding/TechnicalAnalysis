import numpy as np

# Pandas
import pandas as pd

# Plotting
import mplfinance as mpf
import matplotlib.pyplot as plt

import datetime as dt

# Set pandas options to display in normal notation
pd.set_option("display.float_format", "{:,.2f}".format)

# Yahoo Finance
import yfinance as yf

import scipy.signal as Signal

binance_dark = {
    "base_mpl_style": "dark_background",
    "marketcolors": {
        "candle": {"up": "#3dc985", "down": "#ef4f60"},
        "edge": {"up": "#3dc985", "down": "#ef4f60"},
        "wick": {"up": "#3dc985", "down": "#ef4f60"},
        "ohlc": {"up": "green", "down": "red"},
        "volume": {"up": "#247252", "down": "#82333f"},
        "vcedge": {"up": "green", "down": "red"},
        "vcdopcod": False,
        "alpha": 1,
    },
    "mavcolors": ("#ad7739", "#a63ab2", "#62b8ba"),
    "facecolor": "#1b1f24",
    "gridcolor": "#2c2e31",
    "gridstyle": "--",
    "y_on_right": True,
    "rc": {
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.edgecolor": "#474d56",
        "axes.titlecolor": "red",
        "figure.facecolor": "#161a1e",
        "figure.titlesize": "x-large",
        "figure.titleweight": "semibold",
    },
    "base_mpf_style": "binance-dark",
}


class TechnicalAnalysis:
    def __init__(self) -> None:
        self.colors = ["red", "orange", "yellow", "green", "purple"]

    def get_RSI(self, prices: pd.Series, rsi_window: int = 14):
        """
        Calculate the Relative Strength Index (RSI) for a given dataset.

        Parameters
        ----------
        prices : pd.Series
            Time series of price data.
        rsi_window : int, optional
            Window to use for RSI calculation, by default 14

        Returns
        -------
        pd.Series
            Return time series containing RSI data.
        """
        delta = prices.diff()
        # Separate positive and negative gains
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        rsi_ma = rsi.rolling(window=rsi_window).mean()
        df = pd.DataFrame({"RSI": rsi, "RSI_MA": rsi_ma})
        return df

    def get_MACD(
        self,
        prices: pd.Series,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
    ):
        """
        Calculate the Moving Average Convergence Divergence (MACD) for a given dataset.

        Parameters
        ----------
        prices : pd.Series
            Time series of prices data.
        short_window : int, optional
            Short period for 'fast EMA', by default 12
        long_window : int, optional
            Long period for 'slow EMA', by default 26
        signal_window : int, optional
            Period for the signal, by default 9

        Returns
        -------
        pd.DataFrame
            Returns dataframe containing MACD data.
        """
        # Calculate the short-term EMA
        short_ema = prices.ewm(span=short_window, adjust=False).mean()
        # Calculate the long-term EMA
        long_ema = prices.ewm(span=long_window, adjust=False).mean()
        # Calculate the MACD line
        macd_line = short_ema - long_ema
        # Calculate the signal line
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        # Calculate histogram
        histogram = macd_line - signal_line
        # Create a DataFrame to store the results
        macd_df = pd.DataFrame(
            {"MACD": macd_line, "Signal Line": signal_line, "Histogram": histogram}
        )
        return macd_df

    def get_EMA(self, prices: pd.Series, ema_window: int):
        """
        Get exponential moving average based on a price series. Moving average can be adjusted with 'ema_window' function parameter..

        Parameters
        ----------
        prices : pd.Series
            Time series of price data.
        ema_window : int
            Period for ema.

        Returns
        -------
        pd.Series
            Returns series containing EMA data.
        """
        ema = prices.ewm(span=ema_window, adjust=False).mean()
        return ema

    def get_EMA_group(self, df: pd.DataFrame, ema_ranges=[9, 20, 200]) -> pd.DataFrame:
        """
        Get a collection of EMA ranges.

        Parameters
        ----------
        df : pd.DataFrame
            Default dataframe retrieved from Yahoo Finance.
        ema_ranges : list, optional
            EMA ranges to get, by default [9, 20, 200]

        Returns
        -------
        pd.DataFrame
            Same dataframe as passed in function parameter, but with EMA columns
        """

        for r in ema_ranges:
            df[f"EMA-{r}"] = self.get_EMA(df["Close"], r)
        return df

    def get_BB_Bands(self, prices: pd.Series, bb_window: int = 20):
        """
        Get bollinger bands based on a price series. Bands can be adjusted with "bb_window" function parameter.

        Parameters
        ----------
        prices : pd.Series
            Time series of price data.
        bb_window : int, optional
            Period for Bollinger Bands, by default 20

        Returns
        -------
        pd.DataFrame
            Returns DataFrame with columns for "upper_band" and "lower_band".
        """
        moving_avg = prices.rolling(window=bb_window).mean()
        std_dev = prices.rolling(window=bb_window).std()

        # Calculate upper & lower band.
        upper_band = moving_avg + (std_dev * 2)
        lower_band = moving_avg - (std_dev * 2)
        bands = pd.DataFrame({"upper_BB_band": upper_band, "lower_BB_band": lower_band})
        return bands

    def get_ATR_Bands(
        self, high: pd.Series, low: pd.Series, close: pd.Series, atr_window: int = 14
    ):
        """
        Get Average True Range Bands based on a high, low, and close price series.

        Parameters
        ----------
        high : pd.Series
            Time series of "high" stock prices.
        low : pd.Series
            Time series of "low" stock prices.
        close : pd.Series
            Time series of closing stock prices.
        atr_window : int, optional
            Period for ATR calculations, by default 14

        Returns
        -------
        pandas.DataFrame
            DataFrame containing "upper_ATR_band" and "lower_ATR_band" columns.
        """

        # Calculate the True Range (TR)
        tr = np.maximum(
            high - low,
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1)),
        )
        # Calculate 14-day ATR
        atr = tr.rolling(window=atr_window).mean()
        # Calculate 20-day simple moving average (SMA)
        sma = close.rolling(window=20).mean()
        # Calculate the upper and lower ATR bands
        df = pd.DataFrame()
        atr_multiplier = 2
        df["upper_ATR_band"] = sma + (atr_multiplier * atr)
        df["lower_ATR_band"] = sma - (atr_multiplier * atr)
        return df

    def get_Stochastic_Oscillator(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ):
        """
        Get Stochastic Oscillator based on a high, low, and close price series.

        Parameters
        ----------
        high : pd.Series
            Time series of "high" stock prices.
        low : pd.Series
            Time series of "low" stock prices.
        close : pd.Series
            Time series of closing stock prices.
        atr_window : int, optional
            Period for ATR calculations, by default 14

        Returns
        -------
        pandas.DataFrame
            DataFrame containing "D" and "K" columns.
        """

        # Calculate the %K line
        lookback_period = period
        l14 = low.rolling(window=lookback_period).min()
        h14 = high.rolling(window=lookback_period).max()
        k = 100 * ((close - l14) / (h14 - l14))
        # Calculate the %D line (3-day SMA of %K)
        d = k.rolling(window=3).mean()
        df = pd.DataFrame({"D": d, "K": k})
        return df

    def get_Fibonacci_Retracement(
        self, close: pd.Series, start_date: str = "", end_date: str = ""
    ):
        """
        Get Stochastic Oscillator based on a high, low, and close price series.

        Parameters
        ----------
        close : pd.Series
            Time series of closing stock prices.
        start_date : str, optional
            Start date for the calculation.
        end_date : str, optional
            End date for the calculation.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing Fibb level data.
        """

        if start_date != "" and end_date != "":
            close = close.loc[start_date:end_date]
        max_price = close.max()
        min_price = close.min()
        diff = max_price - min_price
        # Fibb levels
        level1 = max_price - 0.236 * diff
        level2 = max_price - 0.382 * diff
        level3 = max_price - 0.5 * diff
        level4 = max_price - 0.618 * diff
        level5 = max_price - 0.786 * diff
        # Dataframe to hold Fibb levels
        df = pd.DataFrame(
            {
                "index": close.index,
                "level1": level1,
                "level2": level2,
                "level3": level3,
                "level4": level4,
                "level5": level5,
                "min_price": min_price,
                "max_price": max_price,
            }
        ).set_index("index")
        return df

    def get_OBV(self, close: pd.Series, volume: pd.Series):

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def get_dividends(
        self,
        close: pd.Series,
        dividends: pd.Series,
        quarterly: bool = True,
        biannual: bool = False,
        monthly: bool = False,
    ):

        d_index = 0
        d_indexes = {}
        prev_timestamp = dividends.index[0]
        sections = []
        # Search indexes where a dividend occurs.
        for i in range(len(dividends)):
            current_timestamp = dividends.index[i]
            d = dividends.iloc[d_index]
            if d > 0:
                d_indexes[dividends.index[i]] = d.item()
                data = {
                    "start": prev_timestamp,
                    "end": current_timestamp,
                    "value": d.item(),
                }
                sections.append(data)
                prev_timestamp = current_timestamp
            d_index += 1

        # Add last section.
        sections.append(
            {
                "start": sections[-1]["end"],
                "end": dividends.index[-1],
                "value": sections[-1]["value"],
            }
        )
        df = pd.DataFrame(index=dividends.index, columns=["close", "dividend"])
        df["close"] = close
        df.index = pd.to_datetime(df.index)
        # df.index = pd.to_datetime(df.index)
        for s in sections:
            df.loc[s["start"] : s["end"], "dividend"] = s["value"]

        # Calculate the yield
        df["close"] = close
        df["yield"] = (df["dividend"] / df["close"]) * 100

        if quarterly:
            df["annual_yield"] = df["yield"] * 4
        elif biannual:
            df["annual_yield"] = df["yield"] * 2
        elif monthly:
            df["annual_yield"] = df["yield"] * 12
        # No parameter is chosen, default to quarterly, since that is the frequency of the majority of companies.
        else:
            df["annual_yield"] = df["yield"] * 4
        return df

    def _apply_all_indicators(self, df: pd.DataFrame):
        # RSI
        rsi = self.get_RSI(df["Close"])
        df["RSI"] = rsi["RSI"]
        df["RSI_MA"] = rsi["RSI_MA"]
        # MACD
        macd = self.get_MACD(df["Close"])
        df["MACD"] = macd["MACD"]
        df["Signal Line"] = macd["Signal Line"]
        df["Histogram"] = macd["Histogram"]
        # EMAs
        df = self.get_EMA_group(df)
        # Bollinger Bands
        bb = self.get_BB_Bands(df["Close"])
        df["upper_BB_band"] = bb["upper_BB_band"]
        df["lower_BB_band"] = bb["lower_BB_band"]
        # ATR Bands
        atr = self.get_ATR_Bands(df["High"], df["Low"], df["Close"])
        df["upper_ATR_band"] = atr["upper_ATR_band"]
        df["lower_ATR_band"] = atr["lower_ATR_band"]
        # Stochastic Oscillator
        oscillator = self.get_Stochastic_Oscillator(df["High"], df["Low"], df["Close"])
        df["D"] = oscillator["D"]
        df["K"] = oscillator["K"]
        # Fibonacci Retracements
        fib = self.get_Fibonacci_Retracement(df["Close"])
        df["fib_level1"] = fib["level1"]
        df["fib_level2"] = fib["level2"]
        df["fib_level3"] = fib["level3"]
        df["fib_level4"] = fib["level4"]
        df["fib_level5"] = fib["level5"]
        df["fib_min_price"] = fib["min_price"]
        df["fib_max_price"] = fib["max_price"]
        # Dividends
        dividends = self.get_dividends(df["Close"], df["Dividends"])
        df["dividend"] = dividends["dividend"]
        df["annual_yield"] = dividends["annual_yield"]

        return df

    def plot_graph(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        price_annotate: bool = False,
        rsi: tuple = (False, False),
        macd: tuple = (False, False),
        sma: tuple = (False, False),
        ema: tuple = (False, False),
        oscillator: tuple = (False, False),
        bb_bands: tuple = (False, False),
        fibonacci: tuple = (False, False),
        dividends: tuple = (False, False),
        ma_ranges=[9, 10, 20, 50, 100, 200],
    ):
        # Number of panels
        panel_num = 2
        # Holds additional plots
        add_plots = []
        df.index = pd.to_datetime(df.index, utc=True)
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]

        if price_annotate:
            price_peak_plot = self._create_annotations(
                df["High"], df["High"], True, "^", 25, 0, distance=10, prominence=2.5
            )
            price_dip_plot = self._create_annotations(
                df["Low"], df["Low"], False, "v", 20, 0, distance=10, prominence=2.5
            )
            add_plots.append(price_peak_plot)
            add_plots.append(price_dip_plot)

        # ----------- RSI Logic -----------
        if rsi[0]:
            rsi_plot = mpf.make_addplot(
                df["RSI"], panel=panel_num, color="blue", ylabel="RSI"
            )
            rsi_ma_plot = mpf.make_addplot(
                df["RSI_MA"], panel=panel_num, color="yellow"
            )
            rsi_panel = panel_num
            panel_num += 1
            add_plots.append(rsi_plot)
            add_plots.append(rsi_ma_plot)
            # Logic to add RSI annotations
            if rsi[1]:
                rsi_peak_plot = self._create_annotations(
                    df["RSI"], df["RSI"], True, "^", 25, rsi_panel
                )
                rsi_dip_plot = self._create_annotations(
                    df["RSI"], df["RSI"], False, "v", 20, rsi_panel
                )
                add_plots.append(rsi_peak_plot)
                add_plots.append(rsi_dip_plot)
        # ----------- MACD Logic -----------
        if macd[0]:
            macd_plot = mpf.make_addplot(
                df["MACD"], panel=panel_num, color="purple", ylabel="MACD"
            )
            signal_plot = mpf.make_addplot(
                df["Signal Line"], panel=panel_num, color="orange"
            )
            histogram_plot = mpf.make_addplot(
                df["Histogram"],
                type="bar",
                panel=panel_num,
                color=self._get_color_array(df["Histogram"]),
                alpha=0.5,
            )
            macd_panel = panel_num
            panel_num += 1
            add_plots.append(macd_plot)
            add_plots.append(signal_plot)
            add_plots.append(histogram_plot)

            if macd[1]:
                macd_peak_plot = self._create_annotations(
                    df["MACD"],
                    df["MACD"],
                    True,
                    "^",
                    25,
                    macd_panel,
                    distance=10,
                    prominence=1,
                )
                macd_dip_plot = self._create_annotations(
                    df["MACD"],
                    df["MACD"],
                    False,
                    "v",
                    20,
                    macd_panel,
                    distance=10,
                    prominence=1,
                )

                add_plots.append(macd_peak_plot)
                add_plots.append(macd_dip_plot)

        # ----------- EMA Logic -----------
        if ema[0]:
            index = 0
            for i in ma_ranges:
                column = f"EMA-{i}"
                try:
                    ema = df[column]
                    p = mpf.make_addplot(ema, panel=0, color=self.colors[index])
                    add_plots.append(p)

                    if ema[1]:
                        ema_peak_plot = self._create_annotations(
                            df[column], df[column], True, "^", 25, 0
                        )
                        ema_dip_plot = self._create_annotations(
                            df[column], df[column], False, "v", 20, 0
                        )
                        add_plots.append(ema_peak_plot)
                        add_plots.append(ema_dip_plot)
                        print("TAG")
                    index += 1
                except KeyError:
                    pass
            # NOTE: Function does not require incrementing "panel_num" since this indicator is placed in panel 0.
        # ----------- Oscillator Logic -----------
        if oscillator[0]:
            d_plot = mpf.make_addplot(
                df["D"], panel=panel_num, color="white", ylabel="Stochastic Oscillator"
            )
            k_plot = mpf.make_addplot(df["K"], panel=panel_num, color="yellow")
            oscillator_panel = panel_num
            panel_num += 1
            add_plots.append(d_plot)
            add_plots.append(k_plot)

            if oscillator[1]:
                oscillator_peak_plot = self._create_annotations(
                    df["D"], df["D"], True, "^", 25, oscillator_panel
                )
                oscillator_dip_plot = self._create_annotations(
                    df["D"], df["D"], False, "v", 20, oscillator_panel
                )
                add_plots.append(oscillator_peak_plot)
                add_plots.append(oscillator_dip_plot)

        # ----------- Bollinger Bands Logic -----------
        if bb_bands[0]:
            upper_band = mpf.make_addplot(df["upper_BB_band"], panel=0, color="gold")
            lower_band = mpf.make_addplot(df["lower_BB_band"], panel=0, color="yellow")
            add_plots.append(upper_band)
            add_plots.append(lower_band)
            # NOTE: Function does not require incrementing "panel_num" since this indicator is placed in panel 0.
        # ----------- Fibonacci Retracement Logic -----------
        if fibonacci[0]:
            fib_levels = [
                "fib_min_price",
                "fib_level1",
                "fib_level2",
                "fib_level3",
                "fib_level4",
                "fib_level5",
                "fib_max_price",
            ]
            index = 0
            for level in fib_levels:
                if level == "fib_min_price" or level == "fib_max_price":
                    color = "blue"
                    linestyle = "solid"
                else:
                    color = self.colors[index]
                    linestyle = "dashed"
                    index += 1

                fib_plot = mpf.make_addplot(
                    df[level],
                    panel=0,
                    color=color,
                    linestyle=linestyle,
                    secondary_y=False,
                )
                add_plots.append(fib_plot)
        # ----------- Dividends Logic -----------
        if dividends[0]:
            dividend = mpf.make_addplot(
                df["dividend"],
                panel=panel_num,
                color="blue",
                linestyle="solid",
                ylabel="Dividend",
            )
            dividend_annual_yield = mpf.make_addplot(
                df["annual_yield"],
                panel=panel_num,
                color="purple",
                linestyle="dashed",
                ylabel="Yield",
            )
            dividends_panel = panel_num
            panel_num += 1
            add_plots.append(dividend)
            add_plots.append(dividend_annual_yield)

            if dividends[1]:
                print("TAGGGG")
                print(f"DF: {df['annual_yield']}")
                yield_peak_plot = self._create_annotations(
                    df["annual_yield"],
                    df["annual_yield"],
                    True,
                    "^",
                    25,
                    dividends_panel,
                    distance=10,
                    prominence=0.05,
                )
                yield_dip_plot = self._create_annotations(
                    df["annual_yield"],
                    df["annual_yield"],
                    False,
                    "v",
                    20,
                    dividends_panel,
                    distance=10,
                    prominence=0.05,
                )
                add_plots.append(yield_peak_plot)
                add_plots.append(yield_dip_plot)

        # Create plot
        mpf.plot(
            ohlcv,
            type="candle",
            style=binance_dark,
            title="{} - OHLCV With Indicators".format(ticker.upper()),
            ylabel="Price",
            addplot=add_plots,
            volume=True,
            returnfig=True,
        )

        plt.show()

    def _get_local_maxima(
        self, values: pd.Series, distance: int = 10, prominence: int = 5
    ):
        peaks, _ = Signal.find_peaks(
            values, distance=distance, prominence=prominence, height=None, width=None
        )
        return peaks

    def _get_local_minima(
        self, values: pd.Series, distance: int = 10, prominence: int = 5
    ):
        inverted = -values
        minima, _ = Signal.find_peaks(
            inverted, distance=distance, prominence=prominence, height=None, width=None
        )
        return minima

    def _create_markers(
        self, series_indexes: list, marker_indexes: list, marker_label: str
    ):
        index = 0
        markers = []
        for i in series_indexes:
            if index in marker_indexes:
                markers.append(marker_label)
            else:
                # markers.append(None)
                markers.append("")
            index += 1
        return markers

    def _create_annotations(
        self,
        values: pd.Series,
        anchor_values: pd.Series,
        maxima: bool,
        marker_label: str,
        marker_size: int,
        panel_num: int,
        distance: int = 10,
        prominence: int = 5,
    ):

        if maxima:
            m = self._get_local_maxima(values, distance, prominence)
            color = "green"
            anchor_multiplier = 1.02
        else:
            m = self._get_local_minima(values, distance, prominence)
            color = "red"
            anchor_multiplier = 0.99
        markers = self._create_markers(values.index.to_list(), m, marker_label)
        plot = mpf.make_addplot(
            anchor_multiplier * anchor_values,
            panel=panel_num,
            type="scatter",
            marker=markers,
            markersize=marker_size,
            color=color,
        )
        return plot

    def _get_color_array(self, df: pd.Series):
        colors = ["green" if value >= 0 else "red" for value in df]
        return colors

    def _get_delta(self, period: int, period_unit: str):
        """
        Create a "timedelta" for date calculations.

        Parameters
        ----------
        period : int
            Number of periods.
        period_unit : str
            The unit of the period. For example, if period=5, and period_unit="Y", then the full period will be 5 years.

        Returns
        -------
        dt.timedelta
            Time delta with the adjusted amount of days according to the 'period_unit'.
        """
        if period_unit == "Y":
            return dt.timedelta(days=(365 * period))
        elif period_unit == "M":
            return dt.timedelta(days=(30 * period))
        elif period_unit == "D":
            return dt.timedelta(days=period)


if __name__ == "__main__":
    ticker = "AAPL"
    ta = TechnicalAnalysis()
    end = dt.datetime.now()
    start = end - ta._get_delta(5, "Y")
    df = yf.download(ticker, start=start, end=end, actions=True)
    # fib = ta.get_Fibonacci_Retracement(df["Close"])
    # df = ta.get_dividends(df["Close"], df["Dividends"])
    df = ta._apply_all_indicators(df)
    # # ta._test_plot(df)
    ta.plot_graph(
        df,
        ticker=ticker,
        price_annotate=True,
        rsi=(True, True),
        macd=(True, True),
        oscillator=(False, False),
        bb_bands=(True, False),
        fibonacci=(False, False),
        dividends=(False, False),
    )
