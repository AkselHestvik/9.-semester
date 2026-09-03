import numpy as np
import pandas as pd


def last_temperatur(path, date_col="dato", temp_col="temperatur"):
    """CSV/Excel -> daglig temperaturserie. Tåler hull og ujevn frekvens."""
    df = pd.read_csv(path, parse_dates=[date_col])
    s = (df.set_index(date_col)[temp_col]
           .sort_index()
           .pipe(lambda x: x[~x.index.duplicated(keep="first")])
           .resample("D").mean()
           .interpolate("time"))
    return s


def klimatologi(daglig):
    """Normalår: middeltemperatur per dag-i-året, glattet."""
    clim = daglig.groupby(daglig.index.dayofyear).mean().reindex(range(1, 367))
    clim = (pd.concat([clim, clim, clim])          # wrap rundt nyttår
              .interpolate()
              .rolling(15, center=True, min_periods=1).mean()
              .iloc[366:732])
    clim.index = range(1, 367)
    return clim


def lag_theta(daglig, modus="hybrid"):
    """Returnerer theta(dato) -> grader C.
    'faktisk'     : kun observert serie (feiler utenfor perioden)
    'klimatologi' : normalår, repeteres i det uendelige
    'hybrid'      : observert der den finnes, ellers normalår
    """
    clim = klimatologi(daglig)
    t0, t1 = daglig.index[0], daglig.index[-1]

    def theta(dato):
        d = pd.Timestamp(dato)
        if modus != "klimatologi" and t0 <= d <= t1:
            v = daglig.get(d, np.nan)
            if not np.isnan(v):
                return float(v)
            if modus == "faktisk":
                raise KeyError(d)
        return float(clim.loc[min(d.dayofyear, 366)])

    return theta
