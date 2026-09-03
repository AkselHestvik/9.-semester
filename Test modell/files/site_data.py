"""Innlasting av lokalitetsdata på site_XXXXX_raw.csv-format (dato x dybde x variabel)."""
import numpy as np
import pandas as pd

KOL = {"temp": "meanTemp", "sal": "meanSal", "strom": "meanCurrSpd",
       "strom95": "95PercCurrSpd", "cop": "meanCopDensity",
       "bolge": "SignWaveHeight", "o2": "do"}


def les_site(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.rename(columns={"date": "dato", "depth": "dybde"}).sort_values(["dato", "dybde"])


def dybdesnitt(df, variabel="temp", dybde=(1, 15), vekter=None):
    """Daglig serie, midlet over et dybdeintervall. vekter: dict dybde->vekt."""
    kol = KOL.get(variabel, variabel)
    d = df[df.dybde.between(*dybde)]
    if vekter is None:
        s = d.groupby("dato")[kol].mean()
    else:
        w = d.dybde.map(vekter).fillna(0.0)
        s = (d[kol] * w).groupby(d.dato).sum() / w.groupby(d.dato).sum()
    return s.asfreq("D").interpolate("time")


def dybdeprofil(df, variabel="temp"):
    """Pivot dato x dybde — for plotting og for å se lagdelingen."""
    return df.pivot(index="dato", columns="dybde", values=KOL.get(variabel, variabel))


def dekning(df):
    return (df.notna().mean().rename("andel_ikke_null").round(3).to_frame()
              .assign(dybder=[df.loc[df[c].notna(), "dybde"].nunique() for c in df.columns]))
