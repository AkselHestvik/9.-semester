"""Blokk-bootstrap av miljøserier for stokastisk simulering.

Trekker HELE kalenderår fra observerte data og limer dem sammen til en
sammenhengende bane. Bevarer sesongform, autokorrelasjon og samvariasjonen
mellom variablene, siden alle verdier for en gitt dag hentes fra samme dato.

Alternativet, å fitte marginalfordelinger og trekke uavhengig, bryter alle tre.
"""
import numpy as np
import pandas as pd


class MiljoBootstrap:
    """serier: dict {navn: pd.Series med DatetimeIndex}, f.eks.
       {'temp': ..., 'cop': ...}. Alle må dekke samme datoer."""

    def __init__(self, serier, skift_sd=None, seed=None):
        self.serier = {k: v.sort_index() for k, v in serier.items()}
        self.skift_sd = skift_sd or {}
        self.rng = np.random.default_rng(seed)
        self.ar = sorted({a for s in self.serier.values() for a in s.index.year.unique()})
        self._tabell = self._bygg_tabell()

    def _bygg_tabell(self):
        """{år: DataFrame indeksert på dag-i-året 1..366}. Skuddår fylles ut."""
        ut = {}
        for a in self.ar:
            d = pd.DataFrame({k: v[v.index.year == a] for k, v in self.serier.items()})
            if d.dropna(how="all").empty:
                continue
            d.index = d.index.dayofyear
            ut[a] = d.reindex(range(1, 367)).interpolate().bfill().ffill()
        return ut

    def hele_ar(self):
        """Årene som har komplett nok dekning til å trekkes."""
        return [a for a, d in self._tabell.items() if d.notna().all().all()]

    def trekk(self, start, dager, skift=True):
        """Én bane. Returnerer DataFrame indeksert på faktiske datoer fra 'start'.
        Hvert kalenderår i banen får sitt eget tilfeldig trukne kildeår."""
        kilder = self.hele_ar()
        if not kilder:
            raise ValueError("ingen år med komplett dekning")

        datoer = pd.date_range(start, periods=dager, freq="D")
        valg = {a: int(self.rng.choice(kilder)) for a in datoer.year.unique()}

        rader = [self._tabell[valg[d.year]].loc[min(d.dayofyear, 366)] for d in datoer]
        ut = pd.DataFrame(rader).set_index(datoer)

        if skift:
            for k, sd in self.skift_sd.items():
                if k in ut.columns and sd:
                    ut[k] = ut[k] + self.rng.normal(0.0, sd)
        ut.attrs["kildear"] = valg
        return ut

    def theta_fra(self, bane, kolonne="temp"):
        """Gjør en trukket bane om til en theta(dato)-funksjon for Vekstmodell."""
        s = bane[kolonne]
        return lambda d: float(s.loc[pd.Timestamp(d)])


def log_serie(s, gulv=1e-5):
    """Log-transform med gulv. For høyreskjeve serier som kopepodtetthet."""
    return np.log(s.clip(lower=0) + gulv)


def eksp_serie(s, gulv=1e-5):
    return np.exp(s) - gulv
