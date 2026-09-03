"""Vekstmodell: Skretting-tabell (Fosund & Strandkleiv / NTNU) med multiplikative miljofaktorer.

    dW/dt = W * kappa * [SGR(W, Theta) / 100] * f_O2 * f_sal * f_strom

    Delingen paa 100 er enhetskonvertering (tabellen er i prosent per dogn).
    kappa er en dimensjonslos realiseringsfaktor og hoerer ikke sammen med den.

Alle f-funksjonene returnerer 1.0 i denne versjonen. De er lagt inn som
utskiftbare argumenter slik at de kan aktiveres uten at resten av modellen endres.
Se miljo.py for kandidatformer med kilder.
"""
import numpy as np
import pandas as pd

from veksttabell import sgr, fcr


# --- miljofaktorer: nøytrale inntil videre -----------------------------------
def f_o2(**kw):     return 1.0     # oksygen
def f_sal(**kw):    return 1.0     # salinitet
def f_strom(**kw):  return 1.0     # stromhastighet


class Vekstmodell:
    """theta(dato) -> °C er eneste obligatoriske input.
    De ovrige miljoseriene er valgfrie og brukes kun av f-funksjonene."""

    def __init__(self, theta, kappa=1.0, f_o2=f_o2, f_sal=f_sal, f_strom=f_strom,
                 o2=None, sal=None, strom=None):
        self.theta, self.kappa = theta, kappa
        self.f_o2, self.f_sal, self.f_strom = f_o2, f_sal, f_strom
        self.o2, self.sal, self.strom = o2, sal, strom

    def _miljo(self, d, w, T):
        """Samlet multiplikator. Argumentene sendes til alle f-ene som kwargs."""
        kw = dict(dato=d, w=w, T=T,
                  o2=None if self.o2 is None else float(self.o2.get(d, np.nan)),
                  sal=None if self.sal is None else float(self.sal.get(d, np.nan)),
                  strom=None if self.strom is None else float(self.strom.get(d, np.nan)))
        return self.f_o2(**kw) * self.f_sal(**kw) * self.f_strom(**kw)

    def dag(self, w, d):
        """Ett dogn. Returnerer (ny vekt, forforbruk g, brukt miljofaktor)."""
        T = self.theta(d)
        m = self._miljo(d, w, T)
        dw = w * self.kappa * m * sgr(w, T) / 100.0
        return w + dw, dw * fcr(w), m

    def periode(self, w0, start, dager=30):
        w, fôr, ms = float(w0), 0.0, []
        d = pd.Timestamp(start)
        for _ in range(dager):
            w, f, m = self.dag(w, d)
            fôr += f; ms.append(m); d += pd.Timedelta(days=1)
        return w, fôr, float(np.mean(ms))

    def bane(self, w0, utsett, dager=30, maks_perioder=24, w_maks=5500.0):
        rader, w, d = [], float(w0), pd.Timestamp(utsett)
        for p in range(1, maks_perioder + 1):
            w_ny, fôr, m = self.periode(w, d, dager)
            rader.append({"periode": p, "dato": d, "w_start": w, "w_slutt": w_ny,
                          "G": w_ny / w, "for_g": fôr, "miljo": m,
                          "theta": np.mean([self.theta(d + pd.Timedelta(days=i))
                                            for i in range(dager)])})
            w, d = w_ny, d + pd.Timedelta(days=dager)
            if w >= w_maks:
                break
        return pd.DataFrame(rader)

    def matrise(self, w0, utsettsdatoer, kolonne="G", **kw):
        """Rader = utsettsdato, kolonner = periode i sjo (1 = forste maned).
        kolonne: 'G' (vekstfaktor), 'w_slutt' (vekt i g), 'for_g' (for i g),
                 'theta' (snittemperatur), 'miljo' (samlet f-faktor)."""
        baner = {pd.Timestamp(u).date(): self.bane(w0, u, **kw).set_index("periode")
                 for u in utsettsdatoer}
        return pd.DataFrame({u: b[kolonne] for u, b in baner.items()}).T

    def G_matrise(self, w0, utsettsdatoer, **kw):
        return self.matrise(w0, utsettsdatoer, "G", **kw)

    def W_matrise(self, w0, utsettsdatoer, enhet="kg", **kw):
        """Vekt ved slutten av hver periode."""
        W = self.matrise(w0, utsettsdatoer, "w_slutt", **kw)
        return W / 1000.0 if enhet == "kg" else W
