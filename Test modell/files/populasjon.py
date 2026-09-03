"""Populasjonslag: antall fisk, biomasse, MAB-beskrankning og salgsinntekter.

Kjører daglig. Vekst hentes fra Vekstmodell, som følger én gjennomsnittsfisk.
Biomasse er B(d) = N(d) * W(d). Overstiger den MAB, slaktes overskuddet ut
samme dag. Sluttslakt når gjennomsnittsvekten passerer målvekten.
"""
import numpy as np
import pandas as pd

# MTB per tillatelse (tonn). Kilde: laksetildelingsforskriften / Fiskeridirektoratet.
MAB_PER_TILLATELSE = {
    "standard": 780.0,          # hovedregel
    "troms_finnmark": 945.0,    # Troms og Finnmark
}


def velg_mab(region="standard", antall_tillatelser=1, lokalitets_mab=None):
    """Bindende MAB i tonn.

    To nivåer: sum av tillatelsene knyttet til lokaliteten, og lokalitets-MTB.
    Den laveste binder. Er lokalitets_mab ukjent, brukes tillatelsessummen.
    """
    if region not in MAB_PER_TILLATELSE:
        raise ValueError(f"ukjent region '{region}', velg blant {list(MAB_PER_TILLATELSE)}")
    fra_tillatelser = MAB_PER_TILLATELSE[region] * antall_tillatelser
    return fra_tillatelser if lokalitets_mab is None else min(fra_tillatelser, lokalitets_mab)


def daglig_taprate(tap_total, dager):
    """Total andel tapt over syklusen -> daglig rate. 0.125 over 480 d -> 0.000278/d."""
    if not 0 <= tap_total < 1:
        raise ValueError("tap_total må ligge i [0, 1)")
    return 1.0 - (1.0 - tap_total) ** (1.0 / dager)


class Populasjonsmodell:
    def __init__(self, vekstmodell, n0=200_000, mab_tonn=780.0,
                 tap_total=0.125, syklus_dager=480, w_slakt=5000.0,
                 pris_kr_kg=72.0, maks_dager=900):
        self.vm = vekstmodell
        self.n0, self.mab_kg = n0, mab_tonn * 1000.0
        self.tap_d = daglig_taprate(tap_total, syklus_dager)
        self.w_slakt, self.pris, self.maks_dager = w_slakt, pris_kr_kg, maks_dager

    def simuler(self, w0, utsett):
        """Daglig simulering. Returnerer (DataFrame per dag, dict med nøkkeltall)."""
        w, n = float(w0), float(self.n0)
        d = pd.Timestamp(utsett)
        rader = []
        slaktet_mab = slaktet_slutt = 0.0     # kg
        inntekt_mab = inntekt_slutt = 0.0     # kr
        dode = 0.0

        for _ in range(self.maks_dager):
            w, _for_g, miljo = self.vm.dag(w, d)

            d_dod = n * self.tap_d
            n -= d_dod
            dode += d_dod

            w_kg = w / 1000.0                 # tabellen er i gram, MAB i kg

            # MAB: slakt ut overskuddet samme dag
            utslakt_n = 0.0
            if n * w_kg > self.mab_kg:
                utslakt_n = (n * w_kg - self.mab_kg) / w_kg
                n -= utslakt_n
                slaktet_mab += utslakt_n * w_kg
                inntekt_mab += utslakt_n * w_kg * self.pris

            sluttslakt = w >= self.w_slakt
            if sluttslakt:
                slaktet_slutt += n * w_kg
                inntekt_slutt += n * w_kg * self.pris

            rader.append({"dato": d, "w_g": w, "N": n, "biomasse_kg": n * w_kg,
                          "utslakt_n": utslakt_n, "dode_n": d_dod, "miljo": miljo})
            if sluttslakt:
                n = 0.0
                break
            d += pd.Timedelta(days=1)

        df = pd.DataFrame(rader)
        dager = len(df)
        nøkkel = {
            "dager_i_sjo": dager,
            "mnd_i_sjo": round(dager / 30, 1),
            "sluttvekt_g": round(df.w_g.iloc[-1]),
            "N_start": self.n0,
            "N_dode": round(dode),
            "dodelighet_pct": round(100 * dode / self.n0, 1),
            "N_utslaktet_mab": round(df.utslakt_n.sum()),
            "dager_paa_mab": int((df.biomasse_kg >= self.mab_kg * 0.999).sum()),
            "maks_biomasse_tonn": round(df.biomasse_kg.max() / 1000, 1),
            "slaktet_mab_tonn": round(slaktet_mab / 1000, 1),
            "slaktet_slutt_tonn": round(slaktet_slutt / 1000, 1),
            "slaktet_totalt_tonn": round((slaktet_mab + slaktet_slutt) / 1000, 1),
            "inntekt_mab_kr": round(inntekt_mab),
            "inntekt_slutt_kr": round(inntekt_slutt),
            "inntekt_totalt_kr": round(inntekt_mab + inntekt_slutt),
        }
        return df, nøkkel
