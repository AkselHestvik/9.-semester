"""Ende-til-ende-demo.

  1. Lokalitetsdata -> temperaturserie
  2. Vekstbane for én kohort + G-, W- og fôrmatriser
  3. Populasjonslag: N, biomasse, MAB og salgsinntekter
  4. Stokastisk kjøring med blokk-bootstrap
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from site_data import les_site, dybdesnitt
from temperatur import lag_theta
from vekst import Vekstmodell
from populasjon import Populasjonsmodell, velg_mab
from bootstrap import MiljoBootstrap, log_serie

ROT = Path(__file__).resolve().parent.parent
DATA = ROT / "data"
STI = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "site_32597_raw.csv"

# --- forutsetninger --------------------------------------------------------
W_SMOLT = 100.0            # g
DYBDE = (1, 15)            # dybdeintervall fisken antas å oppholde seg i
UTSETT = "2021-04-15"
N0 = 200_000               # antall smolt
REGION = "standard"        # 'standard' (780 t) eller 'troms_finnmark' (945 t)
ANTALL_TILLATELSER = 1
TAP_TOTAL = 0.125          # samlet dødelighet over syklusen
W_SLAKT = 5000.0           # g
PRIS = 72.0                # kr/kg
N_SIM = 200                # antall bootstrap-kjøringer


def vis(M, fmt="{:.2f}"):
    """Matrise til tekst, med 'slaktet' der kohorten er ferdig."""
    return M.map(lambda x: fmt.format(x) if pd.notna(x) else "slaktet").to_string()


def tittel(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


df = les_site(STI)
temp = dybdesnitt(df, "temp", DYBDE)
modell = Vekstmodell(lag_theta(temp, modus="klimatologi"), kappa=1.0)   # f_* = 1

# --- 1. én kohort ----------------------------------------------------------
tittel(f"VEKSTBANE — utsett {UTSETT}, {W_SMOLT:.0f} g smolt")
bane = modell.bane(W_SMOLT, UTSETT)
print(bane[["periode", "dato", "theta", "G", "w_slutt", "for_g", "miljo"]]
      .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
n5 = bane.loc[bane.w_slutt >= 5000, "periode"]
print(f"\n5 kg etter {int(n5.min())} perioder | sluttvekt {bane.w_slutt.iloc[-1]:.0f} g "
      f"| FCR {bane.for_g.sum() / (bane.w_slutt.iloc[-1] - W_SMOLT):.3f}")

# --- 2. alle utsettstidspunkt ---------------------------------------------
utsett_alle = pd.date_range("2021-01-15", periods=12, freq="MS")
tittel("MATRISER (utsett x periode i sjø)")
print("\nVekstfaktor G:")
print(vis(modell.G_matrise(W_SMOLT, utsett_alle), "{:.3f}"))
print("\nVekt i kg ved slutten av hver periode:")
print(vis(modell.W_matrise(W_SMOLT, utsett_alle)))
print("\nFôr i kg per periode (per fisk):")
print(vis(modell.matrise(W_SMOLT, utsett_alle, "for_g") / 1000))

# --- 3. populasjon, MAB og inntekter --------------------------------------
mab = velg_mab(REGION, ANTALL_TILLATELSER)
tittel(f"POPULASJON — {N0:,} smolt, MAB {mab:.0f} tonn ({REGION})".replace(",", " "))
pm = Populasjonsmodell(modell, n0=N0, mab_tonn=mab, tap_total=TAP_TOTAL,
                       w_slakt=W_SLAKT, pris_kr_kg=PRIS)
dag, k = pm.simuler(W_SMOLT, UTSETT)

for key in ["mnd_i_sjo", "sluttvekt_g", "dodelighet_pct", "maks_biomasse_tonn",
            "dager_paa_mab", "N_utslaktet_mab", "slaktet_mab_tonn",
            "slaktet_slutt_tonn", "slaktet_totalt_tonn"]:
    print(f"  {key:22s} {k[key]:>14,}".replace(",", " "))
print(f"  {'-' * 37}")
for key in ["inntekt_mab_kr", "inntekt_slutt_kr", "inntekt_totalt_kr"]:
    print(f"  {key:22s} {k[key]:>14,} kr".replace(",", " "))

print("\n  Biomasse per måned (tonn):")
mnd = dag.set_index("dato").biomasse_kg.resample("MS").last() / 1000
print("   " + "  ".join(f"{d:%b}:{v:.0f}" for d, v in mnd.items()))

# --- 4. stokastisk: blokk-bootstrap ---------------------------------------
tittel(f"STOKASTISK — {N_SIM} bootstrap-kjøringer")
bs = MiljoBootstrap({"temp": temp, "lncop": log_serie(dybdesnitt(df, "cop", DYBDE))},
                    skift_sd={"temp": 0.5, "lncop": 0.5}, seed=42)

res = []
for _ in range(N_SIM):
    b = bs.trekk(UTSETT, 900)
    vm_i = Vekstmodell(bs.theta_fra(b), kappa=1.0)
    _, ki = Populasjonsmodell(vm_i, n0=N0, mab_tonn=mab, tap_total=TAP_TOTAL,
                              w_slakt=W_SLAKT, pris_kr_kg=PRIS).simuler(W_SMOLT, UTSETT)
    res.append(ki)

r = pd.DataFrame(res)
kols = ["mnd_i_sjo", "slaktet_totalt_tonn", "inntekt_totalt_kr", "dager_paa_mab"]
print(r[kols].describe(percentiles=[.05, .5, .95])
       .loc[["mean", "std", "5%", "50%", "95%"]]
       .to_string(float_format=lambda x: f"{x:>14,.1f}".replace(",", " ")))
print(f"\n  Andel kjøringer der MAB binder: {(r.dager_paa_mab > 0).mean() * 100:.0f} %")
