"""Ende-til-ende-demo: lokalitetsdata -> temperaturserie -> vekstbane og matriser."""
from pathlib import Path
import sys

import pandas as pd

from site_data import les_site, dybdesnitt
from temperatur import lag_theta
from vekst import Vekstmodell

ROT = Path(__file__).resolve().parent.parent
DATA = ROT / "data"
STI = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "site_32597_raw.csv"

W_SMOLT, DYBDE = 100.0, (1, 15)


def vis(M, fmt="{:.2f}"):
    """Matrise til tekst, med 'slaktet' der kohorten er ferdig."""
    return M.map(lambda x: fmt.format(x) if pd.notna(x) else "slaktet").to_string()


df = les_site(STI)
theta = lag_theta(dybdesnitt(df, "temp", DYBDE), modus="klimatologi")
modell = Vekstmodell(theta, kappa=1.0)          # f_O2 = f_sal = f_strom = 1

# --- én kohort -------------------------------------------------------------
bane = modell.bane(W_SMOLT, "2021-04-15")
print(bane[["periode", "dato", "theta", "G", "w_slutt", "for_g", "miljo"]]
      .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

n5 = bane.loc[bane.w_slutt >= 5000, "periode"]
print(f"\n5 kg etter {int(n5.min())} perioder | sluttvekt {bane.w_slutt.iloc[-1]:.0f} g "
      f"| FCR {bane.for_g.sum() / (bane.w_slutt.iloc[-1] - W_SMOLT):.3f}")

# --- alle utsettstidspunkt -------------------------------------------------
utsett = pd.date_range("2021-01-15", periods=12, freq="MS")

print("\nVekstfaktor G (utsett x periode i sjø):")
print(vis(modell.G_matrise(W_SMOLT, utsett), "{:.3f}"))

print("\nVekt i kg ved slutten av hver periode:")
print(vis(modell.W_matrise(W_SMOLT, utsett)))

print("\nFôr i kg per periode (per fisk):")
print(vis(modell.matrise(W_SMOLT, utsett, "for_g") / 1000))
