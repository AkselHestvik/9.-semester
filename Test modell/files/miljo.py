"""Multiplikative miljofaktorer paa vekst. Alle returnerer verdi i [0,1], 1 = ingen begrensning.

Kilder:
  Remen et al. (2016, Aquaculture 464:582)  DO_maxFI: ~42 % O2 ved 7 C -> ~76 % ved 19 C
  Remen et al. (2013)                       LOS:      ~30 % O2 ved 6 C -> ~55 % ved 19 C
  Remen et al. (2012)                       70 % terskel / 60 % velferdsgrense ved 16 C
  Crampton et al. (2003)                    tilnaermet lineaer vekstrespons 50-100 % O2
  Boeuf & Gaignon (1989), Jorgensen & Jobling (1993)  stromhastighet paavirker vekst (kvalitativt)
"""
import numpy as np


def o2_metning(do_mgl, T, S):
    """mg/l -> % av luftmetning. Garcia & Gordon (1992) losningskurve."""
    Ts = np.log((298.15 - T) / (273.15 + T))
    A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    lnC = (A[0] + A[1]*Ts + A[2]*Ts**2 + A[3]*Ts**3 + A[4]*Ts**4 + A[5]*Ts**5
           + S*(B[0] + B[1]*Ts + B[2]*Ts**2 + B[3]*Ts**3) - 2.75915e-7*S**2)
    return 100.0 * do_mgl / (np.exp(lnC) * 0.032)


def DO_maxFI(T):
    """Metning der forinntaket slutter aa vaere maksimalt. Remen et al. (2016)."""
    return 42.0 + (76.0 - 42.0) / (19.0 - 7.0) * (np.clip(T, 7, 19) - 7.0)


def LOS(T):
    """Limiting oxygen saturation. Remen et al. (2013)."""
    return 30.0 + (55.0 - 30.0) / (19.0 - 6.0) * (np.clip(T, 6, 19) - 6.0)


def f_o2(metning, T, uttak=0.0):
    """Lineaer nedtrapping fra DO_maxFI(T) til LOS(T).
    uttak: prosentpoeng metning fisken selv forbruker i merden (ambient -> in-cage)."""
    s = np.asarray(metning, float) - uttak
    lo, hi = LOS(T), DO_maxFI(T)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def f_sal(S, S_opt=32.0, S_min=20.0):
    """Postsmolt i full sjo er upaavirket. Kun grov straff ved kraftig brakkvann.
    ADVARSEL: ingen publisert dose-respons bak denne. Bruk som sensitivitet, ikke som estimat."""
    return np.clip((np.asarray(S, float) - S_min) / (S_opt - S_min), 0.0, 1.0)


def kroppslengde(w_g):
    """Grov lengde-vekt: W[g] = 0.01 * L[cm]^3  ->  L i meter."""
    return (np.asarray(w_g, float) / 0.01) ** (1/3) / 100.0


def f_strom(v_ms, w_g, v_opt=1.0, v_maks=3.0, tap=0.10):
    """Stromhastighet i kroppslengder/s. Optimum ~1 BL/s, fallende over ~3 BL/s.
    ADVARSEL: formen er antatt. Litteraturen er retningsgivende, ikke kvantitativ.
    'tap' er maksimalt vekstap ved null strom - hold den som sensitivitetsparameter."""
    bl = np.asarray(v_ms, float) / kroppslengde(w_g)
    under = 1.0 - tap * np.clip((v_opt - bl) / v_opt, 0.0, 1.0)
    over = 1.0 - tap * np.clip((bl - v_maks) / v_maks, 0.0, 1.0)
    return np.where(bl < v_opt, under, over)
