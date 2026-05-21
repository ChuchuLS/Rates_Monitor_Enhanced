"""
config/tickers.py
=================
Single source of truth for *what data the dashboard knows about*.

`TICKERS` maps a stable internal key (e.g. "SOFR") to the exact Bloomberg
column name as it appears in DATA.xlsx / latest.parquet. Everything downstream
references the key, never the raw Bloomberg string, so a vendor renaming a
ticker only needs a one-line change here.

MOVE / VIX are intentionally included even though they are *not* in the current
data file. They demonstrate the dashboard's missing-ticker handling: the Data
Quality panel flags them as absent and the index engine simply skips them.
"""

from __future__ import annotations

TICKERS: dict[str, str] = {
    # --- Curve slopes — 2s10s ---------------------------------------------
    "US_2s10s": "USYC2Y10 INDEX",
    "DE_2s10s": "DEYC2Y10 INDEX",
    "JP_2s10s": "JPYC2Y10 INDEX",
    "AU_2s10s": "AUYC2Y10 INDEX",
    "UK_2s10s": "UKYC2Y10 INDEX",
    "CA_2s10s": "CAYC2Y10 INDEX",
    # --- Curve slopes — 5s30s ---------------------------------------------
    "US_5s30s": "USYC5Y30 INDEX",
    "DE_5s30s": "DE020510 INDEX",
    "JP_5s30s": "JPYC1030 INDEX",
    "AU_5s30s": "AD020510 INDEX",
    "UK_5s30s": "UK020510 INDEX",
    "CA_5s30s": "CB020510 INDEX",
    # --- Real rates — 10Y --------------------------------------------------
    "US_real_10y": "GTII10 GOVT",
    "CA_real_10y": "GTCADII10Y GOVT",
    "DE_real_10y": "GTDEMII10Y GOVT",
    "UK_real_10y": "GTGBPII10Y GOVT",
    "JP_real_10y": "GTJPYII10Y GOVT",
    "AU_real_10y": "GTAUDII10YR GOVT",
    # --- Real rates — full term structure ---------------------------------
    "US_real_5y":  "GTII5 GOVT",       "US_real_30y": "GTII30 GOVT",
    "UK_real_5y":  "GTGBPII5Y GOVT",   "UK_real_30y": "GTGBPII30Y GOVT",
    "DE_real_3y":  "GTDEMII3Y GOVT",   "DE_real_7y":  "GTDEMII7Y GOVT",
    "DE_real_25y": "GTDEMII25Y GOVT",
    "JP_real_5y":  "GTJPYII5Y GOVT",   "JP_real_7y":  "GTJPYII7Y GOVT",
    "AU_real_5y":  "GTAUDII5YR GOVT",
    "CA_real_5y":  "GTCADII5Y GOVT",   "CA_real_30y": "GTCADII30Y GOVT",
    # --- Money-market funding rates ---------------------------------------
    "SOFR": "SOFRRATE INDEX",
    "IORB": "IRRBIOER INDEX",
    "EFFR": "FEDL01 INDEX",
    "GCF":  "UREPGATO INDEX",
    "TGCR": "TGCRRATE INDEX",
    "RRP":  "FDTRFTRL INDEX",
    "BGCR": "USBGRATE INDEX",
    "TPR":  "UREPTATO INDEX",
    # --- XCCY basis swaps (3M) --------------------------------------------
    "XCCY_EUR": "EUXOQQC CURNCY",
    "XCCY_GBP": "BPXOQQC CURNCY",
    "XCCY_JPY": "JYBSS3M CURNCY",
    "XCCY_CAD": "CDXOQQC CURNCY",
    "XCCY_AUD": "ADBSQQC CURNCY",
    # --- XCCY basis swaps (12M) -------------------------------------------
    "XCCY12_EUR": "EUXOQQ1 CURNCY",
    "XCCY12_GBP": "BPXOQQ1 CURNCY",
    "XCCY12_JPY": "JYBSS12M CURNCY",
    "XCCY12_CAD": "CDXOQQ1 CURNCY",
    "XCCY12_AUD": "ADBSQQ1 CURNCY",
    # --- Credit ------------------------------------------------------------
    "IG_OAS":     "LUACOAS INDEX",
    "HY_OAS":     "LF98OAS INDEX",
    "EMBI":       "JPEIGLSP INDEX",
    "CDS_BOFA":   "BOFA CDS USD SR 5Y D14 CORP",
    "CDS_CITI":   "CITIB CDS USD SR 5Y D14 CORP",
    "CDS_JPM":    "JPMCC CDS USD SR 5Y D14 CORP",
    "CDS_GS":     "GS CDS USD SR 5Y D14 CORP",
    "CDS_UBS":    "UBS AG CDS EUR SR 5Y D14 CORP",
    "CDS_DB_SR":  "DB CDS EUR SR 5Y D14 CORP",
    "CDS_DB_SUB": "DB CDS EUR SUB 5Y D14 CORP",
    # --- Nominal yields (regime classification) ---------------------------
    "US_2Y":  "USGG2YR INDEX",  "US_5Y":  "USGG5YR INDEX",
    "US_10Y": "USGG10YR INDEX", "US_30Y": "USGG30YR INDEX",
    "DE_2Y":  "GDBR2 INDEX",    "DE_5Y":  "GDBR5 INDEX",
    "DE_10Y": "GDBR10 INDEX",   "DE_30Y": "GDBR30 INDEX",
    "JP_2Y":  "GJGB2 INDEX",    "JP_5Y":  "GJGB5 INDEX",
    "JP_10Y": "GJGB10 INDEX",   "JP_30Y": "GJGB30 INDEX",
    "UK_2Y":  "GUKG2 INDEX",    "UK_5Y":  "GUKG5 INDEX",
    "UK_10Y": "GUKG10 INDEX",   "UK_30Y": "GUKG30 INDEX",
    "CA_2Y":  "GCAN2YR INDEX",  "CA_5Y":  "GCAN5YR INDEX",
    "CA_10Y": "GCAN10YR INDEX", "CA_30Y": "GCAN30YR INDEX",
    "AU_2Y":  "GACGB2 INDEX",   "AU_5Y":  "GACGB5 INDEX",
    "AU_10Y": "GACGB10 INDEX",  "AU_30Y": "GACGB30 INDEX",
    # --- Inflation expectations -------------------------------------------
    "BE_2Y":  "USGGBE02 INDEX", "BE_5Y":  "USGGBE05 INDEX",
    "BE_10Y": "USGGBE10 INDEX", "BE_20Y": "USGGBE20 INDEX",
    "BE_30Y": "USGGBE30 INDEX",
    "ZCIS_1Y":  "USSWIT1 CURNCY",  "ZCIS_2Y":  "USSWIT2 CURNCY",
    "ZCIS_3Y":  "USSWIT3 CURNCY",  "ZCIS_4Y":  "USSWIT4 CURNCY",
    "ZCIS_5Y":  "USSWIT5 CURNCY",  "ZCIS_7Y":  "USSWIT7 CURNCY",
    "ZCIS_10Y": "USSWIT10 CURNCY", "ZCIS_20Y": "USSWIT20 CURNCY",
    "ZCIS_30Y": "USSWIT30 CURNCY",
    "INFL_5Y5Y": "FWISUS55 INDEX",
    # --- Money-market additions / overnight composite ---------------------
    "TOMO_TCSO": "TOMOTCSO INDEX",
    "USRG_1T":   "USRG1T CURNCY",
    # --- Central bank / reserve liquidity ---------------------------------
    "FED_RESERVES": "FARBRBFB INDEX",   # reserve balances ($mn)
    "FED_REPO":     "FARWCBLS INDEX",   # Fed repo / SRF-style usage (Wed level)
    # --- Financial-conditions benchmarks (NOT index components) -----------
    "FCI_BBG":  "BFCIUS INDEX",
    "FCI_NFCI": "NFCIINDX INDEX",
    # --- Credit indices (CDX / iTraxx) ------------------------------------
    "CDX_IG":       "IBOXUMAE CBBT CURNCY",
    "CDX_HY":       "IBOXHYAE CBIN CURNCY",
    "CDX_EM":       "IBOXUMSE CURNCY",
    "ITRX_EUROPE":  "ITRXEBE CBBT CURNCY",
    "ITRX_XOVER":   "ITRXEXE CBBT CURNCY",
    "ITRX_SR_FIN":  "ITRXESE CBBT CURNCY",
    "ITRX_SUB_FIN": "ITRXEUE CBBT CURNCY",
    "ITRX_JAPAN":   "ITRXAJE CBIN CURNCY",
    "ITRX_ASIA_XJ": "ITRXAGE CBBT CURNCY",
    "ITRX_AUS":     "ITRXAAE CBBT CURNCY",
    # --- Market liquidity / volatility ------------------------------------
    "UST_LIQ":  "GVLQUSD INDEX",   # Bloomberg US govt liquidity (higher = worse)
    "SWAP_10Y": "USSFCT10 CURNCY", # 10Y USD swap spread
    "MOVE":     "MOVE INDEX",      # NOT in current data — handled gracefully
    "VIX":      "VIX INDEX",       # NOT in current data — handled gracefully
    # --- Mortgage ----------------------------------------------------------
    "MTG_30Y": "APORF30Y INDEX",
}

# Countries with full 2/5/10/30 nominal coverage for regime classification
REGIME_COUNTRIES = ("US", "DE", "JP", "UK", "CA", "AU")

# Tenor configuration for real-rate / inflation curve plots
REAL_RATE_TENORS = {
    "US": [("5Y", 5, "US_real_5y"), ("10Y", 10, "US_real_10y"), ("30Y", 30, "US_real_30y")],
    "UK": [("5Y", 5, "UK_real_5y"), ("10Y", 10, "UK_real_10y"), ("30Y", 30, "UK_real_30y")],
    "DE": [("7Y", 7, "DE_real_7y"), ("10Y", 10, "DE_real_10y"), ("25Y", 25, "DE_real_25y")],
    "JP": [("5Y", 5, "JP_real_5y"), ("7Y", 7, "JP_real_7y"), ("10Y", 10, "JP_real_10y")],
    "AU": [("5Y", 5, "AU_real_5y"), ("10Y", 10, "AU_real_10y")],
    "CA": [("5Y", 5, "CA_real_5y"), ("10Y", 10, "CA_real_10y"), ("30Y", 30, "CA_real_30y")],
}

INFL_BE_TENORS = [
    ("2Y", 2, "BE_2Y"), ("5Y", 5, "BE_5Y"), ("10Y", 10, "BE_10Y"),
    ("20Y", 20, "BE_20Y"), ("30Y", 30, "BE_30Y"),
]
INFL_ZCIS_TENORS = [
    ("1Y", 1, "ZCIS_1Y"), ("2Y", 2, "ZCIS_2Y"), ("3Y", 3, "ZCIS_3Y"),
    ("4Y", 4, "ZCIS_4Y"), ("5Y", 5, "ZCIS_5Y"), ("7Y", 7, "ZCIS_7Y"),
    ("10Y", 10, "ZCIS_10Y"), ("20Y", 20, "ZCIS_20Y"), ("30Y", 30, "ZCIS_30Y"),
]

# Tenor pairs for slope/regime mode
TENOR_PAIRS = {
    "2s10s":  ("2Y", "10Y"),
    "2s30s":  ("2Y", "30Y"),
    "5s10s":  ("5Y", "10Y"),
    "5s30s":  ("5Y", "30Y"),
    "10s30s": ("10Y", "30Y"),
}

# Credit-index explorer options: label -> (key, unit)
CREDIT_INDICES = {
    "CDX NA IG (price)":         ("CDX_IG",       "price"),
    "CDX NA HY (price)":         ("CDX_HY",       "price"),
    "CDX EM (spread, bp)":       ("CDX_EM",       "spread"),
    "iTraxx Europe Main (bp)":   ("ITRX_EUROPE",  "spread"),
    "iTraxx Crossover (bp)":     ("ITRX_XOVER",   "spread"),
    "iTraxx Sr Financial (bp)":  ("ITRX_SR_FIN",  "spread"),
    "iTraxx Sub Financial (bp)": ("ITRX_SUB_FIN", "spread"),
    "iTraxx Japan (bp)":         ("ITRX_JAPAN",   "spread"),
    "iTraxx Asia ex-Japan (bp)": ("ITRX_ASIA_XJ", "spread"),
    "iTraxx Australia (bp)":     ("ITRX_AUS",     "spread"),
}
