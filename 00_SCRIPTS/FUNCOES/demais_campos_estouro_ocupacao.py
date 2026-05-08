import numpy as np
import pandas as pd
from collections import defaultdict, deque

# --------------------------------- ### Calculando demais campos ### --------------------------------
# Calcular os campos NEC_ESTOURO_PCS, NEC_ARRASTE_PCS, %_OCUP_REC, %_OCUP_FER
def calcular_demais_campos(df):
    df = df.copy()
    n = len(df)

    # ============================================================
    # Arrays base
    # ============================================================
    prior_matpar = df["PRIOR_MATPAR"].to_numpy()
    prior_rot = df["PRIOR_ROT"].to_numpy()

    id_ult = df["ID_ULT_PRIORI"]
    id_prod = df["ID_PROD_UNID_FAT"]
    id_ant = df["ID_PROD_UNID_FAT_ANT"]

    id_ult_notna = id_ult.notna().to_numpy()

    nec_nao_atend_pcs = pd.to_numeric(df["NEC_NAO_ATEND_PCS"], errors="coerce").fillna(0.0).to_numpy()
    pcs_hora = pd.to_numeric(df["PCS_HORA"], errors="coerce").fillna(0.0).to_numpy()
    est_seg_pcs = pd.to_numeric(df["EST_SEG_PCS"], errors="coerce").fillna(0.0).to_numpy()
    nec_pcs = pd.to_numeric(df["NEC_PCS"], errors="coerce").fillna(0.0).to_numpy()
    rec_cap_var_hr = pd.to_numeric(df["REC_CAP_VAR_HR"], errors="coerce").fillna(0.0).to_numpy()
    fer_cap_var_hr = pd.to_numeric(df["FER_CAP_VAR_HR"], errors="coerce").fillna(0.0).to_numpy()
    nec_atend_pcs = pd.to_numeric(df["NEC_ATEND_PCS"], errors="coerce").fillna(0.0).to_numpy()
    hor_rec = pd.to_numeric(df["HOR_REC"], errors="coerce").fillna(0.0).to_numpy()
    hor_fer = pd.to_numeric(df["HOR_FER"], errors="coerce").fillna(0.0).to_numpy()

    mask_prior_ult = (prior_matpar == 1) & id_ult_notna
    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # ============================================================
    # NEC_ESTOURO_PCS
    # Regra original:
    # tab_NEC_N_ATEND_PCS:
    # - ID_ULT_PRIORI notna
    # - NEC_NAO_ATEND_PCS > 0
    # Depois set_index(ID_PROD_UNID_FAT).to_dict()
    # Em duplicados, o último prevalece.
    # ============================================================
    mask_tab_estouro = (nec_nao_atend_pcs > 0) & id_ult_notna

    serie_estouro = pd.Series(
        nec_nao_atend_pcs[mask_tab_estouro],
        index=id_prod[mask_tab_estouro]
    )

    if not serie_estouro.empty:
        serie_estouro = serie_estouro[~serie_estouro.index.duplicated(keep="last")]

    nec_estouro_pcs = np.zeros(n, dtype=float)

    if mask_prior_ult.any() and not serie_estouro.empty:
        nec_estouro_pcs[mask_prior_ult] = (
            id_prod[mask_prior_ult]
            .map(serie_estouro)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    df["NEC_ESTOURO_PCS"] = nec_estouro_pcs

    # ============================================================
    # NEC_ESTOURO_HR
    # ============================================================
    df["NEC_ESTOURO_HR"] = np.divide(
        nec_estouro_pcs,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    # ============================================================
    # NEC_ARRASTE_PCS
    # Regra original:
    # NEC_ARRASTE_PCS = max(NEC_ESTOURO_PCS - EST_SEG_PCS, 0)
    # Depois set_index(ID_PROD_UNID_FAT).to_dict()
    # Busca usando ID_PROD_UNID_FAT_ANT.
    # Em duplicados, o último prevalece.
    # ============================================================
    nec_arraste_base = np.maximum(nec_estouro_pcs - est_seg_pcs, 0.0)

    serie_arraste = pd.Series(
        nec_arraste_base,
        index=id_prod
    )

    if not serie_arraste.empty:
        serie_arraste = serie_arraste[~serie_arraste.index.duplicated(keep="last")]

    nec_arraste_pcs = np.zeros(n, dtype=float)

    if mask_prior_rot.any() and not serie_arraste.empty:
        nec_arraste_pcs[mask_prior_rot] = (
            id_ant[mask_prior_rot]
            .map(serie_arraste)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    df["NEC_ARRASTE_PCS"] = nec_arraste_pcs

    # ============================================================
    # NEC_N_ATEND_PCS_REC / FER
    # ============================================================
    nec_n_atend_pcs_rec = np.maximum(
        0.0,
        nec_pcs - (rec_cap_var_hr * pcs_hora)
    )

    nec_n_atend_pcs_fer = np.maximum(
        0.0,
        nec_pcs - (fer_cap_var_hr * pcs_hora)
    )

    df["NEC_N_ATEND_PCS_REC"] = nec_n_atend_pcs_rec
    df["NEC_N_ATEND_PCS_FER"] = nec_n_atend_pcs_fer

    # ============================================================
    # NEC_ESTOURO_PCS_REC / FER
    # Regra original:
    # tab_NEC_N_ATEND_PCS_REC_FER:
    # - ID_ULT_PRIORI notna
    # - REC > 0 ou FER > 0
    # Depois:
    # set_index(ID_ULT_PRIORI).to_dict()
    # Busca usando ID_PROD_UNID_FAT.
    # Em duplicados, o último prevalece.
    # ============================================================
    mask_tab_rec_fer = (
        id_ult_notna
        & (
            (nec_n_atend_pcs_rec > 0)
            | (nec_n_atend_pcs_fer > 0)
        )
    )

    serie_rec = pd.Series(
        nec_n_atend_pcs_rec[mask_tab_rec_fer],
        index=id_ult[mask_tab_rec_fer]
    )

    serie_fer = pd.Series(
        nec_n_atend_pcs_fer[mask_tab_rec_fer],
        index=id_ult[mask_tab_rec_fer]
    )

    if not serie_rec.empty:
        serie_rec = serie_rec[~serie_rec.index.duplicated(keep="last")]

    if not serie_fer.empty:
        serie_fer = serie_fer[~serie_fer.index.duplicated(keep="last")]

    nec_estouro_pcs_rec = np.zeros(n, dtype=float)
    nec_estouro_pcs_fer = np.zeros(n, dtype=float)

    if mask_prior_ult.any() and not serie_rec.empty:
        nec_estouro_pcs_rec[mask_prior_ult] = (
            id_prod[mask_prior_ult]
            .map(serie_rec)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    if mask_prior_ult.any() and not serie_fer.empty:
        nec_estouro_pcs_fer[mask_prior_ult] = (
            id_prod[mask_prior_ult]
            .map(serie_fer)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    df["NEC_ESTOURO_PCS_REC"] = nec_estouro_pcs_rec
    df["NEC_ESTOURO_PCS_FER"] = nec_estouro_pcs_fer

    # ============================================================
    # NEC_ESTOURO_HR_REC / FER
    # ============================================================
    df["NEC_ESTOURO_HR_REC"] = np.divide(
        nec_estouro_pcs_rec,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    df["NEC_ESTOURO_HR_FER"] = np.divide(
        nec_estouro_pcs_fer,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    # ============================================================
    # %_OCUP_REC
    # Regra original:
    # ((NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA) / HOR_REC
    # Se HOR_REC == 0 ou PCS_HORA == 0, retorna 0
    # Se negativo, retorna 0
    # ============================================================
    hr_ocup_rec_base = np.divide(
        nec_estouro_pcs_rec + nec_atend_pcs,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    ocup_rec = np.divide(
        hr_ocup_rec_base,
        hor_rec,
        out=np.zeros(n, dtype=float),
        where=hor_rec != 0
    )

    ocup_rec = np.maximum(ocup_rec, 0.0)

    # ============================================================
    # %_OCUP_FER
    # ============================================================
    hr_ocup_fer_base = np.divide(
        nec_estouro_pcs_fer + nec_atend_pcs,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    ocup_fer = np.divide(
        hr_ocup_fer_base,
        hor_fer,
        out=np.zeros(n, dtype=float),
        where=hor_fer != 0
    )

    ocup_fer = np.maximum(ocup_fer, 0.0)

    df["%_OCUP_REC"] = ocup_rec
    df["%_OCUP_FER"] = ocup_fer

    # ============================================================
    # HR_OCUP_FER / REC
    # ============================================================
    df["HR_OCUP_FER"] = hor_fer * ocup_fer
    df["HR_OCUP_REC"] = hor_rec * ocup_rec

    return df
