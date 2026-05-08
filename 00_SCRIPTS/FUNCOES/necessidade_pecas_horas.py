import numpy as np
import pandas as pd
from collections import defaultdict, deque

# --------------------- ### Calcular Necessidade de Peças e Horas ### ---------------------
def calc_nec_pcs_hr(df, lote_min_flag, multiplo_emb_flag):
    df = df.copy()
    n = len(df)

    mesma_reg = df["MESMA_REG"].to_numpy()

    tipo_prod = (
        df["TIPO_PROD"]
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    ltp_cart_ant = pd.to_numeric(df["LTP_CART_ARR_MES_ANT"], errors="coerce").fillna(0.0).to_numpy()
    ltp_cart_atual = pd.to_numeric(df["LTP_CART_MES_ATUAL"], errors="coerce").fillna(0.0).to_numpy()
    ltp_saldo_prev = pd.to_numeric(df["LTP_SALDO_PREV_PCS"], errors="coerce").fillna(0.0).to_numpy()
    ltp_nec_comp = pd.to_numeric(df["LTP_COMP_NEC_PCS"], errors="coerce").fillna(0.0).to_numpy()
    ltp_saldo_prox = pd.to_numeric(df["LTP_SALDO_PREV_PROX_MES_PCS"], errors="coerce").fillna(0.0).to_numpy()
    ltp_est_seg = pd.to_numeric(df["LTP_EST_SEG_PCS"], errors="coerce").fillna(0.0).to_numpy()
    ltp_est_ini = pd.to_numeric(df["LTP_EST_INI_PCS"], errors="coerce").fillna(0.0).to_numpy()
    ltp_est_trans = pd.to_numeric(df["LTP_EST_TRANS_PCS"], errors="coerce").fillna(0.0).to_numpy()
    ori_tot = pd.to_numeric(df["ORI_TOT_PCS"], errors="coerce").fillna(0.0).to_numpy()
    triang_tot = pd.to_numeric(df["TRIANG_TOT_PCS"], errors="coerce").fillna(0.0).to_numpy()
    limit_pcs = pd.to_numeric(df["LIMIT_PCS"], errors="coerce").fillna(0.0).to_numpy()
    lote_min = pd.to_numeric(df["LOTE_MIN"], errors="coerce").fillna(0.0).to_numpy()
    qtd_emb = pd.to_numeric(df["QTD_EMB"], errors="coerce").fillna(0.0).to_numpy()
    pcs_hora = pd.to_numeric(df["PCS_HORA"], errors="coerce").fillna(0.0).to_numpy()

    # Base comum
    base_comum = (
        ltp_cart_ant
        + ltp_cart_atual
        + ltp_saldo_prev
        + ltp_est_seg
        - (
            ltp_est_ini
            + ltp_est_trans
            + ori_tot
            + triang_tot
        )
    )

    # Se MESMA_REG == NAO, soma saldo próximo mês
    nec_pcs = np.where(
        mesma_reg == "NAO",
        base_comum + ltp_saldo_prox,
        base_comum
    )

    # Zera negativos
    nec_pcs = np.maximum(nec_pcs, 0.0)

    # Soma necessidade de componentes
    nec_pcs = nec_pcs + ltp_nec_comp

    mask_pos = nec_pcs > 0
    nec_final = np.zeros(n, dtype=float)

    if mask_pos.any():
        v = np.maximum(nec_pcs, limit_pcs)

        # Lote mínimo
        if lote_min_flag == "SIM":
            mask_lote = mask_pos & (lote_min > 0)
            v = np.where(mask_lote, np.maximum(v, lote_min), v)

        # Múltiplo de embalagem
        if multiplo_emb_flag == "SIM":
            mask_emb = (
                mask_pos
                & ((tipo_prod == "PA") | (tipo_prod == "MR"))
                & (qtd_emb > 0)
            )

            if mask_emb.any():
                v_emb = v.copy()
                v_emb[mask_emb] = (
                    np.ceil(v[mask_emb] / qtd_emb[mask_emb])
                    * qtd_emb[mask_emb]
                )
                v = v_emb

        nec_final = np.where(mask_pos, v, 0.0)

    nec_hr = np.divide(
        nec_final,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    df["NEC_PCS"] = nec_final
    df["NEC_HR"] = nec_hr

    return df
