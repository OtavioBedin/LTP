import numpy as np
import pandas as pd
from collections import defaultdict, deque

# --------------------- ### Calcular Estoque Deduzindo Demanda Bruta ### ---------------------
# Função para calcular estoque final, deduzindo demanda bruta para consumo na função de explosão de estrutura
def calc_estoque_deduzindo_demanda_bruta(df, lote_min_flag, multiplo_emb_flag):
    df = df.copy()

    # Coluna de estoque total
    df["ESTOQUE_TOTAL_PCS"] = (
        df["LTP_EST_INI_PCS"].fillna(0) +
        df["LTP_EST_TRANS_PCS"].fillna(0) +
        df["ORI_TOT_PCS"].fillna(0) +
        df["TRIANG_TOT_PCS"].fillna(0)
    )

    # Arrays
    mesma_reg = df['MESMA_REG'].values
    tipo_prod = df['TIPO_PROD'].astype(str).str.upper().values
    ltp_cart_ant = df['LTP_CART_ARR_MES_ANT'].fillna(0).values
    ltp_cart_atual = df['LTP_CART_MES_ATUAL'].fillna(0).values
    ltp_saldo_prev = df['LTP_SALDO_PREV_PCS'].fillna(0).values
    ltp_saldo_prox = df['LTP_SALDO_PREV_PROX_MES_PCS'].fillna(0).values
    ltp_est_seg = df['LTP_EST_SEG_PCS'].fillna(0).values
    limit_pcs = df['LIMIT_PCS'].astype(float).fillna(0.0).values
    lote_min = df['LOTE_MIN'].fillna(0).values
    qtd_emb = df['QTD_EMB'].fillna(0).values

    # Demanda base: duas fórmulas diferentes
    demanda_base = (
        ltp_cart_ant + ltp_cart_atual + ltp_saldo_prev + ltp_est_seg
    )
    mask_mr = (mesma_reg == 'NAO') # & (tipo_prod == 'MR')
    demanda_base[mask_mr] = (
        ltp_cart_ant[mask_mr] + ltp_cart_atual[mask_mr] +
        ltp_saldo_prev[mask_mr] + ltp_saldo_prox[mask_mr] + ltp_est_seg[mask_mr]
    )

    # Aplicar limite
    var_nec1 = np.maximum(demanda_base, limit_pcs)

    # Aplicar lote mínimo
    if lote_min_flag != 'NAO':
        var_nec2 = np.where(var_nec1 == 0, 0, np.where(var_nec1 < lote_min, lote_min, var_nec1))
    else:
        var_nec2 = var_nec1

    # Aplicar múltiplo de embalagem
    cond_emb = (np.isin(tipo_prod, ['PA', 'MR'])) & (multiplo_emb_flag != 'NAO') & (qtd_emb != 0)
    var_nec3 = var_nec2.copy()
    var_nec3[cond_emb] = np.ceil(var_nec2[cond_emb] / qtd_emb[cond_emb]) * qtd_emb[cond_emb]

    # Demanda final (não negativa)
    df['DEMANDA_PCS'] = np.maximum(var_nec3, 0)

    # Estoque final = estoque total - demanda (não negativo)
    df['ESTOQUE_TOTAL_PCS'] = np.maximum(df['ESTOQUE_TOTAL_PCS'] - df['DEMANDA_PCS'], 0)

    # Agrupar por COD_PROD e UNID_FAT retornando maximo do estoque final
    df_estoque_final = df.groupby(['COD_PROD', 'UNID_FAT'], as_index=False)['ESTOQUE_TOTAL_PCS'].max()
    
    return df_estoque_final
