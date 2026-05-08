import numpy as np
import pandas as pd
from collections import defaultdict, deque

try:
    from .estoque_deduzido_demanda_bruta import calc_estoque_deduzindo_demanda_bruta
except ImportError:
    from estoque_deduzido_demanda_bruta import calc_estoque_deduzindo_demanda_bruta

# --------------------- ### Calcular Explosão de Necessidades ### ---------------------
def calcular_explosao_necessidades(
    bd_explodida,
    bd_ltp,
    lote_min_flag,
    multiplo_emb_flag
):
    # ----------------------------------------------------------------------
    # 1. ESTOQUE GLOBAL
    # ----------------------------------------------------------------------
    bd_estoque = calc_estoque_deduzindo_demanda_bruta(
        bd_ltp,
        lote_min_flag,
        multiplo_emb_flag
    )

    est_dict = {
        (str(u), str(p)): e
        for u, p, e in bd_estoque[
            ["UNID_FAT", "COD_PROD", "ESTOQUE_TOTAL_PCS"]
        ].itertuples(index=False, name=None)
    }

    # ----------------------------------------------------------------------
    # 2. NECESSIDADE BASE
    # ----------------------------------------------------------------------
    bd_necessidade = (
        bd_ltp
        .groupby(["COD_PROD", "UNID_PROD"], as_index=False)["NEC_ATEND_PCS"]
        .sum()
    )

    bd_necessidade["COD_PROD"] = bd_necessidade["COD_PROD"].astype(str)
    bd_necessidade["UNID_PROD"] = bd_necessidade["UNID_PROD"].astype(str)

    # ----------------------------------------------------------------------
    # 3. MERGE BASE
    # ----------------------------------------------------------------------
    df = (
        bd_explodida
        .merge(bd_necessidade, on=["COD_PROD", "UNID_PROD"], how="left")
        .sort_values(["COD_PROD", "UNID_PROD", "NIVEL"], kind="mergesort")
        .reset_index(drop=True)
    )

    if df.empty:
        return df

    # Cast mínimo (evitar múltiplos astype)
    df["COD_PROD"] = df["COD_PROD"].astype(str)
    df["UNID_PROD"] = df["UNID_PROD"].astype(str)
    df["COD_PROD_ACAB"] = df["COD_PROD_ACAB"].astype(str)
    df["COD_INSUMO"] = df["COD_INSUMO"].astype(str)

    # ----------------------------------------------------------------------
    # 4. CONVERSÃO PARA NUMPY (CRÍTICO PRA PERFORMANCE)
    # ----------------------------------------------------------------------
    cod_prod = df["COD_PROD"].to_numpy()
    unid_prod = df["UNID_PROD"].to_numpy()
    cod_pai = df["COD_PROD_ACAB"].to_numpy()
    cod_insumo = df["COD_INSUMO"].to_numpy()
    qtd_util = df["QTD_UTIL_PCS"].to_numpy()
    nivel = df["NIVEL"].to_numpy()
    trilha = df["TRILHA"].to_numpy()
    nec_atend = df["NEC_ATEND_PCS"].to_numpy()

    n = len(df)

    # Saída pré-alocada
    nec_comp = np.zeros(n)
    nec_liq = np.zeros(n)
    est_antes = np.zeros(n)
    est_depois = np.zeros(n)
    deve_explodir = np.zeros(n, dtype=bool)

    # ----------------------------------------------------------------------
    # 5. IDENTIFICA BLOCOS (SEM GROUPBY)
    # ----------------------------------------------------------------------
    chave = cod_prod + "|" + unid_prod
    mudanca = np.r_[True, chave[1:] != chave[:-1]]
    idx_inicio = np.flatnonzero(mudanca)
    idx_fim = np.r_[idx_inicio[1:], n]

    # NEC inicial lookup
    nec_lookup = {
        (c, u): v
        for c, u, v in bd_necessidade[
            ["COD_PROD", "UNID_PROD", "NEC_ATEND_PCS"]
        ].itertuples(index=False, name=None)
    }

    # ----------------------------------------------------------------------
    # 6. LOOP MRP OTIMIZADO
    # ----------------------------------------------------------------------
    for start, end in zip(idx_inicio, idx_fim):

        cod = cod_prod[start]
        emp = unid_prod[start]

        nec_dict = {(cod, emp): nec_lookup.get((cod, emp), 0)}

        for i in range(start, end):

            pai = cod_pai[i]
            ins = cod_insumo[i]

            nec_pai = nec_dict.get((pai, emp), 0.0)

            if nec_pai == 0:
                continue

            nc = nec_pai * qtd_util[i]

            chave_est = (emp, ins)
            est_a = est_dict.get(chave_est, 0.0)

            if est_a >= nc:
                est_d = est_a - nc
                nl = 0.0
                de = False
            else:
                nl = nc - est_a
                est_d = 0.0
                de = True

            est_dict[chave_est] = est_d

            if de:
                nec_dict[(ins, emp)] = nl

            # grava direto no numpy
            nec_comp[i] = nc
            nec_liq[i] = nl
            est_antes[i] = est_a
            est_depois[i] = est_d
            deve_explodir[i] = de

    # ----------------------------------------------------------------------
    # 7. OUTPUT FINAL (SEM CONCAT)
    # ----------------------------------------------------------------------
    df["NEC_COMP_PCS"] = nec_comp
    df["NEC_LIQ_PCS"] = nec_liq
    df["EST_PCS_ANTES"] = est_antes
    df["EST_PCS_DEPOIS"] = est_depois
    df["DEVE_EXPLODIR"] = deve_explodir

    return df
