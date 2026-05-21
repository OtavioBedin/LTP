import numpy as np
import pandas as pd
from collections import defaultdict, deque

# --------------------- ### Calcular Distribuição de Capacidade ### ---------------------
def calcular_distrib_capacidade(df, lote_min_flag, multiplo_emb_flag):
    df = df.copy()
    n = len(df)

    if n == 0:
        return (
            df,
            pd.DataFrame(columns=["ID_RECURSO", "HOR_REC", "NEC_ATEND_HR", "REC_HR_SALDO"]),
            pd.DataFrame(columns=["ID_FERRAMENTA", "HOR_FER", "NEC_ATEND_HR", "FER_HR_SALDO"]),
        )

    # ============================================================
    # Arrays base
    # ============================================================
    prior_matpar = pd.to_numeric(df["PRIOR_MATPAR"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    prior_rot = pd.to_numeric(df["PRIOR_ROT"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    id_recurso = df["ID_RECURSO"].fillna("").astype(str).to_numpy()
    id_ferramenta = df["ID_FERRAMENTA"].fillna("").astype(str).to_numpy()
    id_prod_unid_fat = df["ID_PROD_UNID_FAT"].fillna("").astype(str).to_numpy()

    nec_pcs = pd.to_numeric(df["NEC_PCS"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pcs_hora = pd.to_numeric(df["PCS_HORA"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    limit_pcs = pd.to_numeric(df["LIMIT_PCS"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lote_min = pd.to_numeric(df["LOTE_MIN"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    qtd_emb = pd.to_numeric(df["QTD_EMB"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    hor_rec = pd.to_numeric(df["HOR_REC"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hor_fer = pd.to_numeric(df["HOR_FER"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    tipo_prod = (
        df["TIPO_PROD"]
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    # ============================================================
    # Índices recurso/ferramenta sem groupby
    # np.unique preserva chaves e cria índice numérico rápido
    # ============================================================
    rec_keys, rec_first_idx, rec_idx = np.unique(
        id_recurso,
        return_index=True,
        return_inverse=True
    )

    fer_keys, fer_first_idx, fer_idx = np.unique(
        id_ferramenta,
        return_index=True,
        return_inverse=True
    )

    rec_hor_ini = hor_rec[rec_first_idx].astype(float, copy=True)
    fer_hor_ini = hor_fer[fer_first_idx].astype(float, copy=True)

    rec_saldo = rec_hor_ini.copy()
    fer_saldo = fer_hor_ini.copy()

    rec_nec_atend_total = np.zeros(len(rec_keys), dtype=float)
    fer_nec_atend_total = np.zeros(len(fer_keys), dtype=float)

    # ============================================================
    # Arrays resultado
    # ============================================================
    nec_hr = np.zeros(n, dtype=float)
    rec_cap = np.zeros(n, dtype=float)
    fer_cap = np.zeros(n, dtype=float)
    rot_cap = np.zeros(n, dtype=float)
    cap_prod = np.zeros(n, dtype=float)
    nec_atend_pcs = np.zeros(n, dtype=float)
    nec_atend_hr = np.zeros(n, dtype=float)
    nec_nao_pcs = np.zeros(n, dtype=float)
    nec_nao_hr = np.zeros(n, dtype=float)
    rec_hr_saldo_linha = np.zeros(n, dtype=float)
    fer_hr_saldo_linha = np.zeros(n, dtype=float)

    usa_lote = lote_min_flag != "NAO"
    usa_emb = multiplo_emb_flag != "NAO"
    mask_tipo_emb = (tipo_prod == "PA") | (tipo_prod == "MR")

    # ============================================================
    # Loop sequencial necessário pela regra de saldo acumulado
    # ============================================================
    nec_atend_acum_por_prod = defaultdict(float)
    
    for i in range(n):
        
        # - NEC_PCS da linha que questão, deduzindo a soma do NEC_ATEND_PCS das linhas anteriores com mesma chave de ID_PROD_UNID_FAT
        # Necessidade restante da linha atual,
        # descontando o que já foi atendido anteriormente
        # para o mesmo ID_PROD_UNID_FAT.
        chave_prod = id_prod_unid_fat[i]

        nec_pcs_i = nec_pcs[i] - nec_atend_acum_por_prod[chave_prod]

        if nec_pcs_i < 0.0:
            nec_pcs_i = 0.0

        if nec_pcs_i > 0.0:
            v = nec_pcs_i if nec_pcs_i > limit_pcs[i] else limit_pcs[i]

            if usa_lote and lote_min[i] > 0.0 and v < lote_min[i]:
                v = lote_min[i]

            # Múltiplo embalagem sem warning de divisão
            if usa_emb and mask_tipo_emb[i] and qtd_emb[i] > 0.0:
                v = np.ceil(v / qtd_emb[i]) * qtd_emb[i]

            nec_pcs_i = v
        else:
            nec_pcs_i = 0.0

        nec_pcs[i] = nec_pcs_i

        ph = pcs_hora[i]

        if ph != 0.0:
            nh = nec_pcs_i / ph
        else:
            nh = 0.0

        r = rec_idx[i]
        f = fer_idx[i]

        rc = rec_saldo[r]
        fc = fer_saldo[f]

        if rc < fc:
            rot = rc
        else:
            rot = fc

        cap = rot * ph if ph != 0.0 else 0.0

        if cap < nec_pcs_i:
            nap = cap
        else:
            nap = nec_pcs_i

        if ph != 0.0:
            nah = nap / ph
        else:
            nah = 0.0

        rec_saldo[r] = rc - nah if rc > nah else 0.0
        fer_saldo[f] = fc - nah if fc > nah else 0.0
        
        rec_hr_saldo_linha[i] = rec_saldo[r]
        fer_hr_saldo_linha[i] = fer_saldo[f]

        rec_nec_atend_total[r] += nah
        fer_nec_atend_total[f] += nah
        nec_atend_acum_por_prod[chave_prod] += nap

        nna_pcs = nec_pcs_i - nap

        if ph != 0.0:
            nna_hr = nna_pcs / ph
        else:
            nna_hr = 0.0

        nec_hr[i] = nh
        rec_cap[i] = rc
        fer_cap[i] = fc
        rot_cap[i] = rot
        cap_prod[i] = cap
        nec_atend_pcs[i] = nap
        nec_atend_hr[i] = nah
        nec_nao_pcs[i] = nna_pcs
        nec_nao_hr[i] = nna_hr

    # ============================================================
    # Assign final
    # ============================================================
    df["NEC_PCS"] = nec_pcs
    df["NEC_HR"] = nec_hr
    df["REC_CAP_VAR_HR"] = rec_cap
    df["FER_CAP_VAR_HR"] = fer_cap
    df["ROT_CAP_VAR_HR"] = rot_cap
    df["CAP_PROD_PCS"] = cap_prod
    df["NEC_ATEND_PCS"] = nec_atend_pcs
    df["NEC_ATEND_HR"] = nec_atend_hr
    df["NEC_NAO_ATEND_PCS"] = nec_nao_pcs
    df["NEC_NAO_ATEND_HR"] = nec_nao_hr
    df["REC_HR_SALDO"] = rec_hr_saldo_linha
    df["FER_HR_SALDO"] = fer_hr_saldo_linha

    # ============================================================
    # Outputs auxiliares
    # ============================================================
    df_dict_hor_rec = pd.DataFrame({
        "ID_RECURSO": rec_keys,
        "HOR_REC": rec_hor_ini,
        "NEC_ATEND_HR": rec_nec_atend_total,
        "REC_HR_SALDO": rec_saldo,
    })

    df_dict_hor_fer = pd.DataFrame({
        "ID_FERRAMENTA": fer_keys,
        "HOR_FER": fer_hor_ini,
        "NEC_ATEND_HR": fer_nec_atend_total,
        "FER_HR_SALDO": fer_saldo,
    })

    return df, df_dict_hor_rec, df_dict_hor_fer
