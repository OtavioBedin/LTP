
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

    usa_lote = lote_min_flag != "NAO"
    usa_emb = multiplo_emb_flag != "NAO"
    mask_tipo_emb = (tipo_prod == "PA") | (tipo_prod == "MR")

    # ============================================================
    # Loop sequencial necessário pela regra de saldo acumulado
    # ============================================================
    for i in range(n):

        # Herança de NEC não atendida da linha anterior
        if i > 0 and (prior_matpar[i] != 1 or prior_rot[i] != 1):
            nec_pcs_i = nec_nao_pcs[i - 1]
        else:
            nec_pcs_i = nec_pcs[i]

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

        rec_nec_atend_total[r] += nah
        fer_nec_atend_total[f] += nah

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
    df["REC_HR_SALDO"] = rec_saldo[rec_idx]
    df["FER_HR_SALDO"] = fer_saldo[fer_idx]

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

# --------------------- ### Explodir Estruturas de Produção ### ---------------------
def explodir_estrutura_ltp(bd_estrut, bd_ltp):
    """
    Explode a estrutura apenas 1 vez por combinação única de (COD_PROD, UNID_PROD).
    """

    cols_saida = [
        "COD_PROD",
        "UNID_PROD",
        "COD_PROD_ACAB",
        "COD_INSUMO",
        "QTD_UTIL_PCS",
        "NIVEL",
        "TRILHA",
    ]

    if bd_estrut.empty or bd_ltp.empty:
        return pd.DataFrame(columns=cols_saida)

    # Apenas as colunas necessárias
    estrut = bd_estrut[
        ["empresa", "cod_prod_acabado", "cod_insumo", "qtd_utilizada_pcs"]
    ]

    ltp = bd_ltp[["COD_PROD", "UNID_PROD"]]

    # Conversões mínimas, mantendo a regra original
    cod_pai = estrut["cod_prod_acabado"].astype(str).to_numpy()
    cod_filho = estrut["cod_insumo"].astype(str).to_numpy()
    empresa_arr = estrut["empresa"].to_numpy()
    qtd_arr = estrut["qtd_utilizada_pcs"].to_numpy()

    # Índice rápido:
    # estrutura_idx[empresa][pai] = [(filho, qtd), ...]
    estrutura_idx = defaultdict(lambda: defaultdict(list))

    for emp, pai, filho, qtd in zip(empresa_arr, cod_pai, cod_filho, qtd_arr):
        estrutura_idx[emp][pai].append((filho, qtd))

    # Combinações únicas, preservando ordem de aparição
    pares_ltp = (
        ltp.assign(COD_PROD=ltp["COD_PROD"].astype(str))
           .drop_duplicates(["COD_PROD", "UNID_PROD"], ignore_index=True)
    )

    resultados = []

    for prod_root, empresa in pares_ltp.itertuples(index=False, name=None):

        estrutura_empresa = estrutura_idx.get(empresa)
        if not estrutura_empresa:
            continue

        if prod_root not in estrutura_empresa:
            continue

        fila = deque()
        fila.append((prod_root, 0, prod_root))

        # Mantém a mesma lógica da função original:
        # evita reprocessar a mesma aresta pai -> filho
        visitados = set()

        while fila:
            pai, nivel, trilha = fila.popleft()

            filhos = estrutura_empresa.get(pai)
            if not filhos:
                continue

            nivel_filho = nivel + 1

            for filho, qtd in filhos:
                trilha_filho = f"{trilha} → {filho}"

                resultados.append((
                    prod_root,
                    empresa,
                    pai,
                    filho,
                    qtd,
                    nivel_filho,
                    trilha_filho,
                ))

                chave = (pai, filho)

                if chave not in visitados:
                    visitados.add(chave)
                    fila.append((filho, nivel_filho, trilha_filho))

    return pd.DataFrame.from_records(resultados, columns=cols_saida)
    
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
# ------------ ### Calcula fator estrutural para conversão de NEC_ATEND_PCS entre os niveis ### ---------------------
def criar_estrutura_com_fator_estrutural(df):
    """
    Reconstrói a estrutura multinível a partir da tabela achatada,
    calculando nível e fator estrutural acumulado.
    """

    df = df.copy()
    df['cod_prod_acabado'] = df['cod_prod_acabado'].astype(str)
    df['cod_insumo'] = df['cod_insumo'].astype(str)

    # Mapa: (empresa, pai) -> [(filho, qtd)]
    estrutura = (
        df.groupby(['empresa', 'cod_prod_acabado'])
          .apply(lambda x: list(zip(x['cod_insumo'], x['qtd_utilizada_pcs'])))
          .to_dict()
    )

    resultados = []

    def explodir(empresa, raiz, atual, nivel, fator):
        filhos = estrutura.get((empresa, atual))
        if not filhos:
            return

        for filho, qtd in filhos:
            fator_novo = fator * qtd
            resultados.append({
                'UNID_PROD': empresa,
                'COD_PROD': raiz,
                'COD_INSUMO': filho,
                'NIVEL': nivel,
                'FATOR_ESTRUTURAL': fator_novo
            })
            explodir(
                empresa,
                raiz,
                filho,
                nivel + 1,
                fator_novo
            )

    # Inicia a explosão para cada produto acabado
    for (empresa, cod_pa) in estrutura.keys():
        explodir(
            empresa=empresa,
            raiz=cod_pa,
            atual=cod_pa,
            nivel=1,
            fator=1
        )

    return pd.DataFrame(resultados)
# --------------------- ### Calcular e Criar tabela matriz de cortes para Recursos ### ---------------------
def cria_bd_mat_cortes_REC(bd_LTP):
    # Criar Matriz de Cortes e sequencia de priorização: Agregar as colunas ALOC_REC, IND%, somando a coluna OCUP_FER com base na bd_LTP, max da coluna HOR_REC, somar NEC_ESTOURO_HR_REC, somar NEC_ATEND_HR e formar tabela bd_mat_cortes
    bd_mat_cortes = bd_LTP.groupby(['UNID_PROD', 'ALOC_REC'], as_index=False).agg({
        'HOR_REC': 'max',
        'NEC_ESTOURO_HR_REC': 'sum',
        'NEC_ATEND_HR': 'sum',
        '%_OCUP_REC': 'sum',
    })
    
    bd_mat_cortes['CORTE_HR']  = (bd_mat_cortes['NEC_ESTOURO_HR_REC'] + bd_mat_cortes['NEC_ATEND_HR']) - bd_mat_cortes['HOR_REC']
    
    bd_mat_cortes = bd_mat_cortes.sort_values(by=['UNID_PROD', '%_OCUP_REC'], ascending=[True, False]).reset_index(drop=True)
    # bd_mat_cortes = bd_mat_cortes[bd_mat_cortes['%_OCUP_REC'] > 0].reset_index(drop=True)
    bd_mat_cortes['%_OCUP_REC'] = bd_mat_cortes['%_OCUP_REC'] * 100

    return bd_mat_cortes
# --------------------- ### Calcular e Criar tabela matriz de cortes para Ferramentas ### ---------------------
def cria_bd_mat_cortes_FER(bd_LTP):
    # Criar Matriz de Cortes e sequência de priorização para FERRAMENTA
    # Agregar as colunas COD_FER_UNID, somando %_OCUP_FER,
    # max da coluna HOR_FER, somar NEC_ESTOURO_HR_FER, somar NEC_ATEND_HR

    bd_mat_cortes = bd_LTP.groupby(['UNID_PROD', 'COD_FER_UNID'], as_index=False).agg({
        'HOR_FER': 'max',
        'NEC_ESTOURO_HR_FER': 'sum',
        'NEC_ATEND_HR': 'sum',
        '%_OCUP_FER': 'sum',
    })
    
    bd_mat_cortes['CORTE_HR'] = (
        bd_mat_cortes['NEC_ESTOURO_HR_FER'] +
        bd_mat_cortes['NEC_ATEND_HR']
    ) - bd_mat_cortes['HOR_FER']
    
    bd_mat_cortes = (
        bd_mat_cortes
        .sort_values(by=['UNID_PROD', '%_OCUP_FER'], ascending=[True, False])
        .reset_index(drop=True)
    )
    
    bd_mat_cortes['%_OCUP_FER'] = bd_mat_cortes['%_OCUP_FER'] * 100

    return bd_mat_cortes
# --------------------- ### Criar Matriz de Horas para definir aonde aplicar os cortes ### ---------------------
def matriz_logica_cortes_horas(bd_LTP, col_chave):
    
    df = bd_LTP[
        ['UNID_PROD', col_chave, 'MESMA_REG',
         'ES_HR', 'PV_PROX_HR', 'PV_HR', 'C_AT_HR', 'C_ARR_HR']
    ].copy()

    metric_cols = ['ES_HR', 'PV_PROX_HR', 'PV_HR', 'C_AT_HR', 'C_ARR_HR']

    # Long
    long = df.melt(
        id_vars=['UNID_PROD', col_chave, 'MESMA_REG'],
        value_vars=metric_cols,
        var_name='METRICA',
        value_name='VALOR'
    )

    # Agregação
    agg = (
        long.groupby(['UNID_PROD', col_chave, 'MESMA_REG', 'METRICA'], as_index=False)['VALOR']
            .sum()
    )

    # Pivot
    wide = (
        agg.pivot(
            index=['UNID_PROD', col_chave],
            columns=['MESMA_REG', 'METRICA'],
            values='VALOR'
        )
        .fillna(0)
    )

    # Flatten
    wide.columns = [f"{reg}|{met}" for reg, met in wide.columns]
    wide = wide.reset_index()

    # ORDEM FIXA USANDO NOMES ORIGINAIS
    ordered_cols = [
        'SIM|ES_HR',
        'NAO|PV_PROX_HR',
        'NAO|ES_HR',
        'SIM|PV_HR',
        'NAO|PV_HR',
        'SIM|C_AT_HR',
        'NAO|C_AT_HR',
        'SIM|C_ARR_HR',
        'NAO|C_ARR_HR'
    ]

    # Garante existência das colunas
    for col in ordered_cols:
        if col not in wide.columns:
            wide[col] = 0

    wide = wide[['UNID_PROD', col_chave] + ordered_cols]

    return wide
# *************************# Retorno da NEC_COMP_PCS para a estrutura LTP #********************************
def atualizar_ltp_comp_nec_pcs(bd_LTP, bd_nec_comp_expl):

        # Agregar campos UNID_PROD, COD_INSUMO, NEC_LIQ_PCS
        bd_nec_comp_expl_agreg = (
            bd_nec_comp_expl
            .groupby(["UNID_PROD", "COD_INSUMO"], as_index=False)["NEC_LIQ_PCS"]
            .sum()
        )
        
        # Faz o merge entre bd_LTP e bd_nec_comp_expl_agreg
        bd_LTP = bd_LTP.merge(
            bd_nec_comp_expl_agreg,
            left_on=["UNID_FAT", "COD_PROD"],
            right_on=["UNID_PROD", "COD_INSUMO"],
            how="left",
            suffixes=("", "_eliminar")
        )

        # Atualiza o campo LTP_COMP_NEC_PCS com NEC_LIQ_PCS quando existir
        bd_LTP["LTP_COMP_NEC_PCS"] = bd_LTP["NEC_LIQ_PCS"].fillna(bd_LTP["LTP_COMP_NEC_PCS"])

        # Remove coluna auxiliar COD_INSUMO se não precisar mais
        bd_LTP = bd_LTP.drop(columns=["COD_INSUMO", "NEC_LIQ_PCS", "UNID_PROD_eliminar"])

        return bd_LTP
# *************************# Decomposicao NEC_PCS para precisao de corte #********************************
def calcular_decomposicao_nec_pcs_para_precisao_corte(bd_LTP):

    # Remover colunas calculadas anteriormente, caso já existam, para evitar resíduos de execuções passadas
    cols_calculadas = [
        'ET_PCS',
        'C_ARR_PCS',
        'C_AT_PCS',
        'PV_PCS',
        'PV_PROX_PCS',
        'ES_PCS',
        'DIF_LM_PCS',
        'DIF_EMB_PCS',
        'C_ARR_HR',
        'C_AT_HR',
        'PV_HR',
        'PV_PROX_HR',
        'ES_HR',
        'DIF_LM_HR',
        'DIF_EMB_HR'
    ]

    cols_existentes = [col for col in cols_calculadas if col in bd_LTP.columns]
    if cols_existentes:
        bd_LTP = bd_LTP.drop(columns=cols_existentes)

    # Coluna Total Estoque para otimizar e reduzir tamanho dos próximos calculos que debitam estoque
    ET_PCS = (
        bd_LTP['LTP_EST_INI_PCS'] +
        bd_LTP['LTP_EST_TRANS_PCS'] +
        bd_LTP['TRIANG_TOT_PCS'] +
        bd_LTP['ORI_TOT_PCS']
    )

    bd_LTP['ET_PCS'] = np.where(
        bd_LTP['NEC_PCS'] == 0,
        0,
        ET_PCS
    )

    # Calculos identificando quantidades PCS não cobertas por estoque e que devem ser cortadas
    C_ARR_PCS = (bd_LTP['LTP_CART_ARR_MES_ANT'] - bd_LTP['ET_PCS']).clip(lower=0)

    bd_LTP['C_ARR_PCS'] = np.where(
        bd_LTP['NEC_PCS'] == 0,
        0,
        C_ARR_PCS
    )

    C_AT_PCS = (
        (bd_LTP['LTP_CART_ARR_MES_ANT'] + bd_LTP['LTP_CART_MES_ATUAL'])
        - bd_LTP['ET_PCS']
    ).clip(lower=0) - bd_LTP['C_ARR_PCS']

    bd_LTP['C_AT_PCS'] = np.where(
        bd_LTP['NEC_PCS'] == 0,
        0,
        C_AT_PCS
    )

    PV_PCS = (
        (bd_LTP['LTP_CART_ARR_MES_ANT'] +
         bd_LTP['LTP_CART_MES_ATUAL'] +
         bd_LTP['LTP_SALDO_PREV_PCS'])
        - bd_LTP['ET_PCS']
    ).clip(lower=0) - (bd_LTP['C_ARR_PCS'] + bd_LTP['C_AT_PCS'])

    bd_LTP['PV_PCS'] = np.where(
        bd_LTP['NEC_PCS'] == 0,
        0,
        PV_PCS
    )

    PV_PROX_PCS = np.where(
        bd_LTP['MESMA_REG'] == 'NAO',
        (
            (bd_LTP['LTP_CART_ARR_MES_ANT'] +
             bd_LTP['LTP_CART_MES_ATUAL'] +
             bd_LTP['LTP_SALDO_PREV_PCS'] +
             bd_LTP['LTP_SALDO_PREV_PROX_MES_PCS'])
            - bd_LTP['ET_PCS']
        ).clip(lower=0)
        - (bd_LTP['C_ARR_PCS'] +
           bd_LTP['C_AT_PCS'] +
           bd_LTP['PV_PCS']),
        0
    )

    bd_LTP['PV_PROX_PCS'] = np.where(
        bd_LTP['NEC_PCS'] == 0,
        0,
        PV_PROX_PCS
    )

    bd_LTP['ES_PCS'] = np.where(
        bd_LTP['NEC_PCS'] == 0,
        0,
        np.where(
            bd_LTP['MESMA_REG'] == 'NAO',
            np.maximum(
                (bd_LTP['LTP_CART_ARR_MES_ANT'] +
                 bd_LTP['LTP_CART_MES_ATUAL'] +
                 bd_LTP['LTP_SALDO_PREV_PCS'] +
                 bd_LTP['LTP_SALDO_PREV_PROX_MES_PCS'] +
                 bd_LTP['LTP_EST_SEG_PCS'])
                - bd_LTP['ET_PCS']
                - (bd_LTP['C_ARR_PCS'] +
                   bd_LTP['C_AT_PCS'] +
                   bd_LTP['PV_PCS'] +
                   bd_LTP['PV_PROX_PCS']),
                0
            ),
            np.maximum(
                (bd_LTP['LTP_CART_ARR_MES_ANT'] +
                 bd_LTP['LTP_CART_MES_ATUAL'] +
                 bd_LTP['LTP_SALDO_PREV_PCS'] +
                 bd_LTP['LTP_EST_SEG_PCS'])
                - bd_LTP['ET_PCS']
                - (bd_LTP['C_ARR_PCS'] +
                   bd_LTP['C_AT_PCS'] +
                   bd_LTP['PV_PCS'] +
                   bd_LTP['PV_PROX_PCS']),
                0
            )
        )
    )

    DIF_LM_PCS = (
        bd_LTP['NEC_PCS']
        - (bd_LTP['C_ARR_PCS'] +
           bd_LTP['C_AT_PCS'] +
           bd_LTP['PV_PCS'] +
           bd_LTP['PV_PROX_PCS'] +
           bd_LTP['ES_PCS'])
    )

    bd_LTP['DIF_LM_PCS'] = np.where(
        bd_LTP['NEC_PCS'] <= 0,
        0,
        np.where(
            DIF_LM_PCS > bd_LTP['LOTE_MIN'],
            bd_LTP['LOTE_MIN'],
            DIF_LM_PCS
        )
    )

    DIF_EMB_PCS = (
        bd_LTP['NEC_PCS']
        - (bd_LTP['C_ARR_PCS'] +
           bd_LTP['C_AT_PCS'] +
           bd_LTP['PV_PCS'] +
           bd_LTP['PV_PROX_PCS'] +
           bd_LTP['ES_PCS'] +
           bd_LTP['DIF_LM_PCS'])
    )

    bd_LTP['DIF_EMB_PCS'] = np.where(
        bd_LTP['NEC_PCS'] <= 0,
        0,
        DIF_EMB_PCS
    )

    # Transformar em HR as colunas calculadas em PCS para cortes
    pcs_hora_seguro = bd_LTP['PCS_HORA'].replace(0, np.nan)

    bd_LTP['C_ARR_HR'] = (bd_LTP['C_ARR_PCS'] / pcs_hora_seguro).fillna(0)
    bd_LTP['C_AT_HR'] = (bd_LTP['C_AT_PCS'] / pcs_hora_seguro).fillna(0)
    bd_LTP['PV_HR'] = (bd_LTP['PV_PCS'] / pcs_hora_seguro).fillna(0)
    bd_LTP['PV_PROX_HR'] = (bd_LTP['PV_PROX_PCS'] / pcs_hora_seguro).fillna(0)
    bd_LTP['ES_HR'] = (bd_LTP['ES_PCS'] / pcs_hora_seguro).fillna(0)
    bd_LTP['DIF_LM_HR'] = (bd_LTP['DIF_LM_PCS'] / pcs_hora_seguro).fillna(0)
    bd_LTP['DIF_EMB_HR'] = (bd_LTP['DIF_EMB_PCS'] / pcs_hora_seguro).fillna(0)

    return bd_LTP