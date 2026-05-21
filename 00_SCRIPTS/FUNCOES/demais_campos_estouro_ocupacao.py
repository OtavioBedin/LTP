import numpy as np
import pandas as pd
from collections import defaultdict, deque


# --------------------------------- ### Calculando demais campos ### --------------------------------
# Calcular os campos NEC_ESTOURO_PCS, NEC_ARRASTE_PCS, %_OCUP_REC, %_OCUP_FER
def calcular_demais_campos(df, inds_cortados=None):
    """
    Calcula campos complementares do motor de cortes.

    Nova regra de NEC_ESTOURO:
    - O estouro não é mais alocado em uma única linha via ID_PRI_PRIORI.
    - O estouro passa a ser rateado dentro do ID_PROD_UNID_FAT.
    - Para cada ID_PROD_UNID_FAT:
        1. remove os INDs já cortados;
        2. soma o campo base de não atendimento;
        3. calcula o percentual de participação de cada linha;
        4. pega o último valor válido do grupo;
        5. rateia esse último valor pelos percentuais;
        6. grava o resultado no campo de estouro.

    Campos calculados por rateio:
    - NEC_ESTOURO_PCS      baseado em NEC_NAO_ATEND_PCS
    - NEC_ESTOURO_PCS_REC  baseado em NEC_N_ATEND_PCS_REC
    - NEC_ESTOURO_PCS_FER  baseado em NEC_N_ATEND_PCS_FER
    """

    df = df.copy()
    n = len(df)

    if inds_cortados is None:
        inds_cortados = set()
    else:
        inds_cortados = set(inds_cortados)

    # ============================================================
    # Arrays base
    # ============================================================
    prior_matpar = df["PRIOR_MATPAR"].to_numpy()
    prior_rot = df["PRIOR_ROT"].to_numpy()

    id_prod = df["ID_PROD_UNID_FAT"]
    id_ant = df["ID_PROD_UNID_FAT_ANT"]

    nec_nao_atend_pcs = pd.to_numeric(
        df["NEC_NAO_ATEND_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    pcs_hora = pd.to_numeric(
        df["PCS_HORA"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    est_seg_pcs = pd.to_numeric(
        df["EST_SEG_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    nec_pcs = pd.to_numeric(
        df["NEC_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    rec_cap_var_hr = pd.to_numeric(
        df["REC_CAP_VAR_HR"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    fer_cap_var_hr = pd.to_numeric(
        df["FER_CAP_VAR_HR"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    nec_atend_pcs = pd.to_numeric(
        df["NEC_ATEND_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    hor_rec = pd.to_numeric(
        df["HOR_REC"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    hor_fer = pd.to_numeric(
        df["HOR_FER"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # ============================================================
    # Máscara de linhas elegíveis para rateio
    # ============================================================
    # Regra:
    # - Linha sem IND não é considerada cortada.
    # - Linha cujo IND está em inds_cortados não participa do rateio.
    # - Linha com valor base <= 0 também não participa daquele rateio específico.
    # ============================================================
    if "IND" in df.columns and inds_cortados:
        mask_ind_disponivel = ~df["IND"].isin(inds_cortados).to_numpy()
    else:
        mask_ind_disponivel = np.ones(n, dtype=bool)

    # ============================================================
    # Função interna: rateia o último valor válido por ID_PROD_UNID_FAT
    # ============================================================
    def _ratear_ultimo_valor_por_id(valores_base):
        """
        Rateia o último valor válido de cada ID_PROD_UNID_FAT entre as linhas
        disponíveis do mesmo ID_PROD_UNID_FAT.

        Parâmetros:
        - valores_base: array com o campo que será usado como base do rateio.
          Exemplos:
            NEC_NAO_ATEND_PCS
            NEC_N_ATEND_PCS_REC
            NEC_N_ATEND_PCS_FER

        Regra:
        - considera apenas linhas ainda não cortadas;
        - considera apenas valores_base > 0;
        - soma valores_base por ID_PROD_UNID_FAT;
        - calcula percentual da linha no grupo;
        - pega o último valores_base válido do grupo;
        - multiplica último valor pelo percentual da linha.
        """

        valores_base = np.asarray(valores_base, dtype=float)

        # Linha participa somente se:
        # - ainda não foi cortada;
        # - tem valor base positivo;
        # - tem ID_PROD_UNID_FAT preenchido.
        mask_participa = (
            mask_ind_disponivel
            & (valores_base > 0)
            & id_prod.notna().to_numpy()
        )

        if not mask_participa.any():
            return np.zeros(n, dtype=float)

        serie_valor = pd.Series(valores_base, index=df.index)
        serie_id = id_prod.astype(str)

        # Soma do campo base por ID_PROD_UNID_FAT,
        # considerando somente linhas que participam do rateio.
        soma_por_id = (
            serie_valor
            .where(mask_participa, 0.0)
            .groupby(serie_id, sort=False)
            .transform("sum")
            .to_numpy(dtype=float)
        )

        # Percentual de participação de cada linha no grupo.
        percentual = np.divide(
            valores_base,
            soma_por_id,
            out=np.zeros(n, dtype=float),
            where=(soma_por_id != 0) & mask_participa
        )

        # Último valor válido por ID_PROD_UNID_FAT.
        # Importante:
        # - usa a ordem atual do DataFrame;
        # - considera somente linhas que participam;
        # - se um IND já foi cortado, ele não pode ser o último valor válido.
        ultimo_valor_por_id = (
            pd.DataFrame({
                "ID_PROD_UNID_FAT": serie_id,
                "VALOR_BASE": valores_base,
                "PARTICIPA": mask_participa,
            }, index=df.index)
            .loc[lambda x: x["PARTICIPA"]]
            .groupby("ID_PROD_UNID_FAT", sort=False)["VALOR_BASE"]
            .last()
        )

        ultimo_valor_linha = (
            serie_id
            .map(ultimo_valor_por_id)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # Rateio final:
        # último valor válido do grupo * percentual da linha.
        resultado = ultimo_valor_linha * percentual

        # Garante que linhas não participantes fiquem zeradas.
        resultado = np.where(mask_participa, resultado, 0.0)

        return resultado

    # ============================================================
    # NEC_ESTOURO_PCS
    # Nova regra:
    # - rateia o último NEC_NAO_ATEND_PCS do ID_PROD_UNID_FAT
    #   entre as linhas ainda não cortadas do mesmo ID_PROD_UNID_FAT.
    # ============================================================
    nec_estouro_pcs = _ratear_ultimo_valor_por_id(nec_nao_atend_pcs)

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
    # Regra mantida:
    # NEC_ARRASTE_PCS = max(NEC_ESTOURO_PCS - EST_SEG_PCS, 0)
    # Depois busca pelo ID_PROD_UNID_FAT_ANT nas linhas PRIOR_MATPAR = 1 e PRIOR_ROT = 1.
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
    # Nova regra:
    # - não usa mais ID_PRI_PRIORI;
    # - calcula por rateio dentro do ID_PROD_UNID_FAT;
    # - ignora INDs já cortados.
    # ============================================================
    nec_estouro_pcs_rec = _ratear_ultimo_valor_por_id(nec_n_atend_pcs_rec)
    nec_estouro_pcs_fer = _ratear_ultimo_valor_por_id(nec_n_atend_pcs_fer)

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
    # Regra mantida:
    # ((NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA) / HOR_REC
    # Se HOR_REC == 0 ou PCS_HORA == 0, retorna 0.
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
    # Regra mantida:
    # ((NEC_ESTOURO_PCS_FER + NEC_ATEND_PCS) / PCS_HORA) / HOR_FER
    # Se HOR_FER == 0 ou PCS_HORA == 0, retorna 0.
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