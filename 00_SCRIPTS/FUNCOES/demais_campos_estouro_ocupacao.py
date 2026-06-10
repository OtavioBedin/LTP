import numpy as np
import pandas as pd


# --------------------------------- ### Calculando demais campos ### --------------------------------
# Esta função calcula campos complementares do motor de cortes:
#
# - NEC_ESTOURO_PCS
# - NEC_ESTOURO_HR
# - NEC_ARRASTE_PCS
# - NEC_N_ATEND_PCS_REC
# - NEC_N_ATEND_PCS_FER
# - NEC_ESTOURO_PCS_REC
# - NEC_ESTOURO_PCS_FER
# - NEC_ESTOURO_HR_REC
# - NEC_ESTOURO_HR_FER
# - %_OCUP_REC
# - %_OCUP_FER
# - HR_OCUP_REC
# - HR_OCUP_FER
#
# REGRA DE ESTOURO:
#
# Grupo de distribuição:
# ID_PROD_UNID_FAT
#
# TOTAL_ATEND:
# - Geral: NEC_ATEND_HR + NEC_NAO_ATEND_HR
# - REC:   NEC_ATEND_HR + NEC_N_ATEND_HR_REC
# - FER:   NEC_ATEND_HR + NEC_N_ATEND_HR_FER
#
# %_DIST_ESTOURO:
# TOTAL_ATEND da linha / soma TOTAL_ATEND do grupo ID_PROD_UNID_FAT
#
# Todas as linhas do grupo participam.
# Não filtra mais PRIOR_MATPAR = 1.
# Não usa mais PCS_HORA como peso.
# Não faz mais média entre pesos.
#
# NEC_ESTOURO_PCS:
# NEC_NAO_ATEND_PCS da última linha do grupo ID_PROD_UNID_FAT
# * %_DIST_ESTOURO
#
# NEC_ESTOURO_PCS_REC:
# NEC_N_ATEND_PCS_REC da última linha do grupo ID_PROD_UNID_FAT
# * %_DIST_ESTOURO_REC
#
# NEC_ESTOURO_PCS_FER:
# NEC_N_ATEND_PCS_FER da última linha do grupo ID_PROD_UNID_FAT
# * %_DIST_ESTOURO_FER


def calcular_demais_campos(df):
    df = df.copy()
    n = len(df)

    # =============================================================================================
    # Helpers

    def col_num(nome_coluna):
        return pd.to_numeric(
            df[nome_coluna],
            errors="coerce"
        ).fillna(0.0).to_numpy(dtype="float64", copy=False)

    def dividir_seguro(numerador, denominador):
        numerador = np.asarray(numerador, dtype="float64")
        denominador = np.asarray(denominador, dtype="float64")

        return np.divide(
            numerador,
            denominador,
            out=np.zeros_like(numerador, dtype="float64"),
            where=denominador != 0
        )

    def calcular_percentual_distribuicao(total_atend_base):
        """
        Calcula o percentual de distribuição do estouro.

        Regra:
        %_DIST_ESTOURO =
        TOTAL_ATEND da linha / soma TOTAL_ATEND do grupo ID_PROD_UNID_FAT

        Todas as linhas do grupo participam.
        """

        total_atend_serie = pd.Series(
            total_atend_base,
            index=df.index
        )

        soma_total_atend = (
            total_atend_serie
            .groupby(df["ID_PROD_UNID_FAT"], sort=False)
            .transform("sum")
            .to_numpy(dtype="float64", copy=False)
        )

        pct_dist = dividir_seguro(
            total_atend_base,
            soma_total_atend
        )

        return pct_dist

    def decompor_estouro(nome_coluna_base_pcs, total_atend_base):
        """
        Busca o valor da coluna base PCS da última linha do grupo ID_PROD_UNID_FAT.

        Depois distribui esse valor proporcionalmente ao TOTAL_ATEND
        dentro do mesmo grupo ID_PROD_UNID_FAT.
        """

        pct_dist_estouro = calcular_percentual_distribuicao(total_atend_base)

        estouro_base_grupo = (
            df.groupby("ID_PROD_UNID_FAT", sort=False)[nome_coluna_base_pcs]
            .transform("last")
            .fillna(0.0)
            .to_numpy(dtype="float64", copy=False)
        )

        return estouro_base_grupo * pct_dist_estouro

    # =============================================================================================
    # Arrays base de prioridade e identificação

    prior_matpar = df["PRIOR_MATPAR"].to_numpy()
    prior_rot = df["PRIOR_ROT"].to_numpy()

    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # IDs usados pela lógica antiga do arraste
    id_prod_unid_fat = df["ID_PROD_UNID_FAT"]
    id_prod_unid_fat_ant = df["ID_PROD_UNID_FAT_ANT"]

    # =============================================================================================
    # Conversão dos campos numéricos

    nec_nao_atend_pcs = col_num("NEC_NAO_ATEND_PCS")
    pcs_hora = col_num("PCS_HORA")
    est_seg_pcs = col_num("EST_SEG_PCS")
    nec_pcs = col_num("NEC_PCS")
    rec_cap_var_hr = col_num("REC_CAP_VAR_HR")
    fer_cap_var_hr = col_num("FER_CAP_VAR_HR")
    nec_atend_pcs = col_num("NEC_ATEND_PCS")
    hor_rec = col_num("HOR_REC")
    hor_fer = col_num("HOR_FER")

    # =============================================================================================
    # NEC_N_ATEND_PCS_REC
    #
    # Fórmula:
    # NEC_N_ATEND_PCS_REC = max(NEC_PCS - REC_CAP_VAR_HR * PCS_HORA, 0)

    nec_n_atend_pcs_rec = np.maximum(
        nec_pcs - (rec_cap_var_hr * pcs_hora),
        0.0
    )

    df["NEC_N_ATEND_PCS_REC"] = nec_n_atend_pcs_rec

    # =============================================================================================
    # NEC_N_ATEND_PCS_FER
    #
    # Fórmula:
    # NEC_N_ATEND_PCS_FER = max(NEC_PCS - FER_CAP_VAR_HR * PCS_HORA, 0)

    nec_n_atend_pcs_fer = np.maximum(
        nec_pcs - (fer_cap_var_hr * pcs_hora),
        0.0
    )

    df["NEC_N_ATEND_PCS_FER"] = nec_n_atend_pcs_fer

    # =============================================================================================
    # Bases HR para cálculo dos percentuais de distribuição

    if "NEC_ATEND_HR" in df.columns:
        nec_atend_hr = col_num("NEC_ATEND_HR")
    else:
        nec_atend_hr = dividir_seguro(nec_atend_pcs, pcs_hora)

    if "NEC_NAO_ATEND_HR" in df.columns:
        nec_nao_atend_hr = col_num("NEC_NAO_ATEND_HR")
    else:
        nec_nao_atend_hr = dividir_seguro(nec_nao_atend_pcs, pcs_hora)

    nec_n_atend_hr_rec = dividir_seguro(nec_n_atend_pcs_rec, pcs_hora)
    nec_n_atend_hr_fer = dividir_seguro(nec_n_atend_pcs_fer, pcs_hora)

    total_atend_geral = nec_atend_hr + nec_nao_atend_hr
    total_atend_rec = nec_atend_hr + nec_n_atend_hr_rec
    total_atend_fer = nec_atend_hr + nec_n_atend_hr_fer

    # =============================================================================================
    # NEC_ESTOURO_PCS / NEC_ESTOURO_HR
    #
    # Geral:
    # - base PCS: NEC_NAO_ATEND_PCS
    # - grupo: ID_PROD_UNID_FAT
    # - peso: TOTAL_ATEND_GERAL da linha / soma TOTAL_ATEND_GERAL do grupo

    nec_estouro_pcs = decompor_estouro(
        nome_coluna_base_pcs="NEC_NAO_ATEND_PCS",
        total_atend_base=total_atend_geral
    )

    nec_estouro_hr = dividir_seguro(nec_estouro_pcs, pcs_hora)

    df["NEC_ESTOURO_PCS"] = nec_estouro_pcs
    df["NEC_ESTOURO_HR"] = nec_estouro_hr

    # =============================================================================================
    # NEC_ARRASTE_PCS
    #
    # Mantém lógica antiga:
    #
    # NEC_ARRASTE_BASE = max(NEC_ESTOURO_PCS - EST_SEG_PCS, 0)
    #
    # Depois:
    # - monta uma série indexada por ID_PROD_UNID_FAT;
    # - busca nas linhas elegíveis usando ID_PROD_UNID_FAT_ANT;
    # - aplica apenas quando PRIOR_MATPAR = 1 e PRIOR_ROT = 1.

    nec_arraste_base = np.maximum(
        nec_estouro_pcs - est_seg_pcs,
        0.0
    )

    serie_arraste = pd.Series(
        nec_arraste_base,
        index=id_prod_unid_fat
    )

    if not serie_arraste.empty:
        serie_arraste = serie_arraste[~serie_arraste.index.duplicated(keep="last")]

    nec_arraste_pcs = np.zeros(n, dtype="float64")

    if mask_prior_rot.any() and not serie_arraste.empty:
        nec_arraste_pcs[mask_prior_rot] = (
            id_prod_unid_fat_ant[mask_prior_rot]
            .map(serie_arraste)
            .fillna(0.0)
            .to_numpy(dtype="float64", copy=False)
        )

    df["NEC_ARRASTE_PCS"] = nec_arraste_pcs

    # =============================================================================================
    # NEC_ESTOURO_PCS_REC / NEC_ESTOURO_HR_REC
    #
    # REC:
    # - base PCS: NEC_N_ATEND_PCS_REC
    # - grupo: ID_PROD_UNID_FAT
    # - peso: TOTAL_ATEND_REC da linha / soma TOTAL_ATEND_REC do grupo

    nec_estouro_pcs_rec = decompor_estouro(
        nome_coluna_base_pcs="NEC_N_ATEND_PCS_REC",
        total_atend_base=total_atend_rec
    )

    nec_estouro_hr_rec = dividir_seguro(
        nec_estouro_pcs_rec,
        pcs_hora
    )

    df["NEC_ESTOURO_PCS_REC"] = nec_estouro_pcs_rec
    df["NEC_ESTOURO_HR_REC"] = nec_estouro_hr_rec

    # =============================================================================================
    # NEC_ESTOURO_PCS_FER / NEC_ESTOURO_HR_FER
    #
    # FER:
    # - base PCS: NEC_N_ATEND_PCS_FER
    # - grupo: ID_PROD_UNID_FAT
    # - peso: TOTAL_ATEND_FER da linha / soma TOTAL_ATEND_FER do grupo

    nec_estouro_pcs_fer = decompor_estouro(
        nome_coluna_base_pcs="NEC_N_ATEND_PCS_FER",
        total_atend_base=total_atend_fer
    )

    nec_estouro_hr_fer = dividir_seguro(
        nec_estouro_pcs_fer,
        pcs_hora
    )

    df["NEC_ESTOURO_PCS_FER"] = nec_estouro_pcs_fer
    df["NEC_ESTOURO_HR_FER"] = nec_estouro_hr_fer

    # =============================================================================================
    # %_OCUP_REC / HR_OCUP_REC
    #
    # Fórmula:
    # %_OCUP_REC =
    # ((NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA) / HOR_REC

    hr_ocup_rec_base = dividir_seguro(
        nec_estouro_pcs_rec + nec_atend_pcs,
        pcs_hora
    )

    ocup_rec = dividir_seguro(
        hr_ocup_rec_base,
        hor_rec
    )

    ocup_rec = np.maximum(ocup_rec, 0.0)

    df["%_OCUP_REC"] = ocup_rec
    df["HR_OCUP_REC"] = hor_rec * ocup_rec

    # =============================================================================================
    # %_OCUP_FER / HR_OCUP_FER
    #
    # Fórmula:
    # %_OCUP_FER =
    # ((NEC_ESTOURO_PCS_FER + NEC_ATEND_PCS) / PCS_HORA) / HOR_FER

    hr_ocup_fer_base = dividir_seguro(
        nec_estouro_pcs_fer + nec_atend_pcs,
        pcs_hora
    )

    ocup_fer = dividir_seguro(
        hr_ocup_fer_base,
        hor_fer
    )

    ocup_fer = np.maximum(ocup_fer, 0.0)

    df["%_OCUP_FER"] = ocup_fer
    df["HR_OCUP_FER"] = hor_fer * ocup_fer

    return df