import numpy as np
import pandas as pd

def calcular_demais_campos(df):
    # Trabalha em uma cópia para não alterar o DataFrame original fora da função.
    df = df.copy()

    # Quantidade de linhas do DataFrame.
    n = len(df)

    # ============================================================
    # Arrays base de prioridade e identificação
    # ============================================================
    prior_matpar = df["PRIOR_MATPAR"].to_numpy()
    prior_rot = df["PRIOR_ROT"].to_numpy()

    id_pri = df["ID_PRI_PRIORI"]
    id_prod = df["ID_PROD_UNID_FAT"]
    id_ant = df["ID_PROD_UNID_FAT_ANT"]

    id_pri_notna = id_pri.notna().to_numpy()

    # ============================================================
    # Conversão dos campos numéricos
    # ============================================================
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

    # ============================================================
    # Máscaras de regra
    # ============================================================
    mask_prior_pri = (prior_matpar == 1) & id_pri_notna
    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # ============================================================
    # Helpers internos
    # ============================================================
    chaves_dist_estouro = [
        df["UNID_FAT"],
        df["COD_PROD"],
        df["PRIOR_MATPAR"],
    ]

    pcs_hora_s = pd.Series(pcs_hora, index=df.index)

    def dividir_array(numerador, denominador):
        return np.divide(
            numerador,
            denominador,
            out=np.zeros(n, dtype=float),
            where=denominador != 0
        )

    def montar_estouro_concentrado(
        valores,
        chave_origem,
        mask_origem,
        chave_destino,
        mask_destino
    ):
        """
        Replica a forma antiga de encontrar o valor do estouro:
        - cria uma série por chave;
        - em duplicidade, mantém o último valor;
        - busca na chave de destino;
        - aplica somente na prioridade principal.

        Depois esse valor concentrado será usado apenas como total base
        para a nova distribuição.
        """

        resultado = np.zeros(n, dtype=float)

        if not mask_origem.any() or not mask_destino.any():
            return resultado

        serie = pd.Series(
            valores[mask_origem],
            index=chave_origem[mask_origem]
        )

        if serie.empty:
            return resultado

        serie = serie[~serie.index.duplicated(keep="last")]

        resultado[mask_destino] = (
            chave_destino[mask_destino]
            .map(serie)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        return resultado

    def distribuir_estouro(estouro_concentrado, total_nec_hr):
        """
        Nova regra de distribuição:

        PESO_NEC_HR =
            TOTAL_NEC_HR / soma TOTAL_NEC_HR do grupo

        PESO_PCS_HORA =
            PCS_HORA / soma PCS_HORA do grupo

        PERC_DIST_ESTOURO =
            (PESO_NEC_HR + PESO_PCS_HORA) / 2

        NEC_ESTOURO_PCS novo =
            total do estouro concentrado no grupo * PERC_DIST_ESTOURO

        Chave lógica:
            UNID_FAT + COD_PROD + PRIOR_MATPAR

        Não cria ID_DIST_ESTOURO físico no DataFrame.
        """

        total_nec_hr_s = pd.Series(total_nec_hr, index=df.index).fillna(0.0)
        estouro_concentrado_s = pd.Series(estouro_concentrado, index=df.index)

        total_nec_hr_prior_1 = total_nec_hr_s.where(mask_prior_pri, 0.0)
        pcs_hora_prior_1 = pcs_hora_s.where(mask_prior_pri, 0.0)
        estouro_prior_1 = estouro_concentrado_s.where(mask_prior_pri, 0.0)

        soma_total_nec_hr_grupo = (
            total_nec_hr_prior_1
            .groupby(chaves_dist_estouro, sort=False)
            .transform("sum")
        )

        soma_pcs_hora_grupo = (
            pcs_hora_prior_1
            .groupby(chaves_dist_estouro, sort=False)
            .transform("sum")
        )

        total_estouro_grupo = (
            estouro_prior_1
            .groupby(chaves_dist_estouro, sort=False)
            .transform("sum")
        )

        arr_total_nec_hr = total_nec_hr_s.to_numpy(dtype=float)
        arr_pcs_hora = pcs_hora_s.to_numpy(dtype=float)

        arr_soma_total_nec_hr = soma_total_nec_hr_grupo.to_numpy(dtype=float)
        arr_soma_pcs_hora = soma_pcs_hora_grupo.to_numpy(dtype=float)
        arr_total_estouro_grupo = total_estouro_grupo.to_numpy(dtype=float)

        peso_nec_hr = np.divide(
            arr_total_nec_hr,
            arr_soma_total_nec_hr,
            out=np.zeros(n, dtype=float),
            where=mask_prior_pri & (arr_soma_total_nec_hr != 0)
        )

        peso_pcs_hora = np.divide(
            arr_pcs_hora,
            arr_soma_pcs_hora,
            out=np.zeros(n, dtype=float),
            where=mask_prior_pri & (arr_soma_pcs_hora != 0)
        )

        perc_dist_estouro = (peso_nec_hr + peso_pcs_hora) / 2.0

        resultado = np.where(
            mask_prior_pri,
            arr_total_estouro_grupo * perc_dist_estouro,
            0.0
        )

        return resultado

    # ============================================================
    # NEC_ESTOURO_PCS
    # Antes: concentrava na primeira prioridade.
    # Agora: encontra o mesmo valor e distribui.
    #
    # TOTAL_NEC_HR geral:
    # (NEC_ATEND_PCS + NEC_NAO_ATEND_PCS) / PCS_HORA
    # ============================================================
    mask_tab_estouro = (nec_nao_atend_pcs > 0) & id_pri_notna

    nec_estouro_pcs_concentrado = montar_estouro_concentrado(
        valores=nec_nao_atend_pcs,
        chave_origem=id_prod,
        mask_origem=mask_tab_estouro,
        chave_destino=id_prod,
        mask_destino=mask_prior_pri
    )

    total_nec_hr = dividir_array(
        nec_atend_pcs + nec_nao_atend_pcs,
        pcs_hora
    )

    nec_estouro_pcs = distribuir_estouro(
        estouro_concentrado=nec_estouro_pcs_concentrado,
        total_nec_hr=total_nec_hr
    )

    df["NEC_ESTOURO_PCS"] = nec_estouro_pcs

    # ============================================================
    # NEC_ESTOURO_HR
    # Recalcula horas a partir do novo NEC_ESTOURO_PCS.
    # ============================================================
    df["NEC_ESTOURO_HR"] = dividir_array(
        nec_estouro_pcs,
        pcs_hora
    )

    # ============================================================
    # NEC_ARRASTE_PCS
    # Mantém a lógica original, mas agora baseada no estouro distribuído.
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
    # NEC_N_ATEND_PCS_REC
    # Mantém o cálculo original.
    # ============================================================
    nec_n_atend_pcs_rec = np.maximum(
        0.0,
        nec_pcs - (rec_cap_var_hr * pcs_hora)
    )

    # ============================================================
    # NEC_N_ATEND_PCS_FER
    # Mantém o cálculo original.
    # ============================================================
    nec_n_atend_pcs_fer = np.maximum(
        0.0,
        nec_pcs - (fer_cap_var_hr * pcs_hora)
    )

    df["NEC_N_ATEND_PCS_REC"] = nec_n_atend_pcs_rec
    df["NEC_N_ATEND_PCS_FER"] = nec_n_atend_pcs_fer

    # ============================================================
    # NEC_ESTOURO_PCS_REC / NEC_ESTOURO_PCS_FER
    # Antes: concentrava REC/FER na primeira prioridade.
    # Agora: encontra o mesmo valor e distribui.
    #
    # REC:
    # TOTAL_NEC_HR_REC =
    # (NEC_ATEND_PCS + NEC_N_ATEND_PCS_REC) / PCS_HORA
    #
    # FER:
    # TOTAL_NEC_HR_FER =
    # (NEC_ATEND_PCS + NEC_N_ATEND_PCS_FER) / PCS_HORA
    # ============================================================
    mask_tab_rec_fer = (
        id_pri_notna
        & (
            (nec_n_atend_pcs_rec > 0)
            | (nec_n_atend_pcs_fer > 0)
        )
    )

    nec_estouro_pcs_rec_concentrado = montar_estouro_concentrado(
        valores=nec_n_atend_pcs_rec,
        chave_origem=id_pri,
        mask_origem=mask_tab_rec_fer,
        chave_destino=id_prod,
        mask_destino=mask_prior_pri
    )

    nec_estouro_pcs_fer_concentrado = montar_estouro_concentrado(
        valores=nec_n_atend_pcs_fer,
        chave_origem=id_pri,
        mask_origem=mask_tab_rec_fer,
        chave_destino=id_prod,
        mask_destino=mask_prior_pri
    )

    total_nec_hr_rec = dividir_array(
        nec_atend_pcs + nec_n_atend_pcs_rec,
        pcs_hora
    )

    total_nec_hr_fer = dividir_array(
        nec_atend_pcs + nec_n_atend_pcs_fer,
        pcs_hora
    )

    nec_estouro_pcs_rec = distribuir_estouro(
        estouro_concentrado=nec_estouro_pcs_rec_concentrado,
        total_nec_hr=total_nec_hr_rec
    )

    nec_estouro_pcs_fer = distribuir_estouro(
        estouro_concentrado=nec_estouro_pcs_fer_concentrado,
        total_nec_hr=total_nec_hr_fer
    )

    df["NEC_ESTOURO_PCS_REC"] = nec_estouro_pcs_rec
    df["NEC_ESTOURO_PCS_FER"] = nec_estouro_pcs_fer

    # ============================================================
    # NEC_ESTOURO_HR_REC
    # Recalcula horas a partir do novo NEC_ESTOURO_PCS_REC.
    # ============================================================
    df["NEC_ESTOURO_HR_REC"] = dividir_array(
        nec_estouro_pcs_rec,
        pcs_hora
    )

    # ============================================================
    # NEC_ESTOURO_HR_FER
    # Recalcula horas a partir do novo NEC_ESTOURO_PCS_FER.
    # ============================================================
    df["NEC_ESTOURO_HR_FER"] = dividir_array(
        nec_estouro_pcs_fer,
        pcs_hora
    )

    # ============================================================
    # %_OCUP_REC
    # Mantém o cálculo original, usando o novo NEC_ESTOURO_PCS_REC.
    # ============================================================
    hr_ocup_rec_base = dividir_array(
        nec_estouro_pcs_rec + nec_atend_pcs,
        pcs_hora
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
    # Mantém o cálculo original, usando o novo NEC_ESTOURO_PCS_FER.
    # ============================================================
    hr_ocup_fer_base = dividir_array(
        nec_estouro_pcs_fer + nec_atend_pcs,
        pcs_hora
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
    # HR_OCUP_REC / HR_OCUP_FER
    # ============================================================
    df["HR_OCUP_FER"] = hor_fer * ocup_fer
    df["HR_OCUP_REC"] = hor_rec * ocup_rec

    return df