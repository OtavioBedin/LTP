import numpy as np
import pandas as pd


def calcular_demais_campos(df):
    """
    Recalcula os campos derivados da LTP.

    Regra ajustada do estouro:
        - Nao existe mais distribuicao proporcional de estouro.
        - Nao usa mais peso por TOTAL_NEC_HR.
        - Nao usa mais peso por PCS_HORA.
        - Nao usa mais grupo UNID_FAT + COD_PROD + PRIOR_MATPAR para ratear.

    Nova regra:
        Para cada ID_PROD_UNID_FAT:
            1. encontra a ultima ocorrencia fisica desse ID no DataFrame;
            2. pega o valor da propria linha dessa ultima ocorrencia;
            3. grava esse valor no respectivo campo de estouro, na mesma linha;
            4. todas as outras linhas do mesmo ID ficam com zero no campo de estouro.

    Campos de origem e destino:
        NEC_NAO_ATEND_PCS     -> NEC_ESTOURO_PCS
        NEC_N_ATEND_PCS_REC   -> NEC_ESTOURO_PCS_REC
        NEC_N_ATEND_PCS_FER   -> NEC_ESTOURO_PCS_FER
    """

    # Trabalha em uma copia para nao alterar o DataFrame original fora da funcao.
    df = df.copy()

    # Quantidade de linhas do DataFrame.
    n = len(df)

    # ============================================================
    # Arrays base de prioridade e identificacao
    # ============================================================
    prior_matpar = df["PRIOR_MATPAR"].to_numpy()
    prior_rot = df["PRIOR_ROT"].to_numpy()

    id_prod = df["ID_PROD_UNID_FAT"]
    id_ant = df["ID_PROD_UNID_FAT_ANT"]

    # Mascara dos IDs validos. Linhas sem ID_PROD_UNID_FAT nao recebem estouro.
    id_prod_notna = id_prod.notna().to_numpy()

    # ============================================================
    # Conversao dos campos numericos
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
    # Mascaras de regra
    # ============================================================
    # Mantida apenas para a regra original do NEC_ARRASTE_PCS.
    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # ============================================================
    # Helpers internos
    # ============================================================
    def dividir_array(numerador, denominador):
        """
        Divisao segura entre arrays.

        Quando o denominador for zero, retorna zero.
        Evita inf, -inf e erro de divisao por zero.
        """
        return np.divide(
            numerador,
            denominador,
            out=np.zeros(n, dtype=float),
            where=denominador != 0
        )

    def atribuir_estouro_na_ultima_ocorrencia_id(valores):
        """
        Atribui o valor direto no campo de estouro.

        Regra:
            - Para cada ID_PROD_UNID_FAT;
            - encontra a ultima ocorrencia fisica desse ID no DataFrame;
            - pega o valor da propria linha dessa ultima ocorrencia;
            - grava esse valor no campo de estouro, nessa mesma linha;
            - todas as demais linhas ficam zero.

        Nao distribui.
        Nao rateia.
        Nao usa peso.
        Nao usa PRIOR_MATPAR como filtro de destino.
        Nao usa UNID_FAT + COD_PROD + PRIOR_MATPAR.
        """
        resultado = np.zeros(n, dtype=float)

        if n == 0 or not id_prod_notna.any():
            return resultado

        valores_s = (
            pd.Series(valores, index=df.index)
            .fillna(0.0)
            .astype(float)
        )

        # Posicao fisica da linha dentro do DataFrame atual.
        posicao_linha = np.arange(n)

        # Cria uma serie indexada por ID_PROD_UNID_FAT contendo a posicao da linha.
        # Como o mesmo ID pode aparecer varias vezes, keep="last" mantem a ultima
        # ocorrencia fisica de cada ID no DataFrame.
        ultima_posicao_por_id = pd.Series(
            posicao_linha[id_prod_notna],
            index=id_prod[id_prod_notna].to_numpy()
        )

        ultima_posicao_por_id = ultima_posicao_por_id[
            ~ultima_posicao_por_id.index.duplicated(keep="last")
        ]

        if ultima_posicao_por_id.empty:
            return resultado

        # Destino: ultima linha fisica de cada ID_PROD_UNID_FAT.
        posicoes_destino = ultima_posicao_por_id.to_numpy(dtype=int)

        # Valor: valor da propria linha encontrada como ultima ocorrencia.
        resultado[posicoes_destino] = valores_s.iloc[posicoes_destino].to_numpy(dtype=float)

        return resultado

    # ============================================================
    # NEC_ESTOURO_PCS
    #
    # Origem:
    #     NEC_NAO_ATEND_PCS
    #
    # Destino:
    #     NEC_ESTOURO_PCS
    #
    # Regra:
    #     Para cada ID_PROD_UNID_FAT, pega NEC_NAO_ATEND_PCS da ultima
    #     ocorrencia fisica desse ID e grava em NEC_ESTOURO_PCS nessa mesma linha.
    #     As demais linhas ficam zero.
    # ============================================================
    nec_estouro_pcs = atribuir_estouro_na_ultima_ocorrencia_id(
        valores=nec_nao_atend_pcs
    )

    df["NEC_ESTOURO_PCS"] = nec_estouro_pcs

    # ============================================================
    # NEC_ESTOURO_HR
    #
    # Recalcula horas a partir do novo NEC_ESTOURO_PCS.
    # ============================================================
    df["NEC_ESTOURO_HR"] = dividir_array(
        nec_estouro_pcs,
        pcs_hora
    )

    # ============================================================
    # NEC_ARRASTE_PCS
    #
    # Mantem a logica original:
    #
    #     NEC_ARRASTE_BASE = max(NEC_ESTOURO_PCS - EST_SEG_PCS, 0)
    #
    # Depois:
    #     - cria uma serie indexada por ID_PROD_UNID_FAT;
    #     - em duplicidade, mantem a ultima ocorrencia;
    #     - busca usando ID_PROD_UNID_FAT_ANT;
    #     - aplica somente quando PRIOR_MATPAR = 1 e PRIOR_ROT = 1.
    #
    # Observacao:
    #     A base usa o NEC_ESTOURO_PCS ja ajustado pela nova regra.
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
    #
    # Mantem o calculo original:
    #
    #     NEC_N_ATEND_PCS_REC = max(NEC_PCS - REC_CAP_VAR_HR * PCS_HORA, 0)
    # ============================================================
    nec_n_atend_pcs_rec = np.maximum(
        0.0,
        nec_pcs - (rec_cap_var_hr * pcs_hora)
    )

    # ============================================================
    # NEC_N_ATEND_PCS_FER
    #
    # Mantem o calculo original:
    #
    #     NEC_N_ATEND_PCS_FER = max(NEC_PCS - FER_CAP_VAR_HR * PCS_HORA, 0)
    # ============================================================
    nec_n_atend_pcs_fer = np.maximum(
        0.0,
        nec_pcs - (fer_cap_var_hr * pcs_hora)
    )

    df["NEC_N_ATEND_PCS_REC"] = nec_n_atend_pcs_rec
    df["NEC_N_ATEND_PCS_FER"] = nec_n_atend_pcs_fer

    # ============================================================
    # NEC_ESTOURO_PCS_REC
    #
    # Origem:
    #     NEC_N_ATEND_PCS_REC
    #
    # Destino:
    #     NEC_ESTOURO_PCS_REC
    #
    # Regra:
    #     Para cada ID_PROD_UNID_FAT, pega NEC_N_ATEND_PCS_REC da ultima
    #     ocorrencia fisica desse ID e grava em NEC_ESTOURO_PCS_REC nessa mesma linha.
    #     As demais linhas ficam zero.
    # ============================================================
    nec_estouro_pcs_rec = atribuir_estouro_na_ultima_ocorrencia_id(
        valores=nec_n_atend_pcs_rec
    )

    # ============================================================
    # NEC_ESTOURO_PCS_FER
    #
    # Origem:
    #     NEC_N_ATEND_PCS_FER
    #
    # Destino:
    #     NEC_ESTOURO_PCS_FER
    #
    # Regra:
    #     Para cada ID_PROD_UNID_FAT, pega NEC_N_ATEND_PCS_FER da ultima
    #     ocorrencia fisica desse ID e grava em NEC_ESTOURO_PCS_FER nessa mesma linha.
    #     As demais linhas ficam zero.
    # ============================================================
    nec_estouro_pcs_fer = atribuir_estouro_na_ultima_ocorrencia_id(
        valores=nec_n_atend_pcs_fer
    )

    df["NEC_ESTOURO_PCS_REC"] = nec_estouro_pcs_rec
    df["NEC_ESTOURO_PCS_FER"] = nec_estouro_pcs_fer

    # ============================================================
    # NEC_ESTOURO_HR_REC
    #
    # Recalcula horas a partir do novo NEC_ESTOURO_PCS_REC.
    # ============================================================
    df["NEC_ESTOURO_HR_REC"] = dividir_array(
        nec_estouro_pcs_rec,
        pcs_hora
    )

    # ============================================================
    # NEC_ESTOURO_HR_FER
    #
    # Recalcula horas a partir do novo NEC_ESTOURO_PCS_FER.
    # ============================================================
    df["NEC_ESTOURO_HR_FER"] = dividir_array(
        nec_estouro_pcs_fer,
        pcs_hora
    )

    # ============================================================
    # %_OCUP_REC
    #
    # Mantem o calculo original, usando o novo NEC_ESTOURO_PCS_REC:
    #
    #     HR_OCUP_REC_BASE = (NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA
    #     %_OCUP_REC      = HR_OCUP_REC_BASE / HOR_REC
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
    #
    # Mantem o calculo original, usando o novo NEC_ESTOURO_PCS_FER:
    #
    #     HR_OCUP_FER_BASE = (NEC_ESTOURO_PCS_FER + NEC_ATEND_PCS) / PCS_HORA
    #     %_OCUP_FER      = HR_OCUP_FER_BASE / HOR_FER
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
    #
    # Converte o percentual de ocupacao novamente em horas ocupadas.
    # ============================================================
    df["HR_OCUP_FER"] = hor_fer * ocup_fer
    df["HR_OCUP_REC"] = hor_rec * ocup_rec

    return df