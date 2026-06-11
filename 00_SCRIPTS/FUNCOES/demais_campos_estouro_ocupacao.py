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

    id_prod_notna = id_prod.notna().to_numpy()
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
    # Primeira prioridade:
    # continua sendo o ponto onde o estouro é inicialmente alocado/concentrado.
    mask_prior_pri = (prior_matpar == 1) & id_pri_notna

    # Prioridade de roteiro:
    # usada na lógica antiga do NEC_ARRASTE_PCS.
    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # ============================================================
    # Helpers internos
    # ============================================================
    # Grupo lógico usado para distribuir o estouro depois que ele é encontrado.
    #
    # Mantido como:
    # UNID_FAT + COD_PROD + PRIOR_MATPAR
    #
    # Essa chave não é criada fisicamente no DataFrame.
    chaves_dist_estouro = [
        df["UNID_FAT"],
        df["COD_PROD"],
        df["PRIOR_MATPAR"],
    ]

    pcs_hora_s = pd.Series(pcs_hora, index=df.index)

    def dividir_array(numerador, denominador):
        """
        Divisão segura entre arrays.

        Quando o denominador for zero, retorna zero.
        Evita inf, -inf e erro de divisão por zero.
        """
        return np.divide(
            numerador,
            denominador,
            out=np.zeros(n, dtype=float),
            where=denominador != 0
        )

    def buscar_ultima_ocorrencia_por_id_prod(valores, mask_destino):
        """
        Busca o valor de estouro pela nova referência definida:

        Referência:
            ID_PROD_UNID_FAT

        Regra:
            Para cada linha de destino:
            1. pega o ID_PROD_UNID_FAT da própria linha;
            2. procura a última ocorrência desse mesmo ID_PROD_UNID_FAT
               no DataFrame;
            3. retorna o valor da coluna informada em `valores`
               nessa última ocorrência.

        Importante:
            - A origem dos valores NÃO muda.
            - O que muda é somente a chave/referência de busca.

        Exemplos:
            NEC_ESTOURO_PCS:
                busca a última ocorrência do ID_PROD_UNID_FAT
                e retorna NEC_NAO_ATEND_PCS.

            NEC_ESTOURO_PCS_REC:
                busca a última ocorrência do ID_PROD_UNID_FAT
                e retorna NEC_N_ATEND_PCS_REC.

            NEC_ESTOURO_PCS_FER:
                busca a última ocorrência do ID_PROD_UNID_FAT
                e retorna NEC_N_ATEND_PCS_FER.

        Depois desse retorno, o valor ainda passa pela função
        distribuir_estouro().
        """

        resultado = np.zeros(n, dtype=float)

        if not id_prod_notna.any() or not mask_destino.any():
            return resultado

        # Cria uma série indexada por ID_PROD_UNID_FAT.
        # Como o índice pode ter duplicidades, mantém a última ocorrência.
        serie_ultima_ocorrencia = pd.Series(
            valores[id_prod_notna],
            index=id_prod[id_prod_notna]
        )

        if serie_ultima_ocorrencia.empty:
            return resultado

        # Mantém exatamente a última ocorrência de cada ID_PROD_UNID_FAT.
        serie_ultima_ocorrencia = serie_ultima_ocorrencia[
            ~serie_ultima_ocorrencia.index.duplicated(keep="last")
        ]

        # Para cada linha de destino, usa o ID_PROD_UNID_FAT da própria linha
        # para buscar o valor da última ocorrência desse mesmo ID_PROD_UNID_FAT.
        resultado[mask_destino] = (
            id_prod[mask_destino]
            .map(serie_ultima_ocorrencia)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        return resultado

    def distribuir_estouro(estouro_concentrado, total_nec_hr):
        """
        Distribui o estouro encontrado/concentrado.

        A busca do valor-base do estouro já foi feita antes, pela regra:
            ID_PROD_UNID_FAT -> última ocorrência de ID_PROD_UNID_FAT.

        Aqui a função apenas distribui o total encontrado dentro do grupo:

            UNID_FAT + COD_PROD + PRIOR_MATPAR

        Regras de peso mantidas neste módulo:

            PESO_NEC_HR =
                TOTAL_NEC_HR da linha
                /
                soma TOTAL_NEC_HR do grupo

            PESO_PCS_HORA =
                PCS_HORA da linha
                /
                soma PCS_HORA do grupo

            PERC_DIST_ESTOURO =
                (PESO_NEC_HR + PESO_PCS_HORA) / 2

            NEC_ESTOURO_PCS distribuído =
                total do estouro concentrado no grupo
                *
                PERC_DIST_ESTOURO

        Participação:
            somente linhas de primeira prioridade participam,
            conforme mask_prior_pri.
        """

        total_nec_hr_s = pd.Series(total_nec_hr, index=df.index).fillna(0.0)
        estouro_concentrado_s = pd.Series(estouro_concentrado, index=df.index)

        # Só a primeira prioridade participa da distribuição.
        total_nec_hr_prior_1 = total_nec_hr_s.where(mask_prior_pri, 0.0)
        pcs_hora_prior_1 = pcs_hora_s.where(mask_prior_pri, 0.0)
        estouro_prior_1 = estouro_concentrado_s.where(mask_prior_pri, 0.0)

        # Soma dos pesos dentro do grupo lógico de distribuição.
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

        # Total de estouro encontrado/concentrado dentro do grupo.
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
    #
    # Origem do valor:
    #     NEC_NAO_ATEND_PCS
    #
    # Referência de busca:
    #     ID_PROD_UNID_FAT
    #
    # Regra:
    #     Para cada linha de primeira prioridade, pega o ID_PROD_UNID_FAT
    #     da própria linha, encontra a última ocorrência desse mesmo
    #     ID_PROD_UNID_FAT e retorna NEC_NAO_ATEND_PCS dessa última ocorrência.
    #
    # Depois:
    #     distribui o valor encontrado pelo grupo
    #     UNID_FAT + COD_PROD + PRIOR_MATPAR.
    #
    # TOTAL_NEC_HR geral:
    #     (NEC_ATEND_PCS + NEC_NAO_ATEND_PCS) / PCS_HORA
    # ============================================================
    nec_estouro_pcs_concentrado = buscar_ultima_ocorrencia_por_id_prod(
        valores=nec_nao_atend_pcs,
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
    # Mantém a lógica original:
    #
    #     NEC_ARRASTE_BASE =
    #         max(NEC_ESTOURO_PCS - EST_SEG_PCS, 0)
    #
    # Depois:
    #     - cria uma série indexada por ID_PROD_UNID_FAT;
    #     - em duplicidade, mantém a última ocorrência;
    #     - busca usando ID_PROD_UNID_FAT_ANT;
    #     - aplica somente quando PRIOR_MATPAR = 1 e PRIOR_ROT = 1.
    #
    # Observação:
    #     A base agora usa o NEC_ESTOURO_PCS já distribuído.
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
    # Mantém o cálculo original:
    #
    #     NEC_N_ATEND_PCS_REC =
    #         max(NEC_PCS - REC_CAP_VAR_HR * PCS_HORA, 0)
    # ============================================================
    nec_n_atend_pcs_rec = np.maximum(
        0.0,
        nec_pcs - (rec_cap_var_hr * pcs_hora)
    )

    # ============================================================
    # NEC_N_ATEND_PCS_FER
    #
    # Mantém o cálculo original:
    #
    #     NEC_N_ATEND_PCS_FER =
    #         max(NEC_PCS - FER_CAP_VAR_HR * PCS_HORA, 0)
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
    # Origem do valor:
    #     NEC_N_ATEND_PCS_REC
    #
    # Referência de busca:
    #     ID_PROD_UNID_FAT
    #
    # Regra:
    #     Para cada linha de primeira prioridade, pega o ID_PROD_UNID_FAT
    #     da própria linha, encontra a última ocorrência desse mesmo
    #     ID_PROD_UNID_FAT e retorna NEC_N_ATEND_PCS_REC dessa última ocorrência.
    #
    # Depois:
    #     distribui o valor encontrado pelo grupo
    #     UNID_FAT + COD_PROD + PRIOR_MATPAR.
    #
    # TOTAL_NEC_HR_REC:
    #     (NEC_ATEND_PCS + NEC_N_ATEND_PCS_REC) / PCS_HORA
    # ============================================================
    nec_estouro_pcs_rec_concentrado = buscar_ultima_ocorrencia_por_id_prod(
        valores=nec_n_atend_pcs_rec,
        mask_destino=mask_prior_pri
    )

    total_nec_hr_rec = dividir_array(
        nec_atend_pcs + nec_n_atend_pcs_rec,
        pcs_hora
    )

    nec_estouro_pcs_rec = distribuir_estouro(
        estouro_concentrado=nec_estouro_pcs_rec_concentrado,
        total_nec_hr=total_nec_hr_rec
    )

    # ============================================================
    # NEC_ESTOURO_PCS_FER
    #
    # Origem do valor:
    #     NEC_N_ATEND_PCS_FER
    #
    # Referência de busca:
    #     ID_PROD_UNID_FAT
    #
    # Regra:
    #     Para cada linha de primeira prioridade, pega o ID_PROD_UNID_FAT
    #     da própria linha, encontra a última ocorrência desse mesmo
    #     ID_PROD_UNID_FAT e retorna NEC_N_ATEND_PCS_FER dessa última ocorrência.
    #
    # Depois:
    #     distribui o valor encontrado pelo grupo
    #     UNID_FAT + COD_PROD + PRIOR_MATPAR.
    #
    # TOTAL_NEC_HR_FER:
    #     (NEC_ATEND_PCS + NEC_N_ATEND_PCS_FER) / PCS_HORA
    # ============================================================
    nec_estouro_pcs_fer_concentrado = buscar_ultima_ocorrencia_por_id_prod(
        valores=nec_n_atend_pcs_fer,
        mask_destino=mask_prior_pri
    )

    total_nec_hr_fer = dividir_array(
        nec_atend_pcs + nec_n_atend_pcs_fer,
        pcs_hora
    )

    nec_estouro_pcs_fer = distribuir_estouro(
        estouro_concentrado=nec_estouro_pcs_fer_concentrado,
        total_nec_hr=total_nec_hr_fer
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
    # Mantém o cálculo original, usando o novo NEC_ESTOURO_PCS_REC:
    #
    #     HR_OCUP_REC_BASE =
    #         (NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA
    #
    #     %_OCUP_REC =
    #         HR_OCUP_REC_BASE / HOR_REC
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
    # Mantém o cálculo original, usando o novo NEC_ESTOURO_PCS_FER:
    #
    #     HR_OCUP_FER_BASE =
    #         (NEC_ESTOURO_PCS_FER + NEC_ATEND_PCS) / PCS_HORA
    #
    #     %_OCUP_FER =
    #         HR_OCUP_FER_BASE / HOR_FER
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
    # Converte o percentual de ocupação novamente em horas ocupadas.
    # ============================================================
    df["HR_OCUP_FER"] = hor_fer * ocup_fer
    df["HR_OCUP_REC"] = hor_rec * ocup_rec

    return df