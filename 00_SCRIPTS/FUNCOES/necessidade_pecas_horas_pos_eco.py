import numpy as np
import pandas as pd


# --------------------- ### Recalcular Necessidade de Peças e Horas Pós-ECO ### ---------------------
def calc_nec_pcs_hr_pos_eco(df, inds_eco, lote_min_flag, multiplo_emb_flag):
    """
    Recalcula NEC_PCS e NEC_HR após atualização da ECO/dimensão cortada.

    Esta função deve ser usada dentro do motor de cortes, depois que a ECO
    já foi aplicada/atualizada no LTP.

    Diferença para a função inicial calc_nec_pcs_hr:
        - A fórmula de cálculo é a mesma.
        - A ordem de atualização muda.

    Ordem de recálculo:
        1. Primeiro recalcula os INDs que apareceram na ECO.
        2. Depois recalcula os INDs que não apareceram na ECO.

    Motivo:
        Os INDs da ECO sofreram impacto direto do corte.
        Então eles precisam ser recalculados primeiro.
        Depois recalculamos o restante do LTP já considerando o DataFrame
        atualizado.

    Parâmetros:
        df:
            DataFrame completo do LTP.

        inds_eco:
            Lista, array, Series ou set com os valores da coluna IND
            que apareceram na ECO.

        lote_min_flag:
            "SIM" ou outro valor.
            Quando "SIM", aplica regra de LOTE_MIN.

        multiplo_emb_flag:
            "SIM" ou outro valor.
            Quando "SIM", aplica arredondamento por QTD_EMB para PA/MR.

    Retorno:
        df:
            DataFrame final completo, após recalcular os INDs da ECO
            e também os INDs fora da ECO.
            Esta é a saída que deve voltar para bd_LTP_MES.

        df_etapa_1:
            DataFrame completo após recalcular NEC_PCS e NEC_HR somente
            para os INDs que apareceram na ECO.

        df_etapa_2:
            DataFrame completo após recalcular NEC_PCS e NEC_HR para os
            INDs que apareceram na ECO e também para os demais INDs.
            Esta saída é igual ao df final, mas mantida separada para análise.
    """

    df = df.copy()

    if "IND" not in df.columns:
        raise KeyError("Coluna obrigatória não encontrada: IND")

    # Normaliza flags para evitar problema com minúsculo ou espaço.
    lote_min_flag = str(lote_min_flag).strip().upper()
    multiplo_emb_flag = str(multiplo_emb_flag).strip().upper()

    # Normaliza inds_eco para set, deixando a checagem mais rápida.
    if inds_eco is None:
        inds_eco = set()
    else:
        inds_eco = set(inds_eco)

    # Máscara dos INDs que apareceram na ECO.
    mask_inds_eco = df["IND"].isin(inds_eco).to_numpy()

    # Máscara dos demais INDs, ou seja, os que não apareceram na ECO.
    mask_fora_eco = ~mask_inds_eco

    # Etapa 1:
    # recalcula primeiro os INDs da ECO.
    if mask_inds_eco.any():
        df = _recalcular_nec_pcs_hr_por_mask(
            df=df,
            mask=mask_inds_eco,
            lote_min_flag=lote_min_flag,
            multiplo_emb_flag=multiplo_emb_flag
        )

    # Saída intermediária da etapa 1:
    # neste ponto, somente os INDs que apareceram na ECO foram recalculados.
    df_etapa_1 = df.copy()

    # Etapa 2:
    # depois recalcula os INDs fora da ECO.
    if mask_fora_eco.any():
        df = _recalcular_nec_pcs_hr_por_mask(
            df=df,
            mask=mask_fora_eco,
            lote_min_flag=lote_min_flag,
            multiplo_emb_flag=multiplo_emb_flag
        )

    # Saída final da etapa 2:
    # neste ponto, o DataFrame completo já foi recalculado.
    df_etapa_2 = df.copy()

    # Ordem do retorno:
    # 1. df final para continuar o motor como bd_LTP_MES
    # 2. df_etapa_1 para auditoria/análise após recalcular ECO
    # 3. df_etapa_2 para auditoria/análise após recalcular tudo
    return df, df_etapa_1, df_etapa_2


def _recalcular_nec_pcs_hr_por_mask(df, mask, lote_min_flag, multiplo_emb_flag):
    """
    Helper interno.

    Recalcula NEC_PCS e NEC_HR apenas nas linhas indicadas pela máscara.

    A fórmula é a mesma da função inicial calc_nec_pcs_hr:

        base_comum =
            LTP_CART_ARR_MES_ANT
            + LTP_CART_MES_ATUAL
            + LTP_SALDO_PREV_PCS
            + LTP_EST_SEG_PCS
            - LTP_EST_INI_PCS
            - LTP_EST_TRANS_PCS
            - ORI_TOT_PCS
            - TRIANG_TOT_PCS

        Se MESMA_REG == "NAO":
            soma LTP_SALDO_PREV_PROX_MES_PCS

        Depois:
            zera negativos
            soma LTP_COMP_NEC_PCS
            aplica LIMIT_PCS
            aplica LOTE_MIN, se flag = SIM
            aplica múltiplo de embalagem, se flag = SIM
            calcula NEC_HR = NEC_PCS / PCS_HORA
    """

    n = len(df)

    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        return df

    mesma_reg = df["MESMA_REG"].to_numpy()

    tipo_prod = (
        df["TIPO_PROD"]
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    ltp_cart_ant = _col_num(df, "LTP_CART_ARR_MES_ANT")
    ltp_cart_atual = _col_num(df, "LTP_CART_MES_ATUAL")
    ltp_saldo_prev = _col_num(df, "LTP_SALDO_PREV_PCS")
    ltp_nec_comp = _col_num(df, "LTP_COMP_NEC_PCS")
    ltp_saldo_prox = _col_num(df, "LTP_SALDO_PREV_PROX_MES_PCS")
    ltp_est_seg = _col_num(df, "LTP_EST_SEG_PCS")
    ltp_est_ini = _col_num(df, "LTP_EST_INI_PCS")
    ltp_est_trans = _col_num(df, "LTP_EST_TRANS_PCS")
    ori_tot = _col_num(df, "ORI_TOT_PCS")
    triang_tot = _col_num(df, "TRIANG_TOT_PCS")
    limit_pcs = _col_num(df, "LIMIT_PCS")
    lote_min = _col_num(df, "LOTE_MIN")
    qtd_emb = _col_num(df, "QTD_EMB")
    pcs_hora = _col_num(df, "PCS_HORA")

    # ============================================================
    # Base comum da necessidade
    # ============================================================
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

    # ============================================================
    # Se MESMA_REG == "NAO", soma previsão do próximo mês
    # ============================================================
    nec_pcs = np.where(
        mesma_reg == "NAO",
        base_comum + ltp_saldo_prox,
        base_comum
    )

    # ============================================================
    # Zera necessidades negativas
    # ============================================================
    nec_pcs = np.maximum(nec_pcs, 0.0)

    # ============================================================
    # Soma necessidade de componentes
    #
    # Observação:
    # Mantém a regra atual: LTP_COMP_NEC_PCS entra depois de zerar negativo.
    # ============================================================
    nec_pcs = nec_pcs + ltp_nec_comp

    # Só considera positivas para aplicar limitante/lote/embalagem.
    mask_pos = mask & (nec_pcs > 0)

    nec_final = _col_num_existente_ou_zero(df, "NEC_PCS", n)

    if mask_pos.any():
        v = np.maximum(nec_pcs, limit_pcs)

        # ========================================================
        # Lote mínimo
        # ========================================================
        if lote_min_flag == "SIM":
            mask_lote = mask_pos & (lote_min > 0)
            v = np.where(
                mask_lote,
                np.maximum(v, lote_min),
                v
            )

        # ========================================================
        # Múltiplo de embalagem
        #
        # Mantém a regra atual:
        # aplica somente para TIPO_PROD PA ou MR.
        # ========================================================
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

        # Atualiza apenas as linhas da máscara recebida.
        nec_final[mask] = np.where(
            mask_pos[mask],
            v[mask],
            0.0
        )

    else:
        # Se nenhuma linha da máscara tem necessidade positiva,
        # zera NEC_PCS apenas nas linhas dessa máscara.
        nec_final[mask] = 0.0

    # ============================================================
    # NEC_HR = NEC_PCS / PCS_HORA
    # ============================================================
    nec_hr = _col_num_existente_ou_zero(df, "NEC_HR", n)

    nec_hr_calculado = np.divide(
        nec_final,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    # Atualiza apenas as linhas da máscara recebida.
    nec_hr[mask] = nec_hr_calculado[mask]

    df["NEC_PCS"] = nec_final
    df["NEC_HR"] = nec_hr

    return df


def _col_num(df, nome_coluna):
    """
    Converte uma coluna numérica para array float.

    Valores inválidos ou nulos viram zero.
    """
    return pd.to_numeric(
        df[nome_coluna],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)


def _col_num_existente_ou_zero(df, nome_coluna, n):
    """
    Retorna uma coluna existente como array float.

    Se a coluna ainda não existir, retorna array zerado.

    Isso permite que a função atualize somente uma parte do DataFrame
    sem perder valores já calculados nas demais linhas.
    """
    if nome_coluna in df.columns:
        return pd.to_numeric(
            df[nome_coluna],
            errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)

    return np.zeros(n, dtype=float)