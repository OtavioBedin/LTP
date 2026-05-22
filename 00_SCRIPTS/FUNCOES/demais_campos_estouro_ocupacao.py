import numpy as np
import pandas as pd
from collections import defaultdict, deque


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
# REGRA PRINCIPAL:
# O estouro é alocado na primeira prioridade usando ID_PRI_PRIORI.
def calcular_demais_campos(df):
    # Trabalha em uma cópia para não alterar o DataFrame original fora da função.
    df = df.copy()

    # Quantidade de linhas do DataFrame.
    # Usado para criar arrays zerados com o mesmo tamanho.
    n = len(df)

    # ============================================================
    # Arrays base de prioridade e identificação
    # ============================================================

    # PRIOR_MATPAR indica se a linha participa da regra de material/parceiro.
    prior_matpar = df["PRIOR_MATPAR"].to_numpy()

    # PRIOR_ROT indica se a linha participa da regra de roteiro.
    prior_rot = df["PRIOR_ROT"].to_numpy()

    # ID_PRI_PRIORI representa a primeira prioridade.
    # Essa é a coluna central deste código.
    # O estouro será alocado considerando essa primeira prioridade.
    id_pri = df["ID_PRI_PRIORI"]

    # ID atual do produto/unidade/faturamento.
    # Usado como chave de busca em alguns mapeamentos.
    id_prod = df["ID_PROD_UNID_FAT"]

    # ID anterior do produto/unidade/faturamento.
    # Usado para calcular o arraste.
    id_ant = df["ID_PROD_UNID_FAT_ANT"]

    # Máscara booleana indicando quais linhas possuem ID_PRI_PRIORI preenchido.
    id_pri_notna = id_pri.notna().to_numpy()

    # ============================================================
    # Conversão dos campos numéricos
    # ============================================================
    # Todos os campos abaixo são convertidos para número.
    # Valores inválidos ou nulos são tratados como zero.
    # Isso evita erro em cálculos e divisões.

    # Necessidade não atendida em peças.
    # Base principal para calcular NEC_ESTOURO_PCS.
    nec_nao_atend_pcs = pd.to_numeric(
        df["NEC_NAO_ATEND_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Peças produzidas por hora.
    # Usado para converter necessidade em peças para necessidade em horas.
    pcs_hora = pd.to_numeric(
        df["PCS_HORA"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Estoque de segurança em peças.
    # Usado para calcular NEC_ARRASTE_PCS.
    est_seg_pcs = pd.to_numeric(
        df["EST_SEG_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Necessidade total em peças.
    # Base para calcular necessidade não atendida por recurso e ferramenta.
    nec_pcs = pd.to_numeric(
        df["NEC_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Capacidade variável de recurso em horas.
    rec_cap_var_hr = pd.to_numeric(
        df["REC_CAP_VAR_HR"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Capacidade variável de ferramenta em horas.
    fer_cap_var_hr = pd.to_numeric(
        df["FER_CAP_VAR_HR"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Necessidade já atendida em peças.
    # Entra no cálculo de ocupação.
    nec_atend_pcs = pd.to_numeric(
        df["NEC_ATEND_PCS"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Horas disponíveis de recurso.
    hor_rec = pd.to_numeric(
        df["HOR_REC"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # Horas disponíveis de ferramenta.
    hor_fer = pd.to_numeric(
        df["HOR_FER"],
        errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    # ============================================================
    # Máscaras de regra
    # ============================================================

    # Linhas elegíveis para receber estouro pela primeira prioridade.
    #
    # Regra:
    # - PRIOR_MATPAR precisa ser 1;
    # - ID_PRI_PRIORI precisa estar preenchido.
    mask_prior_pri = (prior_matpar == 1) & id_pri_notna

    # Linhas elegíveis para cálculo de arraste.
    #
    # Regra:
    # - PRIOR_MATPAR precisa ser 1;
    # - PRIOR_ROT precisa ser 1.
    mask_prior_rot = (prior_matpar == 1) & (prior_rot == 1)

    # ============================================================
    # NEC_ESTOURO_PCS
    # ============================================================
    # Objetivo:
    # Calcular o estouro em peças e alocar na primeira prioridade.
    #
    # Regra:
    # 1. Considera apenas linhas com NEC_NAO_ATEND_PCS > 0;
    # 2. Considera apenas linhas com ID_PRI_PRIORI preenchido;
    # 3. Cria uma série indexada por ID_PROD_UNID_FAT;
    # 4. Em duplicidade de ID_PROD_UNID_FAT, mantém o último valor;
    # 5. Busca o valor pelo próprio ID_PROD_UNID_FAT;
    # 6. Aplica somente nas linhas elegíveis por mask_prior_pri.
    # ============================================================

    mask_tab_estouro = (nec_nao_atend_pcs > 0) & id_pri_notna

    serie_estouro = pd.Series(
        nec_nao_atend_pcs[mask_tab_estouro],
        index=id_prod[mask_tab_estouro]
    )

    # Se houver IDs duplicados, mantém o último valor encontrado.
    # Isso replica o comportamento de um set_index(...).to_dict(),
    # onde a última ocorrência sobrescreve as anteriores.
    if not serie_estouro.empty:
        serie_estouro = serie_estouro[~serie_estouro.index.duplicated(keep="last")]

    # Inicializa o resultado zerado.
    nec_estouro_pcs = np.zeros(n, dtype=float)

    # Aplica o estouro apenas nas linhas da primeira prioridade.
    if mask_prior_pri.any() and not serie_estouro.empty:
        nec_estouro_pcs[mask_prior_pri] = (
            id_prod[mask_prior_pri]
            .map(serie_estouro)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    df["NEC_ESTOURO_PCS"] = nec_estouro_pcs

    # ============================================================
    # NEC_ESTOURO_HR
    # ============================================================
    # Converte NEC_ESTOURO_PCS para horas.
    #
    # Fórmula:
    # NEC_ESTOURO_HR = NEC_ESTOURO_PCS / PCS_HORA
    #
    # Se PCS_HORA for zero, retorna zero para evitar divisão inválida.
    # ============================================================

    df["NEC_ESTOURO_HR"] = np.divide(
        nec_estouro_pcs,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    # ============================================================
    # NEC_ARRASTE_PCS
    # ============================================================
    # Objetivo:
    # Calcular quanto do estouro deve ser arrastado para o item anterior.
    #
    # Fórmula base:
    # NEC_ARRASTE_BASE = max(NEC_ESTOURO_PCS - EST_SEG_PCS, 0)
    #
    # Depois:
    # - monta uma série indexada por ID_PROD_UNID_FAT;
    # - busca nas linhas elegíveis usando ID_PROD_UNID_FAT_ANT;
    # - aplica apenas quando PRIOR_MATPAR = 1 e PRIOR_ROT = 1.
    # ============================================================

    nec_arraste_base = np.maximum(nec_estouro_pcs - est_seg_pcs, 0.0)

    serie_arraste = pd.Series(
        nec_arraste_base,
        index=id_prod
    )

    # Em caso de ID duplicado, mantém o último valor.
    if not serie_arraste.empty:
        serie_arraste = serie_arraste[~serie_arraste.index.duplicated(keep="last")]

    # Inicializa o arraste zerado.
    nec_arraste_pcs = np.zeros(n, dtype=float)

    # Busca o arraste pelo ID anterior.
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
    # ============================================================
    # Calcula a necessidade não atendida considerando capacidade de recurso.
    #
    # Fórmula:
    # NEC_N_ATEND_PCS_REC =
    # max(NEC_PCS - REC_CAP_VAR_HR * PCS_HORA, 0)
    # ============================================================

    nec_n_atend_pcs_rec = np.maximum(
        0.0,
        nec_pcs - (rec_cap_var_hr * pcs_hora)
    )

    # ============================================================
    # NEC_N_ATEND_PCS_FER
    # ============================================================
    # Calcula a necessidade não atendida considerando capacidade de ferramenta.
    #
    # Fórmula:
    # NEC_N_ATEND_PCS_FER =
    # max(NEC_PCS - FER_CAP_VAR_HR * PCS_HORA, 0)
    # ============================================================

    nec_n_atend_pcs_fer = np.maximum(
        0.0,
        nec_pcs - (fer_cap_var_hr * pcs_hora)
    )

    df["NEC_N_ATEND_PCS_REC"] = nec_n_atend_pcs_rec
    df["NEC_N_ATEND_PCS_FER"] = nec_n_atend_pcs_fer

    # ============================================================
    # NEC_ESTOURO_PCS_REC / NEC_ESTOURO_PCS_FER
    # ============================================================
    # Objetivo:
    # Calcular o estouro em peças para recurso e ferramenta,
    # alocando na primeira prioridade.
    #
    # Regra:
    # 1. Considera linhas com ID_PRI_PRIORI preenchido;
    # 2. Considera linhas onde REC ou FER possuem necessidade não atendida;
    # 3. Cria uma série REC indexada por ID_PRI_PRIORI;
    # 4. Cria uma série FER indexada por ID_PRI_PRIORI;
    # 5. Em duplicidade, mantém o último valor;
    # 6. Busca usando ID_PROD_UNID_FAT;
    # 7. Aplica nas linhas da primeira prioridade.
    # ============================================================

    mask_tab_rec_fer = (
        id_pri_notna
        & (
            (nec_n_atend_pcs_rec > 0)
            | (nec_n_atend_pcs_fer > 0)
        )
    )

    serie_rec = pd.Series(
        nec_n_atend_pcs_rec[mask_tab_rec_fer],
        index=id_pri[mask_tab_rec_fer]
    )

    serie_fer = pd.Series(
        nec_n_atend_pcs_fer[mask_tab_rec_fer],
        index=id_pri[mask_tab_rec_fer]
    )

    # Em caso de ID_PRI_PRIORI duplicado, mantém o último valor.
    if not serie_rec.empty:
        serie_rec = serie_rec[~serie_rec.index.duplicated(keep="last")]

    if not serie_fer.empty:
        serie_fer = serie_fer[~serie_fer.index.duplicated(keep="last")]

    # Inicializa os resultados zerados.
    nec_estouro_pcs_rec = np.zeros(n, dtype=float)
    nec_estouro_pcs_fer = np.zeros(n, dtype=float)

    # Aloca estouro de recurso nas linhas da primeira prioridade.
    if mask_prior_pri.any() and not serie_rec.empty:
        nec_estouro_pcs_rec[mask_prior_pri] = (
            id_prod[mask_prior_pri]
            .map(serie_rec)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    # Aloca estouro de ferramenta nas linhas da primeira prioridade.
    if mask_prior_pri.any() and not serie_fer.empty:
        nec_estouro_pcs_fer[mask_prior_pri] = (
            id_prod[mask_prior_pri]
            .map(serie_fer)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    df["NEC_ESTOURO_PCS_REC"] = nec_estouro_pcs_rec
    df["NEC_ESTOURO_PCS_FER"] = nec_estouro_pcs_fer

    # ============================================================
    # NEC_ESTOURO_HR_REC
    # ============================================================
    # Converte o estouro de recurso de peças para horas.
    #
    # Fórmula:
    # NEC_ESTOURO_HR_REC = NEC_ESTOURO_PCS_REC / PCS_HORA
    # ============================================================

    df["NEC_ESTOURO_HR_REC"] = np.divide(
        nec_estouro_pcs_rec,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    # ============================================================
    # NEC_ESTOURO_HR_FER
    # ============================================================
    # Converte o estouro de ferramenta de peças para horas.
    #
    # Fórmula:
    # NEC_ESTOURO_HR_FER = NEC_ESTOURO_PCS_FER / PCS_HORA
    # ============================================================

    df["NEC_ESTOURO_HR_FER"] = np.divide(
        nec_estouro_pcs_fer,
        pcs_hora,
        out=np.zeros(n, dtype=float),
        where=pcs_hora != 0
    )

    # ============================================================
    # %_OCUP_REC
    # ============================================================
    # Calcula o percentual de ocupação do recurso.
    #
    # Fórmula:
    # %_OCUP_REC =
    # ((NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA) / HOR_REC
    #
    # Regras:
    # - se PCS_HORA = 0, retorna 0;
    # - se HOR_REC = 0, retorna 0;
    # - se o resultado for negativo, força para 0.
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
    # Calcula o percentual de ocupação da ferramenta.
    #
    # Fórmula:
    # %_OCUP_FER =
    # ((NEC_ESTOURO_PCS_FER + NEC_ATEND_PCS) / PCS_HORA) / HOR_FER
    #
    # Regras:
    # - se PCS_HORA = 0, retorna 0;
    # - se HOR_FER = 0, retorna 0;
    # - se o resultado for negativo, força para 0.
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
    # HR_OCUP_REC / HR_OCUP_FER
    # ============================================================
    # Converte o percentual de ocupação em horas ocupadas.
    #
    # Fórmulas:
    # HR_OCUP_REC = HOR_REC * %_OCUP_REC
    # HR_OCUP_FER = HOR_FER * %_OCUP_FER
    # ============================================================

    df["HR_OCUP_FER"] = hor_fer * ocup_fer
    df["HR_OCUP_REC"] = hor_rec * ocup_rec

    return df