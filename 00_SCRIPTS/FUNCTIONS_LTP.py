
import numpy as np
import pandas as pd

# --------------------- ### Calcular Necessidade de Peças e Horas ### ---------------------
def calc_nec_pcs_hr(df, lote_min_flag, multiplo_emb_flag):
    df = df.copy()
    n = len(df)

    nec_pcs_arr = np.zeros(n)
    nec_hr_arr = np.zeros(n)

    # Colunas necessárias
    mesma_reg = df['MESMA_REG'].values
    tipo_prod = df['TIPO_PROD'].astype(str).str.upper().values
    ltp_cart_ant = df['LTP_CART_ARR_MES_ANT'].fillna(0).values
    ltp_cart_atual = df['LTP_CART_MES_ATUAL'].fillna(0).values
    ltp_saldo_prev = df['LTP_SALDO_PREV_PCS'].fillna(0).values
    ltp_nec_comp = df['LTP_COMP_NEC_PCS'].fillna(0).values
    ltp_saldo_prox = df['LTP_SALDO_PREV_PROX_MES_PCS'].fillna(0).values
    ltp_est_seg = df['LTP_EST_SEG_PCS'].fillna(0).values
    ltp_est_ini = df['LTP_EST_INI_PCS'].fillna(0).values
    ltp_est_trans = df['LTP_EST_TRANS_PCS'].fillna(0).values
    ori_tot = df['ORI_TOT_PCS'].fillna(0).values
    triang_tot = df['TRIANG_TOT_PCS'].fillna(0).values
    limit_pcs = df['LIMIT_PCS'].astype(float).fillna(0.0).values
    lote_min = df['LOTE_MIN'].fillna(0).values
    qtd_emb = df['QTD_EMB'].fillna(0).values
    pcs_hora = df['PCS_HORA'].fillna(0).values

    for i in range(n):
        if mesma_reg[i] == 'NAO':
            nec_pcs = (
                ltp_cart_ant[i] + ltp_cart_atual[i] + ltp_saldo_prev[i] + ltp_saldo_prox[i] + ltp_est_seg[i] -
                (ltp_est_ini[i] + ltp_est_trans[i] + ori_tot[i] + triang_tot[i])
            )
        else:
            nec_pcs = (
                ltp_cart_ant[i] + ltp_cart_atual[i] + ltp_saldo_prev[i] + ltp_est_seg[i] -
                (ltp_est_ini[i] + ltp_est_trans[i] + ori_tot[i] + triang_tot[i])
            )
            
        if nec_pcs < 0:
            nec_pcs = 0
        
        nec_pcs = nec_pcs + ltp_nec_comp[i]
        
        if nec_pcs > 0:
            # limite mínimo operacional
            var_nec1 = max(nec_pcs, limit_pcs[i])

            # lote mínimo por produto
            if lote_min_flag == 'SIM' and lote_min[i] > 0:
                var_nec2 = max(var_nec1, lote_min[i])
            else:
                var_nec2 = var_nec1

            # múltiplo de embalagem (PA ou MR)
            tipo = str(tipo_prod[i]).strip().upper()

            if multiplo_emb_flag == 'SIM' and tipo in ('PA', 'MR') and qtd_emb[i] > 0:
                var_nec3 = int(np.ceil(var_nec2 / qtd_emb[i]) * qtd_emb[i])
            else:
                var_nec3 = var_nec2

            nec_pcs_final = var_nec3
        else:
            nec_pcs_final = 0

        nec_pcs_arr[i] = nec_pcs_final
        nec_hr_arr[i] = nec_pcs_final / pcs_hora[i] if pcs_hora[i] else 0

    df['NEC_PCS'] = nec_pcs_arr
    df['NEC_HR'] = nec_hr_arr

    return df

# --------------------- ### Calcular Distribuição de Capacidade ### ---------------------
def calcular_distrib_capacidade(df, lote_min_flag, multiplo_emb_flag):
    df = df.copy()
    n = len(df)
    
    # ----- Criar colunas necessárias para calculos
    colunas_calculos = [
        'REC_CAP_VAR_HR', 'FER_CAP_VAR_HR', 'ROT_CAP_VAR_HR',
        'REC_HR_SALDO', 'FER_HR_SALDO',
        'CAP_PROD_PCS', 'NEC_ATEND_PCS', 'NEC_ATEND_HR',
        'NEC_NAO_ATEND_PCS', 'NEC_NAO_ATEND_HR'
    ]
    
    for col in colunas_calculos:
        if col not in df.columns:
            df[col] = 0.0
    
    df['REC_HR_SALDO'] = df['HOR_REC']
    df['FER_HR_SALDO'] = df['HOR_FER']
            
    # ----- Criar Dicionários de Lookup para Recursos e Ferramentas
    tab_HOR_REC = df[['ID_RECURSO', 'HOR_REC', 'NEC_ATEND_HR', 'REC_HR_SALDO']].drop_duplicates(subset=['ID_RECURSO']).reset_index(drop=True)
    dict_hor_rec = tab_HOR_REC.set_index('ID_RECURSO')[['HOR_REC', 'NEC_ATEND_HR', 'REC_HR_SALDO']].to_dict(orient='index')         
    
    tab_HOR_FER = df[['ID_FERRAMENTA', 'HOR_FER', 'NEC_ATEND_HR', 'FER_HR_SALDO']].drop_duplicates(subset=['ID_FERRAMENTA']).reset_index(drop=True)
    dict_hor_fer = tab_HOR_FER.set_index('ID_FERRAMENTA')[['HOR_FER', 'NEC_ATEND_HR', 'FER_HR_SALDO']].to_dict(orient='index')

    # Inicialização das colunas necessárias
    prior_matpar = df['PRIOR_MATPAR'].fillna(0).astype(int).values
    prior_rot = df['PRIOR_ROT'].fillna(0).astype(int).values
    id_recurso = df['ID_RECURSO'].fillna('').astype(str).values
    id_ferramenta = df['ID_FERRAMENTA'].fillna('').astype(str).values
    nec_pcs = df['NEC_PCS'].fillna(0).astype(float).values
    nec_hr = df['NEC_HR'].fillna(0).astype(float).values    
    pcs_hora = df['PCS_HORA'].fillna(0).astype(float).values
    rec_cap_var_hr = df['REC_CAP_VAR_HR'].fillna(0).astype(float).values
    limit_pcs = df['LIMIT_PCS'].astype(float).fillna(0.0).values
    lote_min = df['LOTE_MIN'].fillna(0).values
    qtd_emb = df['QTD_EMB'].fillna(0).values
    tipo_prod = df['TIPO_PROD'].astype(str).str.upper().values

    # Inicializa arrays de resultado
    rec_cap_var_hr = np.zeros(n)
    fer_cap_var_hr = np.zeros(n)
    rec_hr_saldo = np.zeros(n)
    fer_hr_saldo = np.zeros(n)
    rot_cap_var_hr = np.zeros(n)
    cap_prod_pcs = np.zeros(n)
    nec_atend_pcs = np.zeros(n)
    nec_atend_hr = np.zeros(n)
    nec_nao_atend_pcs = np.zeros(n)
    nec_nao_atend_hr = np.zeros(n)
    soma_nec_atend_hr_por_recurso = {}

    for i in range(n):
        # Regra para atualizar o NEC_PCS e NEC_HR após os calculos
        if prior_matpar[i] != 1 or prior_rot[i] != 1:
            nec_pcs[i] = nec_nao_atend_pcs[i - 1]
        else:
            nec_pcs[i] = nec_pcs[i]
            
        # Aplicando regra do flag no NEC_PCS e NEC_HR
        if nec_pcs[i] > 0:
            var_nec1 = max(nec_pcs[i], limit_pcs[i])
            var_nec2 = var_nec1 if lote_min_flag == 'NAO' else (
                lote_min[i] if var_nec1 < lote_min[i] else var_nec1)
            var_nec3 = int(np.ceil(var_nec2 / qtd_emb[i]) * qtd_emb[i]) if tipo_prod[i] in ['PA', 'MR'] and multiplo_emb_flag != 'NAO' and qtd_emb[i] != 0 else var_nec2
            nec_pcs[i] = max(var_nec3, 0)
        else:
            nec_pcs[i] = 0
            
        nec_hr[i] = nec_pcs[i] / pcs_hora[i] if pcs_hora[i] != 0 else 0
        rec_cap_var_hr[i] = dict_hor_rec.get(id_recurso[i], {}).get('REC_HR_SALDO', 0)
        fer_cap_var_hr[i] = dict_hor_fer.get(id_ferramenta[i], {}).get('FER_HR_SALDO', 0)
        rot_cap_var_hr[i] = min(rec_cap_var_hr[i], fer_cap_var_hr[i])
        cap_prod_pcs[i] = rot_cap_var_hr[i] * pcs_hora[i] if pcs_hora[i] != 0 else 0
        nec_atend_pcs[i] = min(cap_prod_pcs[i], nec_pcs[i])
        nec_atend_hr[i] = nec_atend_pcs[i] / pcs_hora[i] if pcs_hora[i] != 0 else 0
        rec_hr_saldo[i] = max(rec_cap_var_hr[i] - nec_atend_hr[i], 0)
        fer_hr_saldo[i] = max(fer_cap_var_hr[i] - nec_atend_hr[i], 0)
        nec_nao_atend_pcs[i] = nec_pcs[i] - nec_atend_pcs[i]
        nec_nao_atend_hr[i] = nec_nao_atend_pcs[i] / pcs_hora[i] if pcs_hora[i] != 0 else 0
        
        # Agregar ID_RECURSO e ID_FERRAMENTA para atualizar os dicionários, somando os valores de nec_atend_hr
        recurso = id_recurso[i]
        soma_nec_atend_hr_por_recurso[recurso] = soma_nec_atend_hr_por_recurso.get(recurso, 0) + nec_atend_hr[i]
        # Atualizar o dicionário de recursos com o valor acumulado
        if recurso in dict_hor_rec:
            dict_hor_rec[recurso]['NEC_ATEND_HR'] = soma_nec_atend_hr_por_recurso[recurso]
            dict_hor_rec[recurso]['REC_HR_SALDO'] = rec_hr_saldo[i]
            
        # Agregar ID_FERRAMENTA para atualizar os dicionários, somando os valores de nec_atend_hr
        ferramenta = id_ferramenta[i]
        if ferramenta in dict_hor_fer:
            dict_hor_fer[ferramenta]['NEC_ATEND_HR'] = soma_nec_atend_hr_por_recurso.get(ferramenta, 0)
            dict_hor_fer[ferramenta]['FER_HR_SALDO'] = fer_hr_saldo[i]

    # Atribuições finais
    df['NEC_PCS'] = nec_pcs
    df['NEC_HR'] = nec_hr
    df['REC_CAP_VAR_HR'] = rec_cap_var_hr
    df['REC_HR_SALDO'] = rec_hr_saldo
    df['FER_CAP_VAR_HR'] = fer_cap_var_hr
    df['FER_HR_SALDO'] = fer_hr_saldo
    df['ROT_CAP_VAR_HR'] = rot_cap_var_hr
    df['CAP_PROD_PCS'] = cap_prod_pcs
    df['NEC_ATEND_PCS'] = nec_atend_pcs
    df['NEC_ATEND_HR'] = nec_atend_hr
    df['NEC_NAO_ATEND_PCS'] = nec_nao_atend_pcs
    df['NEC_NAO_ATEND_HR'] = nec_nao_atend_hr
    
    # df['REC_HR_ATEND'] = rec_hr_atend
    # df['REC_PCS_ATEND'] = rec_pcs_atend
    
    df_dict_hor_fer = pd.DataFrame.from_dict(dict_hor_fer, orient='index').reset_index()
    df_dict_hor_fer.columns = ['ID_FERRAMENTA', 'HOR_FER', 'NEC_ATEND_HR', 'FER_HR_SALDO']            
    
    df_dict_hor_rec = pd.DataFrame.from_dict(dict_hor_rec, orient='index').reset_index()
    df_dict_hor_rec.columns = ['ID_RECURSO', 'HOR_REC', 'NEC_ATEND_HR', 'REC_HR_SALDO']  
    
    return df, df_dict_hor_rec, df_dict_hor_fer

# --------------------------------- ### Calculando demais campos ### --------------------------------
# Calcular os campos NEC_ESTOURO_PCS, NEC_ARRASTE_PCS, %_OCUP_REC, %_OCUP_FER
def calcular_demais_campos(df):
    
    df = df.copy()
    
    # ============================================================
    # LÓGICA NEC_ESTOURO_PCS
    # ------------------------------------------------------------
    # Regra:
    # 1) Para cada COD_PROD + UNID_FAT, pegar o último valor de
    #    NEC_NAO_ATEND_PCS com base no maior IND.
    # 2) Esse valor será rateado com base em um novo percentual
    #    calculado pela média de:
    #       a) Percentual da linha em TOTAL_NECS_HR
    #       b) Percentual da linha em PCS_HORA
    # 3) TOTAL_NECS_HR = NEC_ATEND_HR + NEC_NAO_ATEND_HR
    # 4) O grupo-base do denominador será sempre:
    #       UNID_FAT + COD_PROD + "1"
    # 5) Percentual final:
    #       (PERC_TOTAL_NECS_HR + PERC_PCS_HORA) / 2
    # 6) Aplicar sobre NEC_ESTOURO_TOTAL
    # 7) Caso não exista base válida, resultado = 0
    # ============================================================

    # 1) Último NEC_NAO_ATEND_PCS por COD_PROD + UNID_FAT
    tab_nec = (
        df[['COD_PROD', 'UNID_FAT', 'IND', 'NEC_NAO_ATEND_PCS']]
        .sort_values(['COD_PROD', 'UNID_FAT', 'IND'])
        .drop_duplicates(['COD_PROD', 'UNID_FAT'], keep='last')
        .set_index(['COD_PROD', 'UNID_FAT'])['NEC_NAO_ATEND_PCS']
    )

    # 2) Mapear total a ratear (sem merge)
    df['NEC_ESTOURO_TOTAL'] = list(zip(df['COD_PROD'], df['UNID_FAT']))
    df['NEC_ESTOURO_TOTAL'] = df['NEC_ESTOURO_TOTAL'].map(tab_nec).fillna(0)

    # ============================================================
    # BASE DE CÁLCULO DO PERCENTUAL
    # ============================================================

    # 3) Criar chave de distribuição
    df['ID_DIST_ESTOURO'] = (
        df['UNID_FAT'].astype(str) + '|' +
        df['COD_PROD'].astype(str) + '|' +
        df['PRIOR_MATPAR'].astype(str)
    )

    # 4) Criar chave base fixa (PRIOR_MATPAR = 1)
    df['ID_DIST_ESTOURO_BASE_1'] = (
        df['UNID_FAT'].astype(str) + '|' +
        df['COD_PROD'].astype(str) + '|1'
    )

    # 5) TOTAL_NECS_HR
    df['TOTAL_NECS_HR'] = (
        df['NEC_ATEND_HR'].fillna(0)
        + df['NEC_NAO_ATEND_HR'].fillna(0)
    )

    # 6) Criar dicionários de soma (performático)
    soma_total_necs_hr_dict = (
        df.groupby('ID_DIST_ESTOURO', sort=False)['TOTAL_NECS_HR']
        .sum()
        .to_dict()
    )

    soma_pcs_hora_dict = (
        df.groupby('ID_DIST_ESTOURO', sort=False)['PCS_HORA']
        .sum()
        .to_dict()
    )

    # 7) Buscar somas do grupo base
    df['SOMA_TOTAL_NECS_HR_BASE_1'] = (
        df['ID_DIST_ESTOURO_BASE_1'].map(soma_total_necs_hr_dict).fillna(0)
    )

    df['SOMA_PCS_HORA_BASE_1'] = (
        df['ID_DIST_ESTOURO_BASE_1'].map(soma_pcs_hora_dict).fillna(0)
    )

    # ============================================================
    # CÁLCULO DOS PERCENTUAIS
    # ============================================================

    # 8) Inicializar percentuais
    df['PERC_TOTAL_NECS_HR'] = 0.0
    df['PERC_PCS_HORA'] = 0.0

    # 9) Percentual TOTAL_NECS_HR
    mask_necs = df['SOMA_TOTAL_NECS_HR_BASE_1'].gt(0)

    df.loc[mask_necs, 'PERC_TOTAL_NECS_HR'] = (
        df.loc[mask_necs, 'TOTAL_NECS_HR']
        / df.loc[mask_necs, 'SOMA_TOTAL_NECS_HR_BASE_1']
    )

    # 10) Percentual PCS_HORA
    mask_pcs = df['SOMA_PCS_HORA_BASE_1'].gt(0)

    df.loc[mask_pcs, 'PERC_PCS_HORA'] = (
        df.loc[mask_pcs, 'PCS_HORA']
        / df.loc[mask_pcs, 'SOMA_PCS_HORA_BASE_1']
    )

    # 11) Percentual final
    df['PERC_RATEIO_ESTOURO'] = (
        df['PERC_TOTAL_NECS_HR']
        + df['PERC_PCS_HORA']
    ) / 2

    # ============================================================
    # APLICAÇÃO DO RATEIO
    # ============================================================

    # 12) Inicializar resultado
    df['NEC_ESTOURO_PCS'] = 0.0

    # 13) Aplicar rateio
    mask_calc = (
        df['NEC_ESTOURO_TOTAL'].gt(0)
        & df['PERC_RATEIO_ESTOURO'].gt(0)
    )

    df.loc[mask_calc, 'NEC_ESTOURO_PCS'] = (
        df.loc[mask_calc, 'PERC_RATEIO_ESTOURO']
        * df.loc[mask_calc, 'NEC_ESTOURO_TOTAL']
    )

    # ============================================================
    # LIMPEZA
    # ============================================================

    df.drop(
        columns=[
            'NEC_ESTOURO_TOTAL',
            'TOTAL_NECS_HR',
            'SOMA_TOTAL_NECS_HR_BASE_1',
            'SOMA_PCS_HORA_BASE_1',
            'PERC_TOTAL_NECS_HR',
            'PERC_PCS_HORA',
            'PERC_RATEIO_ESTOURO',
            'ID_DIST_ESTOURO_BASE_1'
        ],
        inplace=True,
        errors='ignore'
    )

    # ============================================================
    # CONVERSÃO PARA HORA
    # ============================================================

    df['NEC_ESTOURO_HR'] = np.divide(
        df['NEC_ESTOURO_PCS'],
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    # ============================================================
    # LÓGICA ARRASTE (mantida igual)
    # ============================================================

    tab_NEC_ESTOURO_PCS = df[
        ['ID_PROD_UNID_FAT', 'ID_PROD_UNID_FAT_ANT', 'EST_SEG_PCS', 'NEC_ESTOURO_PCS']
    ].copy()

    tab_NEC_ESTOURO_PCS['NEC_ARRASTE_PCS'] = (
        tab_NEC_ESTOURO_PCS['NEC_ESTOURO_PCS']
        - tab_NEC_ESTOURO_PCS['EST_SEG_PCS']
    ).clip(lower=0)

    nec_arraste_dict = (
        tab_NEC_ESTOURO_PCS
        .set_index('ID_PROD_UNID_FAT')['NEC_ARRASTE_PCS']
        .to_dict()
    )

    mask = (df['PRIOR_MATPAR'] == 1) & (df['PRIOR_ROT'] == 1)

    df['NEC_ARRASTE_PCS'] = 0.0

    df.loc[mask, 'NEC_ARRASTE_PCS'] = (
        df.loc[mask, 'ID_PROD_UNID_FAT_ANT']
        .map(nec_arraste_dict)
        .fillna(0)
    )

    # ============================================================
    # LÓGICAS FINAIS (mantidas)
    # ============================================================

    df['NEC_N_ATEND_PCS_REC'] = np.maximum(
        0,
        df['NEC_PCS'] - (df['REC_CAP_VAR_HR'] * df['PCS_HORA'])
    )

    df['NEC_N_ATEND_PCS_FER'] = np.maximum(
        0,
        df['NEC_PCS'] - (df['FER_CAP_VAR_HR'] * df['PCS_HORA'])
    )
    
    # ============================================================
    # LÓGICA NEC_ESTOURO_PCS_REC
    # ------------------------------------------------------------
    # Regra:
    # 1) Para cada COD_PROD + UNID_FAT, pegar o último valor de
    #    NEC_N_ATEND_PCS_REC com base no maior IND.
    # 2) Esse valor será rateado com base em um novo percentual
    #    calculado pela média de:
    #       a) Percentual da linha em TOTAL_NECS_HR_REC
    #       b) Percentual da linha em PCS_HORA
    # 3) TOTAL_NECS_HR_REC =
    #       (NEC_ATEND_PCS + NEC_N_ATEND_PCS_REC) / PCS_HORA
    # 4) O grupo-base do denominador será sempre:
    #       UNID_FAT + COD_PROD + "1"
    # 5) Percentual final:
    #       (PERC_TOTAL_NECS_HR_REC + PERC_PCS_HORA) / 2
    # 6) Aplicar sobre NEC_ESTOURO_PCS_REC_TOTAL
    # 7) Caso não exista base válida, resultado = 0
    # ============================================================

    # 1) Último NEC_N_ATEND_PCS_REC por COD_PROD + UNID_FAT
    tab_nec_rec = (
        df[['COD_PROD', 'UNID_FAT', 'IND', 'NEC_N_ATEND_PCS_REC']]
        .sort_values(['COD_PROD', 'UNID_FAT', 'IND'])
        .drop_duplicates(['COD_PROD', 'UNID_FAT'], keep='last')
        .set_index(['COD_PROD', 'UNID_FAT'])['NEC_N_ATEND_PCS_REC']
    )

    # 2) Mapear total a ratear (sem merge)
    df['NEC_ESTOURO_PCS_REC_TOTAL'] = list(zip(df['COD_PROD'], df['UNID_FAT']))
    df['NEC_ESTOURO_PCS_REC_TOTAL'] = (
        df['NEC_ESTOURO_PCS_REC_TOTAL'].map(tab_nec_rec).fillna(0)
    )

    # ============================================================
    # BASE DE CÁLCULO DO PERCENTUAL
    # ============================================================

    # 3) Criar chave de distribuição
    df['ID_DIST_ESTOURO'] = (
        df['UNID_FAT'].astype(str) + '|' +
        df['COD_PROD'].astype(str) + '|' +
        df['PRIOR_MATPAR'].astype(str)
    )

    # 4) Criar chave base fixa (PRIOR_MATPAR = 1)
    df['ID_DIST_ESTOURO_BASE_1'] = (
        df['UNID_FAT'].astype(str) + '|' +
        df['COD_PROD'].astype(str) + '|1'
    )

    # 5) TOTAL_NECS_HR_REC
    df['TOTAL_NECS_HR_REC'] = np.divide(
        (df['NEC_ATEND_PCS'] + df['NEC_N_ATEND_PCS_REC']),
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    # 6) Criar dicionários de soma
    soma_total_necs_hr_rec_dict = (
        df.groupby('ID_DIST_ESTOURO', sort=False)['TOTAL_NECS_HR_REC']
        .sum()
        .to_dict()
    )

    soma_pcs_hora_dict = (
        df.groupby('ID_DIST_ESTOURO', sort=False)['PCS_HORA']
        .sum()
        .to_dict()
    )

    # 7) Buscar somas do grupo base
    df['SOMA_TOTAL_NECS_HR_REC_BASE_1'] = (
        df['ID_DIST_ESTOURO_BASE_1'].map(soma_total_necs_hr_rec_dict).fillna(0)
    )

    df['SOMA_PCS_HORA_BASE_1'] = (
        df['ID_DIST_ESTOURO_BASE_1'].map(soma_pcs_hora_dict).fillna(0)
    )

    # ============================================================
    # CÁLCULO DOS PERCENTUAIS
    # ============================================================

    df['PERC_TOTAL_NECS_HR_REC'] = 0.0
    df['PERC_PCS_HORA_REC'] = 0.0

    mask_necs = df['SOMA_TOTAL_NECS_HR_REC_BASE_1'].gt(0)
    df.loc[mask_necs, 'PERC_TOTAL_NECS_HR_REC'] = (
        df.loc[mask_necs, 'TOTAL_NECS_HR_REC']
        / df.loc[mask_necs, 'SOMA_TOTAL_NECS_HR_REC_BASE_1']
    )

    mask_pcs = df['SOMA_PCS_HORA_BASE_1'].gt(0)
    df.loc[mask_pcs, 'PERC_PCS_HORA_REC'] = (
        df.loc[mask_pcs, 'PCS_HORA']
        / df.loc[mask_pcs, 'SOMA_PCS_HORA_BASE_1']
    )

    # Percentual final
    df['PERC_RATEIO_ESTOURO_REC'] = (
        df['PERC_TOTAL_NECS_HR_REC']
        + df['PERC_PCS_HORA_REC']
    ) / 2

    # ============================================================
    # APLICAÇÃO DO RATEIO
    # ============================================================

    df['NEC_ESTOURO_PCS_REC'] = 0.0

    mask_calc = (
        df['NEC_ESTOURO_PCS_REC_TOTAL'].gt(0)
        & df['PERC_RATEIO_ESTOURO_REC'].gt(0)
    )

    df.loc[mask_calc, 'NEC_ESTOURO_PCS_REC'] = (
        df.loc[mask_calc, 'PERC_RATEIO_ESTOURO_REC']
        * df.loc[mask_calc, 'NEC_ESTOURO_PCS_REC_TOTAL']
    )

    # ============================================================
    # LIMPEZA
    # ============================================================

    df.drop(
        columns=[
            'NEC_ESTOURO_PCS_REC_TOTAL',
            'TOTAL_NECS_HR_REC',
            'SOMA_TOTAL_NECS_HR_REC_BASE_1',
            'SOMA_PCS_HORA_BASE_1',
            'PERC_TOTAL_NECS_HR_REC',
            'PERC_PCS_HORA_REC',
            'PERC_RATEIO_ESTOURO_REC',
            'ID_DIST_ESTOURO_BASE_1'
        ],
        inplace=True,
        errors='ignore'
    )

    # ============================================================
    # CONVERSÃO PARA HORA
    # ============================================================

    df['NEC_ESTOURO_HR_REC'] = np.divide(
        df['NEC_ESTOURO_PCS_REC'],
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    # ============================================================
    # LÓGICA NEC_ESTOURO_PCS_FER
    # ------------------------------------------------------------
    # Regra:
    # 1) Para cada COD_PROD + UNID_FAT, pegar o último valor de
    #    NEC_N_ATEND_PCS_FER com base no maior IND.
    # 2) Esse valor será rateado com base em um novo percentual
    #    calculado pela média de:
    #       a) Percentual da linha em TOTAL_NECS_HR_FER
    #       b) Percentual da linha em PCS_HORA
    # 3) TOTAL_NECS_HR_FER =
    #       (NEC_ATEND_PCS + NEC_N_ATEND_PCS_FER) / PCS_HORA
    # 4) O grupo-base do denominador será sempre:
    #       UNID_FAT + COD_PROD + "1"
    # 5) Percentual final:
    #       (PERC_TOTAL_NECS_HR_FER + PERC_PCS_HORA_FER) / 2
    # 6) Aplicar sobre NEC_ESTOURO_PCS_FER_TOTAL
    # 7) Caso não exista base válida, resultado = 0
    # ============================================================

    # 1) Último NEC_N_ATEND_PCS_FER por COD_PROD + UNID_FAT
    tab_nec_fer = (
        df[['COD_PROD', 'UNID_FAT', 'IND', 'NEC_N_ATEND_PCS_FER']]
        .sort_values(['COD_PROD', 'UNID_FAT', 'IND'])
        .drop_duplicates(['COD_PROD', 'UNID_FAT'], keep='last')
        .set_index(['COD_PROD', 'UNID_FAT'])['NEC_N_ATEND_PCS_FER']
    )

    # 2) Mapear total a ratear (sem merge)
    df['NEC_ESTOURO_PCS_FER_TOTAL'] = list(zip(df['COD_PROD'], df['UNID_FAT']))
    df['NEC_ESTOURO_PCS_FER_TOTAL'] = (
        df['NEC_ESTOURO_PCS_FER_TOTAL'].map(tab_nec_fer).fillna(0)
    )

    # ============================================================
    # BASE DE CÁLCULO DO PERCENTUAL
    # ============================================================

    # 3) Criar chave de distribuição
    df['ID_DIST_ESTOURO'] = (
        df['UNID_FAT'].astype(str) + '|' +
        df['COD_PROD'].astype(str) + '|' +
        df['PRIOR_MATPAR'].astype(str)
    )

    # 4) Criar chave base fixa (PRIOR_MATPAR = 1)
    df['ID_DIST_ESTOURO_BASE_1'] = (
        df['UNID_FAT'].astype(str) + '|' +
        df['COD_PROD'].astype(str) + '|1'
    )

    # 5) TOTAL_NECS_HR_FER
    df['TOTAL_NECS_HR_FER'] = np.divide(
        (df['NEC_ATEND_PCS'] + df['NEC_N_ATEND_PCS_FER']),
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    # 6) Criar dicionários de soma
    soma_total_necs_hr_fer_dict = (
        df.groupby('ID_DIST_ESTOURO', sort=False)['TOTAL_NECS_HR_FER']
        .sum()
        .to_dict()
    )

    soma_pcs_hora_dict = (
        df.groupby('ID_DIST_ESTOURO', sort=False)['PCS_HORA']
        .sum()
        .to_dict()
    )

    # 7) Buscar somas do grupo base
    df['SOMA_TOTAL_NECS_HR_FER_BASE_1'] = (
        df['ID_DIST_ESTOURO_BASE_1'].map(soma_total_necs_hr_fer_dict).fillna(0)
    )

    df['SOMA_PCS_HORA_BASE_1'] = (
        df['ID_DIST_ESTOURO_BASE_1'].map(soma_pcs_hora_dict).fillna(0)
    )

    # ============================================================
    # CÁLCULO DOS PERCENTUAIS
    # ============================================================

    df['PERC_TOTAL_NECS_HR_FER'] = 0.0
    df['PERC_PCS_HORA_FER'] = 0.0

    mask_necs = df['SOMA_TOTAL_NECS_HR_FER_BASE_1'].gt(0)
    df.loc[mask_necs, 'PERC_TOTAL_NECS_HR_FER'] = (
        df.loc[mask_necs, 'TOTAL_NECS_HR_FER']
        / df.loc[mask_necs, 'SOMA_TOTAL_NECS_HR_FER_BASE_1']
    )

    mask_pcs = df['SOMA_PCS_HORA_BASE_1'].gt(0)
    df.loc[mask_pcs, 'PERC_PCS_HORA_FER'] = (
        df.loc[mask_pcs, 'PCS_HORA']
        / df.loc[mask_pcs, 'SOMA_PCS_HORA_BASE_1']
    )

    # 8) Percentual final
    df['PERC_RATEIO_ESTOURO_FER'] = (
        df['PERC_TOTAL_NECS_HR_FER']
        + df['PERC_PCS_HORA_FER']
    ) / 2

    # ============================================================
    # APLICAÇÃO DO RATEIO
    # ============================================================

    # 9) Inicializar resultado
    df['NEC_ESTOURO_PCS_FER'] = 0.0

    # 10) Aplicar rateio
    mask_calc = (
        df['NEC_ESTOURO_PCS_FER_TOTAL'].gt(0)
        & df['PERC_RATEIO_ESTOURO_FER'].gt(0)
    )

    df.loc[mask_calc, 'NEC_ESTOURO_PCS_FER'] = (
        df.loc[mask_calc, 'PERC_RATEIO_ESTOURO_FER']
        * df.loc[mask_calc, 'NEC_ESTOURO_PCS_FER_TOTAL']
    )

    # ============================================================
    # LIMPEZA
    # ============================================================

    df.drop(
        columns=[
            'NEC_ESTOURO_PCS_FER_TOTAL',
            'TOTAL_NECS_HR_FER',
            'SOMA_TOTAL_NECS_HR_FER_BASE_1',
            'SOMA_PCS_HORA_BASE_1',
            'PERC_TOTAL_NECS_HR_FER',
            'PERC_PCS_HORA_FER',
            'PERC_RATEIO_ESTOURO_FER',
            'ID_DIST_ESTOURO_BASE_1'
        ],
        inplace=True,
        errors='ignore'
    )

    # ============================================================
    # CONVERSÃO PARA HORA
    # ============================================================

    df['NEC_ESTOURO_HR_FER'] = np.divide(
        df['NEC_ESTOURO_PCS_FER'],
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    # ============================================================
    # LÓGICA % OCUPAÇÃO REC
    # ------------------------------------------------------------
    # Regra:
    # Se HOR_REC <= 0 ou PCS_HORA <= 0 -> 0
    # Senão:
    #   ((NEC_ESTOURO_PCS_REC + NEC_ATEND_PCS) / PCS_HORA) / HOR_REC
    # Garantir mínimo 0
    # ============================================================

    base_ocup_rec = np.divide(
        (df['NEC_ESTOURO_PCS_REC'] + df['NEC_ATEND_PCS']),
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    df['%_OCUP_REC'] = np.divide(
        base_ocup_rec,
        df['HOR_REC'],
        out=np.zeros(len(df), dtype=float),
        where=df['HOR_REC'].to_numpy() != 0
    )

    df['%_OCUP_REC'] = np.maximum(0, df['%_OCUP_REC'])

    # ============================================================
    # LÓGICA % OCUPAÇÃO FER
    # ------------------------------------------------------------
    # Regra:
    # Se HOR_FER <= 0 ou PCS_HORA <= 0 -> 0
    # Senão:
    #   ((NEC_ESTOURO_PCS_FER + NEC_ATEND_PCS) / PCS_HORA) / HOR_FER
    # Garantir mínimo 0
    # ============================================================

    base_ocup_fer = np.divide(
        (df['NEC_ESTOURO_PCS_FER'] + df['NEC_ATEND_PCS']),
        df['PCS_HORA'],
        out=np.zeros(len(df), dtype=float),
        where=df['PCS_HORA'].to_numpy() != 0
    )

    df['%_OCUP_FER'] = np.divide(
        base_ocup_fer,
        df['HOR_FER'],
        out=np.zeros(len(df), dtype=float),
        where=df['HOR_FER'].to_numpy() != 0
    )

    df['%_OCUP_FER'] = np.maximum(0, df['%_OCUP_FER'])

    # ============================================================
    # LÓGICA HORAS OCUPADAS
    # ============================================================

    df['HR_OCUP_FER'] = df['HOR_FER'] * df['%_OCUP_FER']
    df['HR_OCUP_REC'] = df['HOR_REC'] * df['%_OCUP_REC']

    return df

# --------------------- ### Explodir Estruturas de Produção ### ---------------------
def explodir_estrutura_ltp(bd_estrut, bd_ltp):
    """
    Explode a estrutura APENAS 1 VEZ por combinação única de (COD_PROD, UNID_PROD)
    evitando duplicações.
    """

    # Padroniza
    bd_estrut = bd_estrut.copy()
    bd_ltp = bd_ltp.copy()
    
    bd_estrut["cod_prod_acabado"] = bd_estrut["cod_prod_acabado"].astype(str)
    bd_estrut["cod_insumo"] = bd_estrut["cod_insumo"].astype(str)
    bd_ltp["COD_PROD"] = bd_ltp["COD_PROD"].astype(str)

    resultados_gerais = []

    # Pré-indexação empresa -> pai -> df de filhos
    estruturas_por_empresa = {}
    for emp in bd_estrut["empresa"].unique():
        df_emp = bd_estrut[bd_estrut["empresa"] == emp].copy()
        estruturas_por_empresa[emp] = {
            pai: df_pai for pai, df_pai in df_emp.groupby("cod_prod_acabado")
        }

    # 🚀 Loop correto: uma explosão por COD_PROD + UNID_PROD
    for (prod_root, empresa), _ in bd_ltp.groupby(["COD_PROD", "UNID_PROD"]):

        # Empresa não existe na estrutura
        if empresa not in estruturas_por_empresa:
            continue

        estrutura_empresa = estruturas_por_empresa[empresa]

        # Produto não é pai
        if prod_root not in estrutura_empresa:
            continue

        resultados_local = []
        fila = [(prod_root, 0, [prod_root])]
        visitados = set()

        while fila:
            pai, nivel, trilha = fila.pop(0)

            filhos = estrutura_empresa.get(pai, None)
            if filhos is None:
                continue

            for _, row in filhos.iterrows():
                filho = row["cod_insumo"]
                qtd = row["qtd_utilizada_pcs"]

                nivel_filho = nivel + 1
                trilha_filho = trilha + [filho]

                resultados_local.append({
                    "COD_PROD": prod_root,
                    "UNID_PROD": empresa,
                    "COD_PROD_ACAB": pai,
                    "COD_INSUMO": filho,
                    "QTD_UTIL_PCS": qtd,
                    "NIVEL": nivel_filho,
                    "TRILHA": " → ".join(trilha_filho)
                })

                chave = (pai, filho)
                if chave not in visitados:
                    visitados.add(chave)
                    fila.append((filho, nivel_filho, trilha_filho))

        if resultados_local:
            resultados_gerais.append(pd.DataFrame(resultados_local))

    if resultados_gerais:
        return pd.concat(resultados_gerais, ignore_index=True)
    else:
        return pd.DataFrame()
    
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

# --------------------- ### Calcular Estoque Deduzindo Demanda Bruta ### ---------------------
# Função que explode necessidades para definir demanda dos componentes, debitando estoque para pais e filhos, uma versão de MRP com estoque global acumulativo, e com maior complexidade
def calcular_explosao_necessidades(bd_explodida, bd_ltp, lote_min_flag, multiplo_emb_flag):
    """
    Função ÚNICA que:
    1. Gera bd_estoque (com ESTOQUE_TOTAL_PCS via calc_estoque_deduzindo_demanda_bruta)
    2. Gera bd_necessidade (NEC_ATEND_PCS)
    3. Executa o MRP de forma isolada para cada COD_PROD, porém com ESTOQUE GLOBAL ACUMULATIVO.
    """

    # ----------------------------------------------------------------------
    # 1. GERA bd_estoque GLOBAL usando calc_estoque_deduzindo_demanda_bruta
    # ----------------------------------------------------------------------
    bd_estoque = calc_estoque_deduzindo_demanda_bruta(bd_ltp, lote_min_flag, multiplo_emb_flag)

    # Dicionário de estoque global ACUMULATIVO
    est_dict = {
        (str(row.UNID_FAT), str(row.COD_PROD)): row.ESTOQUE_TOTAL_PCS
        for _, row in bd_estoque.iterrows()
    }

    # ----------------------------------------------------------------------
    # 2. GERA bd_necessidade (ordem de execução do MRP)
    # ----------------------------------------------------------------------
    bd_necessidade = (
        bd_ltp.groupby(['COD_PROD', 'UNID_PROD'], as_index=False)
        ['NEC_ATEND_PCS'].sum()
    )

    # ----------------------------------------------------------------------
    # 3. PREPARA O DATAFRAME FINAL
    # ----------------------------------------------------------------------
    df = bd_explodida.merge(
        bd_necessidade,
        on=["COD_PROD", "UNID_PROD"],
        how="left"
    ).sort_values(["COD_PROD", "UNID_PROD", "NIVEL"]).reset_index(drop=True)

    # Inicializa colunas de resultado
    df["NEC_COMP_PCS"] = 0.0
    df["NEC_LIQ_PCS"] = 0.0
    df["EST_PCS_ANTES"] = 0.0
    df["EST_PCS_DEPOIS"] = 0.0
    df["DEVE_EXPLODIR"] = False

    # ----------------------------------------------------------------------
    # OTIMIZAÇÃO SEGURA:
    # Pré-cache dos índices por (COD_PROD, UNID_PROD)
    # Isso evita filtrar df para cada produto, sem alterar lógica.
    # ----------------------------------------------------------------------
    grupos = {
        (cod, emp): sub.index
        for (cod, emp), sub in df.groupby(["COD_PROD", "UNID_PROD"])
    }

    # ----------------------------------------------------------------------
    # 4. EXECUTA O MRP POR COD_PROD (ISOLADO), MAS COM ESTOQUE GLOBAL
    # ----------------------------------------------------------------------
    resultados = []

    for _, linha_nec in bd_necessidade.iterrows():
        cod = str(linha_nec["COD_PROD"])
        empresa = str(linha_nec["UNID_PROD"])
        nec_inicial = linha_nec["NEC_ATEND_PCS"]

        # Recupera o grupo pré-cacheado (mesmo efeito do seu filtro original)
        idx_cod = grupos.get((cod, empresa))
        if idx_cod is None:
            continue

        df_cod = df.loc[idx_cod].copy()

        # NEC inicial do produto raiz
        nec_dict = {(cod, empresa): nec_inicial}

        # Loop MRP — MESMA LÓGICA, NADA ALTERADO
        for i, row in df_cod.iterrows():
            pai_real = str(row.COD_PROD_ACAB)

            # NEC herdada do pai
            nec_pai = nec_dict.get((pai_real, empresa), 0)
            if nec_pai == 0:
                continue

            # Necessidade do componente
            nec_comp = nec_pai * row.QTD_UTIL_PCS
            df_cod.at[i, "NEC_COMP_PCS"] = nec_comp

            # Estoque antes
            chave_est = (empresa, str(row.COD_INSUMO))
            est_antes = est_dict.get(chave_est, 0)
            df_cod.at[i, "EST_PCS_ANTES"] = est_antes

            # Cálculo da NEC líquida + explosão
            if est_antes >= nec_comp:
                nec_liq = 0
                est_depois = est_antes - nec_comp
                deve_explodir = False
            else:
                nec_liq = nec_comp - est_antes
                est_depois = 0
                deve_explodir = True

            df_cod.at[i, "NEC_LIQ_PCS"] = nec_liq
            df_cod.at[i, "EST_PCS_DEPOIS"] = est_depois
            df_cod.at[i, "DEVE_EXPLODIR"] = deve_explodir

            # Atualiza o ESTOQUE GLOBAL ACUMULATIVO
            est_dict[chave_est] = est_depois

            # NEC para o próximo nível somente se explodir
            if deve_explodir:
                nec_dict[(str(row.COD_INSUMO), empresa)] = nec_liq

        resultados.append(df_cod)

    df_final = pd.concat(resultados, ignore_index=True)
    
    return df_final
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