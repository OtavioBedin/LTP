import numpy as np
import pandas as pd
from collections import defaultdict, deque

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
