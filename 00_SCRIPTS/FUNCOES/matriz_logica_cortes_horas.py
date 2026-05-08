import numpy as np
import pandas as pd
from collections import defaultdict, deque

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
