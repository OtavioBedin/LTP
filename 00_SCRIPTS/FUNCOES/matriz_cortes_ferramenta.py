import numpy as np
import pandas as pd
from collections import defaultdict, deque

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
