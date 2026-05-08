import numpy as np
import pandas as pd
from collections import defaultdict, deque

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
