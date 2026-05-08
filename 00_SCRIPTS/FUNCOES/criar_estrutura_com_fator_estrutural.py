import numpy as np
import pandas as pd
from collections import defaultdict, deque

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
