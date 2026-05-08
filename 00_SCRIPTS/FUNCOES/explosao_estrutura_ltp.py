import numpy as np
import pandas as pd
from collections import defaultdict, deque

# --------------------- ### Explodir Estruturas de Produção ### ---------------------
def explodir_estrutura_ltp(bd_estrut, bd_ltp):
    """
    Explode a estrutura apenas 1 vez por combinação única de (COD_PROD, UNID_PROD).
    """

    cols_saida = [
        "COD_PROD",
        "UNID_PROD",
        "COD_PROD_ACAB",
        "COD_INSUMO",
        "QTD_UTIL_PCS",
        "NIVEL",
        "TRILHA",
    ]

    if bd_estrut.empty or bd_ltp.empty:
        return pd.DataFrame(columns=cols_saida)

    # Apenas as colunas necessárias
    estrut = bd_estrut[
        ["empresa", "cod_prod_acabado", "cod_insumo", "qtd_utilizada_pcs"]
    ]

    ltp = bd_ltp[["COD_PROD", "UNID_PROD"]]

    # Conversões mínimas, mantendo a regra original
    cod_pai = estrut["cod_prod_acabado"].astype(str).to_numpy()
    cod_filho = estrut["cod_insumo"].astype(str).to_numpy()
    empresa_arr = estrut["empresa"].to_numpy()
    qtd_arr = estrut["qtd_utilizada_pcs"].to_numpy()

    # Índice rápido:
    # estrutura_idx[empresa][pai] = [(filho, qtd), ...]
    estrutura_idx = defaultdict(lambda: defaultdict(list))

    for emp, pai, filho, qtd in zip(empresa_arr, cod_pai, cod_filho, qtd_arr):
        estrutura_idx[emp][pai].append((filho, qtd))

    # Combinações únicas, preservando ordem de aparição
    pares_ltp = (
        ltp.assign(COD_PROD=ltp["COD_PROD"].astype(str))
           .drop_duplicates(["COD_PROD", "UNID_PROD"], ignore_index=True)
    )

    resultados = []

    for prod_root, empresa in pares_ltp.itertuples(index=False, name=None):

        estrutura_empresa = estrutura_idx.get(empresa)
        if not estrutura_empresa:
            continue

        if prod_root not in estrutura_empresa:
            continue

        fila = deque()
        fila.append((prod_root, 0, prod_root))

        # Mantém a mesma lógica da função original:
        # evita reprocessar a mesma aresta pai -> filho
        visitados = set()

        while fila:
            pai, nivel, trilha = fila.popleft()

            filhos = estrutura_empresa.get(pai)
            if not filhos:
                continue

            nivel_filho = nivel + 1

            for filho, qtd in filhos:
                trilha_filho = f"{trilha} → {filho}"

                resultados.append((
                    prod_root,
                    empresa,
                    pai,
                    filho,
                    qtd,
                    nivel_filho,
                    trilha_filho,
                ))

                chave = (pai, filho)

                if chave not in visitados:
                    visitados.add(chave)
                    fila.append((filho, nivel_filho, trilha_filho))

    return pd.DataFrame.from_records(resultados, columns=cols_saida)
