import numpy as np
import pandas as pd

def migrar_id_pri_priori_pos_corte(
    df,
    inds_cortados,
    col_ind="IND",
    col_id="ID_PROD_UNID_FAT",
    col_id_pri="ID_PRI_PRIORI",
    col_ordem="IND",
):
    """
    Migra o ID_PRI_PRIORI das linhas já cortadas para a próxima linha ainda não cortada.

    Regra:
    - Linha cortada não pode continuar com ID_PRI_PRIORI.
    - Apaga ID_PRI_PRIORI da linha cortada.
    - A próxima linha disponível recebe:
        ID_PRI_PRIORI = ID_PROD_UNID_FAT dela mesma.
    - Se não existir próxima linha disponível, ninguém recebe o marcador.
    """

    # Sem IND cortado, não tem nada para ajustar.
    if not inds_cortados:
        return df

    df = df.copy()
    inds_cortados = set(inds_cortados)

    # Ordena uma única vez para definir a sequência da próxima linha.
    df_ord = df.sort_values(
        by=col_ordem,
        ascending=True,
        kind="mergesort"
    )

    # Arrays da base ordenada.
    idx_ord = df_ord.index.to_numpy()
    ind_ord = df_ord[col_ind].to_numpy()
    id_ord = df_ord[col_id].to_numpy()

    # Marca quais posições da base ordenada já foram cortadas.
    mask_cortado_ord = np.isin(ind_ord, list(inds_cortados))

    # Identifica linhas que ainda têm ID_PRI_PRIORI preenchido.
    pri = df[col_id_pri]

    mask_pri_ativa = (
        pri.notna()
        & pri.astype(str).str.strip().ne("")
    )

    # Entre as linhas com marcador ativo, pega somente as que já foram cortadas.
    mask_pri_ativa_cortada = (
        mask_pri_ativa
        & df[col_ind].isin(inds_cortados)
    )

    # Se nenhuma prioridade ativa foi cortada, mantém tudo como está.
    if not mask_pri_ativa_cortada.any():
        return df

    # Índices reais das linhas cortadas que precisam perder o marcador.
    idxs_ativas_cortadas = df.index[mask_pri_ativa_cortada].to_numpy()

    # Mapeia índice real do DataFrame para posição dentro da base ordenada.
    pos_por_idx = pd.Series(
        np.arange(len(idx_ord)),
        index=idx_ord
    )

    # Posições ordenadas das linhas cortadas com marcador ativo.
    pos_ativas = pos_por_idx.loc[idxs_ativas_cortadas].to_numpy()

    # Apaga o ID_PRI_PRIORI das linhas que já foram cortadas.
    df.loc[idxs_ativas_cortadas, col_id_pri] = np.nan

    # Cria um array dizendo, para cada posição, qual é a próxima posição livre depois dela.
    n = len(df_ord)
    next_free_pos = np.full(n, -1, dtype=np.int64)

    proxima_livre = -1

    # Percorre de trás para frente para preencher a próxima posição livre.
    for pos in range(n - 1, -1, -1):
        next_free_pos[pos] = proxima_livre

        if not mask_cortado_ord[pos]:
            proxima_livre = pos

    # Para cada linha cortada, encontra a próxima posição livre.
    pos_proximas = next_free_pos[pos_ativas]

    # Remove casos em que não existe próxima linha livre.
    pos_proximas = pos_proximas[pos_proximas >= 0]

    if len(pos_proximas) == 0:
        return df

    # Evita marcar a mesma próxima linha mais de uma vez.
    pos_proximas = np.unique(pos_proximas)

    # Índices e IDs das próximas linhas disponíveis.
    idxs_proximas = idx_ord[pos_proximas]
    ids_proximas = id_ord[pos_proximas]

    # Coloca o marcador na próxima linha disponível.
    df.loc[idxs_proximas, col_id_pri] = ids_proximas

    return df