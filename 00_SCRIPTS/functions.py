import ctypes
import time
import pandas as pd
import numpy as np

def exibir_msgbox(mensagem: str, titulo: str = "Mensagem", tipo: str = "info"):
    tipos = {
        "info": 0x40,
        "erro": 0x10,
        "alerta": 0x30,
        "simples": 0x00
    }
    estilo = tipos.get(tipo.lower(), 0x40)  # padrão = info
    ctypes.windll.user32.MessageBoxW(0, mensagem, titulo, estilo)
    
######################################################################################

def drop_colunas(df, colunas_excluir):
    return df.drop(columns=colunas_excluir, errors='ignore')

######################################################################################
class Temporizador:
    def __init__(self):
        self.inicio = None

    def iniciar(self):
        self.inicio = time.time()

    def finalizar(self):
        self.fim = time.time()
        tempo_total = self.fim - self.inicio
        minutos, segundos = divmod(tempo_total, 60)
        print(f"Tempo total de processamento: {int(minutos)} min {segundos:.1f} s")
    
    def imprimir(self):
        tempo_total = self.fim - self.inicio
        minutos, segundos = divmod(tempo_total, 60)
        print(f"Tempo total de processamento: {int(minutos)} min {segundos:.1f} s")
        
######################################################################################
def criar_indice_incremental(df, coluna_contar='ID_RECURSO', nome_nova_coluna='ID_NUM_REC'):
    """
    Cria uma coluna com índice incremental reiniciado para cada valor sequencial de ID_RECURSO,
    após ordenar crescentemente pela coluna de recurso. Não utiliza groupby.

    Parâmetros:
    - df: DataFrame original (modificado internamente).
    - coluna_contar: nome da coluna a ser usada como chave de agrupamento (default: 'ID_RECURSO').
    - nome_nova_coluna: nome da nova coluna criada (default: 'ID_NUM_REC').

    Retorna:
    - df ordenado e com a nova coluna adicionada.
    """

    # Ordena crescentemente pela coluna de recurso
    df = df.sort_values(by=[coluna_contar]).reset_index(drop=True)

    # Transforma a coluna em array
    id_array = df[coluna_contar].to_numpy()

    # Detecta mudanças
    mudou = np.empty_like(id_array, dtype=bool)
    mudou[0] = True
    mudou[1:] = id_array[1:] != id_array[:-1]

    # Cria grupos e contadores
    grupos = np.cumsum(mudou)
    contadores = np.zeros_like(grupos, dtype=int)

    idx = 0
    for g in np.unique(grupos):
        mask = grupos == g
        contadores[mask] = np.arange(1, mask.sum() + 1)
        idx += mask.sum()

    # Atribui ao DataFrame
    df[nome_nova_coluna] = contadores
    return df

######################################################################################
def parar_execucao():
    assert False, "Execução interrompida DEBUG."

######################################################################################
def filtrar(df, coluna, valor):
    return df[df[coluna].astype(str).str.strip().str.contains(valor, na=False, regex=False)]

# Exemplo de uso:
# bd_LTP_NEC_calculos = filtrar(bd_LTP_NEC_calculos, "ID_FERRAMENTA", "JUN25|ACESSÓRIOS|0727A|ACE")

######################################################################################