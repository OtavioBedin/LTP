# Gera Resumo de tempos por função do motor de cortes, para ajudar a identificar gargalos e otimizar o código.

import time
from functools import wraps


PERF_ATIVO = False
PERF_TEMPOS = {}
PERF_CONTAGENS = {}
PERF_WHILE_ITERACOES = 0
PERF_T0_TOTAL = None


def configurar_performance(ativo=False):
    global PERF_ATIVO, PERF_TEMPOS, PERF_CONTAGENS, PERF_WHILE_ITERACOES, PERF_T0_TOTAL

    PERF_ATIVO = ativo
    PERF_TEMPOS = {}
    PERF_CONTAGENS = {}
    PERF_WHILE_ITERACOES = 0
    PERF_T0_TOTAL = time.perf_counter() if ativo else None


def aplicar_wrappers(globals_ref, nomes_funcoes):
    if not PERF_ATIVO:
        return

    for nome_func in nomes_funcoes:
        if nome_func not in globals_ref:
            continue

        func = globals_ref[nome_func]

        if getattr(func, "_perf_wrapped", False):
            continue

        @wraps(func)
        def wrapper(*args, __func=func, __nome_func=nome_func, **kwargs):
            t0 = time.perf_counter()
            try:
                return __func(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                PERF_TEMPOS[__nome_func] = PERF_TEMPOS.get(__nome_func, 0.0) + dt
                PERF_CONTAGENS[__nome_func] = PERF_CONTAGENS.get(__nome_func, 0) + 1

        wrapper._perf_wrapped = True
        globals_ref[nome_func] = wrapper


def contar_iteracao():
    global PERF_WHILE_ITERACOES

    if PERF_ATIVO:
        PERF_WHILE_ITERACOES += 1


def imprimir_resumo():
    if not PERF_ATIVO:
        return

    total_exec = time.perf_counter() - PERF_T0_TOTAL if PERF_T0_TOTAL else 0.0
    total_medido = sum(PERF_TEMPOS.values())

    print("\n" + "=" * 90)
    print("PERF - RESUMO DO MOTOR DE CORTES")
    print("=" * 90)
    print(f"TEMPO TOTAL EXECUCAO: {total_exec:.2f}s")
    print(f"ITERACOES WHILE: {PERF_WHILE_ITERACOES}")
    print(f"TEMPO MEDIDO EM FUNCOES: {total_medido:.2f}s")
    print("-" * 90)

    for nome, tempo in sorted(PERF_TEMPOS.items(), key=lambda x: -x[1]):
        qtd = PERF_CONTAGENS.get(nome, 0)
        media = tempo / qtd if qtd else 0.0
        pct_total = (tempo / total_exec * 100.0) if total_exec else 0.0

        print(
            f"{nome:<50} "
            f"{tempo:>10.2f}s  "
            f"{pct_total:>6.1f}%  "
            f"{qtd:>6}x  "
            f"media={media:.4f}s"
        )

    print("=" * 90)
    
    
FUNCOES_PADRAO_MOTOR_CORTES = [
    "calc_nec_pcs_hr",
    "calcular_distrib_capacidade",
    "calcular_demais_campos",
    "explodir_estrutura_ltp",
    "calcular_explosao_necessidades",
    "atualizar_ltp_comp_nec_pcs",
    "calcular_decomposicao_nec_pcs_para_precisao_corte",
    "matriz_logica_cortes_horas",
    "cria_bd_mat_cortes_REC",
    "cria_bd_mat_cortes_FER",
]


def monit_performance_funcoes(globals_ref, ativo=False):
    configurar_performance(ativo=ativo)
    aplicar_wrappers(globals_ref, FUNCOES_PADRAO_MOTOR_CORTES)