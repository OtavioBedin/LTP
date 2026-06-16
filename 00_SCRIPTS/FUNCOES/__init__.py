import importlib


MODULOS = [
    "perf_motor",
    "necessidade_pecas_horas",
    "necessidade_pecas_horas_pos_eco",
    "distribuicao_capacidade",
    "demais_campos_estouro_ocupacao",
    "explosao_estrutura_ltp",
    "estoque_deduzido_demanda_bruta",
    "explosao_necessidades_componentes",
    "criar_estrutura_com_fator_estrutural",
    "matriz_cortes_recurso",
    "matriz_cortes_ferramenta",
    "matriz_logica_cortes_horas",
    "atualizar_ltp_necessidade_componentes",
    "decomposicao_nec_pcs_precisao_corte",
]


def _importar_modulo(nome_modulo):
    return importlib.import_module(f"{__name__}.{nome_modulo}")


def reload_all():
    modulos_carregados = {}

    for nome_modulo in MODULOS:
        modulo = _importar_modulo(nome_modulo)
        modulo = importlib.reload(modulo)
        modulos_carregados[nome_modulo] = modulo

    _exportar_funcoes(modulos_carregados)


def _exportar_funcoes(modulos=None):
    if modulos is None:
        modulos = {nome: _importar_modulo(nome) for nome in MODULOS}

    globals()["monit_performance_funcoes"] = modulos["perf_motor"].monit_performance_funcoes
    globals()["contar_iteracao"] = modulos["perf_motor"].contar_iteracao
    globals()["imprimir_resumo"] = modulos["perf_motor"].imprimir_resumo
    globals()["calc_nec_pcs_hr"] = modulos["necessidade_pecas_horas"].calc_nec_pcs_hr
    globals()["calc_nec_pcs_hr_pos_eco"] = modulos["necessidade_pecas_horas_pos_eco"].calc_nec_pcs_hr_pos_eco
    globals()["calcular_distrib_capacidade"] = modulos["distribuicao_capacidade"].calcular_distrib_capacidade
    globals()["calcular_demais_campos"] = modulos["demais_campos_estouro_ocupacao"].calcular_demais_campos
    globals()["explodir_estrutura_ltp"] = modulos["explosao_estrutura_ltp"].explodir_estrutura_ltp
    globals()["calc_estoque_deduzindo_demanda_bruta"] = modulos["estoque_deduzido_demanda_bruta"].calc_estoque_deduzindo_demanda_bruta
    globals()["calcular_explosao_necessidades"] = modulos["explosao_necessidades_componentes"].calcular_explosao_necessidades
    globals()["criar_estrutura_com_fator_estrutural"] = modulos["criar_estrutura_com_fator_estrutural"].criar_estrutura_com_fator_estrutural
    globals()["cria_bd_mat_cortes_REC"] = modulos["matriz_cortes_recurso"].cria_bd_mat_cortes_REC
    globals()["cria_bd_mat_cortes_FER"] = modulos["matriz_cortes_ferramenta"].cria_bd_mat_cortes_FER
    globals()["matriz_logica_cortes_horas"] = modulos["matriz_logica_cortes_horas"].matriz_logica_cortes_horas
    globals()["atualizar_ltp_comp_nec_pcs"] = modulos["atualizar_ltp_necessidade_componentes"].atualizar_ltp_comp_nec_pcs
    globals()["calcular_decomposicao_nec_pcs_para_precisao_corte"] = modulos["decomposicao_nec_pcs_precisao_corte"].calcular_decomposicao_nec_pcs_para_precisao_corte

_exportar_funcoes()


__all__ = [
    "reload_all",
    "monit_performance_funcoes",
    "contar_iteracao",
    "imprimir_resumo",
    "calc_nec_pcs_hr",
    "calc_nec_pcs_hr_pos_eco",
    "calcular_distrib_capacidade",
    "calcular_demais_campos",
    "explodir_estrutura_ltp",
    "calc_estoque_deduzindo_demanda_bruta",
    "calcular_explosao_necessidades",
    "criar_estrutura_com_fator_estrutural",
    "cria_bd_mat_cortes_REC",
    "cria_bd_mat_cortes_FER",
    "matriz_logica_cortes_horas",
    "atualizar_ltp_comp_nec_pcs",
    "calcular_decomposicao_nec_pcs_para_precisao_corte",
]