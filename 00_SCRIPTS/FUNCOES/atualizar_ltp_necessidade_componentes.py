import numpy as np
import pandas as pd
from collections import defaultdict, deque

# *************************# Retorno da NEC_COMP_PCS para a estrutura LTP #********************************
def atualizar_ltp_comp_nec_pcs(bd_LTP, bd_nec_comp_expl):

        # Agregar campos UNID_PROD, COD_INSUMO, NEC_LIQ_PCS
        bd_nec_comp_expl_agreg = (
            bd_nec_comp_expl
            .groupby(["UNID_PROD", "COD_INSUMO"], as_index=False)["NEC_LIQ_PCS"]
            .sum()
        )
        
        # Faz o merge entre bd_LTP e bd_nec_comp_expl_agreg
        bd_LTP = bd_LTP.merge(
            bd_nec_comp_expl_agreg,
            left_on=["UNID_FAT", "COD_PROD"],
            right_on=["UNID_PROD", "COD_INSUMO"],
            how="left",
            suffixes=("", "_eliminar")
        )

        # Atualiza o campo LTP_COMP_NEC_PCS com NEC_LIQ_PCS quando existir
        bd_LTP["LTP_COMP_NEC_PCS"] = bd_LTP["NEC_LIQ_PCS"].fillna(bd_LTP["LTP_COMP_NEC_PCS"])

        # Remove coluna auxiliar COD_INSUMO se não precisar mais
        bd_LTP = bd_LTP.drop(columns=["COD_INSUMO", "NEC_LIQ_PCS", "UNID_PROD_eliminar"])

        return bd_LTP
