version 0 -> old top1
	datalake_v1 		: antiguo, le faltan algunas temporadas (experimentos originales sin Weekday)
	page_rank		: antiguo, solo tiene dos columnas y con lags-deltas indeterminados
	match_importance_old	: antiguo, agregacion con mean pero erronea (no se hace shift). 
	
version 0.1 --> (0.58)
	datalake_v1 		: antiguo, le faltan algunas temporadas (experimentos originales sin Weekday)
	page_rank_v0.1		: antiguo, solo tiene dos columnas y lags-deltas de 2 años como paper, hecho sobre datalake_v1
	match_importance_old	: antiguo, agregacion con mean pero erronea (no se hace shift). 

version 0.2 
	datalake_v1 		: antiguo, le faltan algunas temporadas (experimentos originales sin Weekday)
	page_rank		: antiguo, solo tiene dos columnas y con lags-deltas indeterminados
	match_importance_v0.2	: antiguo, agregacion con mean y shift.

exp_	data_v1
version 0.3 (0.71)
	datalake_v1 		: antiguo, le faltan algunas temporadas (experimentos originales sin Weekday)
	page_rank		: antiguo, solo tiene dos columnas y con lags-deltas indeterminados
	match_importance_v0.3	: antiguo, agregacion con mean y SIN shift (a ver si hay diferencia con v0)

version 1 -> top1 corregido (0.50)
	datalake 		: antiguo, le faltan algunas temporadas, ademas Weekday añadido (experimentos originales sin Weekday)
	page_rank_v1		: antiguo, solo tiene dos columnas y lags-deltas de 2 años como paper
	match_importance_v1	: antiguo, agregacion con mean y shift. 

version 1.3 -> old con nuevo datalake (0.65)
	datalake 		: antiguo, le faltan algunas temporadas, ademas Weekday añadido (experimentos originales sin Weekday)
	page_rank_v1		: antiguo, solo tiene dos columnas y lags-deltas de 2 años como paper
	match_importance_old_v1	: antiguo, agregacion con mean y SIN shift. 

version 2 -> top1 mejorado
	datalake 		: nuevo, tiene todas las temporadas y Weekday
	page_rank_v2		: nuevo, con diferentes lags de page_rank
	match_importance_v2	: nuevo, agregacion con Sum y shift. 
	
