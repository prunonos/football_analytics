import utils as ut

experiments = ['lgbm_top1_rs',
 'own_original_comp_bal_fac_ss',
 'own_original_simp_fac_ss',
 'own_top1_adv_comp_fac_ss_miv2',
 'tabnet_owntop1_comp_ss_sup',
 'tabnet_own_simp_ss_sup',
 'tabnet_paper_ss_unsup',
 'tabnet_own_comp_rs_unsup_rps']

# ut.make_resume([r'(\w+)'],title='resume_v2.2_rps',save=1).columns
ut.make_resume(experiments,title='deep_analysis',save=1).columns