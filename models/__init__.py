
def get_model(hps):
    if hps.model == 'im_acset_basic':
        from .im_acset import ACSetBasic
        model = ACSetBasic(hps)
    elif hps.model == 'im_acset_flow':
        from .im_acset import ACSetFlow
        model = ACSetFlow(hps)
    elif hps.model == 'imfn_acset_flow':
        from .imfn_acset import ACSetFlow
        model = ACSetFlow(hps)
    elif hps.model == 'imfn_acset_flow_res':
        from .imfn_acset import ACSetFlowRes
        model = ACSetFlowRes(hps)
    elif hps.model == 'imfn_acset_flow_g':
        from .imfn_acset import ACSetFlowG
        model = ACSetFlowG(hps)
    elif hps.model == 'imfn_acset_flow_resg':
        from .imfn_acset import ACSetFlowResG
        model = ACSetFlowResG(hps)
    elif hps.model == 'im_acset_xformer':
        from .im_acset import ACSetXformer
        model = ACSetXformer(hps)
    elif hps.model == 'im_acidp':
        from .im_acidp import ACIdp
        model = ACIdp(hps)
    elif hps.model == 'im_acidp_g':
        from . im_acidp import ACIdpG
        model = ACIdpG(hps)
    elif hps.model == 'pc_acset_flow':
        from .pc_acset import ACSetFlow
        model = ACSetFlow(hps)
    elif hps.model == 'pcfn_acset_flow':
        from .pcfn_acset import ACSetFlow
        model = ACSetFlow(hps)
    elif hps.model == 'ft_acset_basic':
        from .ft_acset import ACSetBasic
        model = ACSetBasic(hps)
    elif hps.model == 'ft_acset_flow':
        from .ft_acset import ACSetFlow
        model = ACSetFlow(hps)
    elif hps.model == 'fns_acidp':
        from .fns_acidp import ACIdp
        model = ACIdp(hps)
    elif hps.model == 'fns_acset':
        from .fns_acset import ACSetFlow
        model = ACSetFlow(hps)
    else:
        raise ValueError()

    return model
