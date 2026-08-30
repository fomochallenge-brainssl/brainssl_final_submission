"""
    Split encoder and downstream task head parameters into two different groups
"""
def build_param_groups(model, encoder_lr_ratio, lr, weight_decay):
    # split model's paramters into groups
    encoder_params = list(model.encoder.parameters())
    encoder_pars_ids = {id(p) for p in encoder_params}
    # everything that is not encoder is counted as head params and optimized as such
    head_params = [p for p in model.parameters() if id(p) not in encoder_pars_ids]

    # assign hyper-paramters to groups
    hyperparms_groups_mapping = [
        {"params": encoder_params, "lr": lr * encoder_lr_ratio, "weight_decay": weight_decay, "name":"encoder"},
        {"params": head_params, "lr": lr, "weight_decay": weight_decay, "name":"head"},
    ]

    return hyperparms_groups_mapping