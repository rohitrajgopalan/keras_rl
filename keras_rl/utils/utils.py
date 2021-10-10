def get_hidden_layer_sizes(fc_dims):
    if type(fc_dims) == int:
        return fc_dims, fc_dims
    elif type(fc_dims) in [list, tuple]:
        return fc_dims[0], fc_dims[1]
    else:
        raise TypeError('fc_dims should be integer, list or tuple')
