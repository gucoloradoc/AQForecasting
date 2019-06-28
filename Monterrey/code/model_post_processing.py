def data_reescaling(data_scaled, p, norm_param, norm_guide):
    """ Scales the pollutant according with the norm guide, retrieving the parameters from the
    norm_param and norm_guide dictionaries.
    """
    from numpy import exp
    if norm_guide[p][0]=='log':
        data_rescld=exp(data_scaled)-norm_guide[p][1]
    elif norm_guide[p][0]=='mean':
        data_rescld=data_scaled*norm_param[p][1]+norm_param[p][0]
    elif norm_guide[p][0]=='max':
        data_rescld=data_scaled*(norm_param[p][1]-norm_param[p][0])+norm_param[p][0]
    else:
        data_rescld=data_scaled-norm_guide[p][1]
    
    return data_rescld
