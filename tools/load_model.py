import mxnet as mx


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def load_param(prefix, epoch, convert=False, context=None):
    """
    wrapper for load checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :param convert: reference model should be converted to GPU NDArray first
    :param context: if convert then context must be designated.
    :return: (arg_params, aux_params)
    """
    def convert_to_gpu(array_dict):
        result_dict = dict()
        for name, nd in array_dict.items():
            result_dict[name] = nd.as_in_context(context)
        return result_dict

    arg_params, aux_params = load_checkpoint(prefix, epoch)
    if convert:
        assert context is not None
        arg_params = convert_to_gpu(arg_params)
        aux_params = convert_to_gpu(aux_params)

    return arg_params, aux_params
