"""Module for model parameters retrieval"""


def stan_init(m):
    """Retrieve parameters from a trained model.

    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for param_name in ["k", "m", "sigma_obs"]:
        res[param_name] = m.params[param_name][0][0]
    for param_name in ["delta", "beta"]:
        res[param_name] = m.params[param_name][0]
    return res
