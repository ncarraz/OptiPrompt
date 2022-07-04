from .maskedlm_connector import MaskedLM
from .causallm_connector import CausalLM
from .base_connector import LM_TYPE

def build_model_by_name(args, verbose=True):
    """Load a model by name.

    """
    lm = args.model_name
    if lm not in LM_TYPE:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    model_type = LM_TYPE[lm]

    MODEL_NAME_TO_CLASS = dict(
        masked=MaskedLM,
        seq2seq=MaskedLM, # interface is the same as maskedlm
        causal=CausalLM,
    )
    if verbose:
        print("Loading %s model..." % lm)
    return MODEL_NAME_TO_CLASS[model_type](args)