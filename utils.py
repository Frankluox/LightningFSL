
def preserve_key(state, preserve_prefix: str):
    """Preserve part of model weights based on the
       prefix of the preserved module name.
    """
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if preserve_prefix + "." in key:
            newkey = key.replace(preserve_prefix + '.', "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    return state