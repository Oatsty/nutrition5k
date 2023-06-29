from .registry import is_model, model_entrypoints


def build_model(config, **kwargs):
    model_name = config["MODEL"]["NAME"]

    if not is_model(model_name):
        raise ValueError(f"Unkown model: {model_name}")

    return model_entrypoints(model_name)(config, **kwargs)
