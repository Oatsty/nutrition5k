from .registry import is_model, model_entrypoints


def build_decoder(config, *args, **kwargs):
    model_name = config["MODEL"]["DECODER"]["NAME"]

    if not is_model(model_name):
        raise ValueError(f"Unkown model: {model_name}")

    return model_entrypoints(model_name)(config, *args, **kwargs)
