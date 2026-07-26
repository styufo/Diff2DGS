import runpy


def load_config(path):
    return {
        key: value
        for key, value in runpy.run_path(path).items()
        if not key.startswith("__")
    }


def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args
