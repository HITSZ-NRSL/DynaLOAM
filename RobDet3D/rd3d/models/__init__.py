from .detectors import build_detector


def build_network(model_cfg, num_class, dataset, logger=None):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger
    )
    return model
