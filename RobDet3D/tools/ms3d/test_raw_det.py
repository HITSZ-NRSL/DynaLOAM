import os
from pathlib import Path

output_dir = Path("tools/ms3d/data/good")

#os.system = print


def predict_all():
    results_dir = output_dir / experiment
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_arg in models:
        for tta_arg in tta:
            model_name, eval_tag = Path(model_arg.split()[1]).stem, tta_arg.split()[1]
            os.system(f"{launcher} tools/test.py {model_arg} {tta_arg} --output {workspace} {args}")

            pred_dicts_result = f"{workspace}/custom/{model_name}/default/test/{eval_tag}/eval/epoch_0/result.pkl"
            os.system(f"cp {pred_dicts_result} {results_dir}/{model_name}_{eval_tag}.pkl")


from configs.config_1f import *

predict_all()

# from configs.config_16f import *
#
# predict_all()
