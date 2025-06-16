import os
import json
from pathlib import Path

from project.eval import (
    eval_forget,
    eval_neighbor,
    eval_mmlu,
    eval_bbh,
    eval_truthfulqa,
    eval_triviaqa,
    eval_fluency,
    eval_mia,
)
from project.utils.logging import get_logger

log = get_logger()

FORGET_LEVEL1 = "forget_level1.json"
FORGET_LEVEL2 = "forget_level2.json"
FORGET_LEVEL3 = "forget_level3.json"
NEIGHBOR_LEVEL1 = "neighbor_level1.json"
NEIGHBOR_LEVEL2 = "neighbor_level2.json"

RETAIN_MMLU = "retain_mmlu.json"
RETAIN_BBH = "retain_bbh.json"
TRUTHFUL = "truthful.json"
TRIVIAQA = "triviaqa.json"
FLUENCY = "fluency.json"
FORGET_MIA = "forget_mia.json"
RETAIN_MIA = "retain_mia.json"

data_root = Path(__file__).parent.parent.parent / "data/rwku/benchmark/Target"


def eval_llm(
    pre_trained_llm,
    pre_trained_llm_tokenizer,
    target_id: str,
    device,
    stage_number: int,
    use_prompt: bool = False,
):
    """
    Evaluate the model on the RWKU benchmark.

    Args:
        model: The model to evaluate
        model_name_or_path: Path to the model or model name for loading the tokenizer
        target: Target entity to evaluate
        output_result_dir: Directory to save results
        device: Device to run the evaluation on
        use_prompt: Whether to use the unlearning prompt
    """

    model = pre_trained_llm.to(device)
    tokenizer = pre_trained_llm_tokenizer

    # Constants for file paths
    FORGET_LEVEL1 = "forget_level1.json"
    FORGET_LEVEL2 = "forget_level2.json"
    FORGET_LEVEL3 = "forget_level3.json"
    NEIGHBOR_LEVEL1 = "neighbor_level1.json"
    NEIGHBOR_LEVEL2 = "neighbor_level2.json"

    # Set the evaluation dataset directory for the target
    eval_dataset_dir = data_root / target_id

    # Load evaluation datasets
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL1), "r") as f:
        forget_level1 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL2), "r") as f:
        forget_level2 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL3), "r") as f:
        forget_level3 = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL1), "r") as f:
        neighbor_level1 = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL2), "r") as f:
        neighbor_level2 = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_MMLU), "r") as f:
        retain_mmlu = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_BBH), "r") as f:
        retain_bbh = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRUTHFUL), "r") as f:
        truthfulqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRIVIAQA), "r") as f:
        triviaqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_MIA), "r") as f:
        forget_mia = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_MIA), "r") as f:
        retain_mia = json.load(f)
    with open(os.path.join(eval_dataset_dir, FLUENCY), "r") as f:
        fluency = json.load(f)

    log.info("Evaluate forgetting...")
    forget_fb, forget_qa, forget_aa = eval_forget(
        model,
        tokenizer,
        forget_level1,
        forget_level2,
        forget_level3,
        batch_size=16,
        output_result_dir=None,
        use_prompt=use_prompt,
    )
    log.info("Evaluate neighbor...")
    neighbor_fb, neighbor_qa = eval_neighbor(
        model,
        tokenizer,
        neighbor_level1,
        neighbor_level2,
        batch_size=16,
        output_result_dir=None,
        use_prompt=use_prompt,
    )
    log.info("Evaluate mmlu...")
    utility_gen = eval_mmlu(
        model,
        tokenizer,
        retain_mmlu,
        batch_size=1,
        output_result_dir=None,
        use_prompt=use_prompt,
    )
    log.info("Evaluate bbh...")
    utility_rea = eval_bbh(
        model,
        tokenizer,
        retain_bbh,
        batch_size=8,
        output_result_dir=None,
        use_prompt=use_prompt,
    )
    log.info("Evaluate truthful...")
    utility_tru, _ = eval_truthfulqa(
        model,
        tokenizer,
        truthfulqa,
        batch_size=4,
        output_result_dir=None,
        use_prompt=use_prompt,
    )
    log.info("Evaluate triviaqa...")
    _, utility_fac = eval_triviaqa(
        model,
        tokenizer,
        triviaqa,
        batch_size=16,
        output_result_dir=None,
        use_prompt=use_prompt,
    )
    log.info("Evaluate forget mia...")
    mia_fm, _, _ = eval_mia(
        model, tokenizer, forget_mia, output_result_dir=None, use_prompt=use_prompt
    )
    log.info("Evaluate retain mia...")
    mia_rt, _, _ = eval_mia(
        model, tokenizer, retain_mia, output_result_dir=None, use_prompt=use_prompt
    )
    log.info("Evaluate fluency...")
    utility_flu = eval_fluency(
        model,
        tokenizer,
        fluency,
        batch_size=8,
        output_result_dir=None,
        use_prompt=use_prompt,
    )

    results = {
        "eval/forget/fb": forget_fb,
        "eval/forget/qa": forget_qa,
        "eval/forget/aa": forget_aa,
        "eval/neighbor/fb": neighbor_fb,
        "eval/neighbor/qa": neighbor_qa,
        "eval/mia/fm": mia_fm,
        "eval/mia/rt": mia_rt,
        "eval/utility/gen": utility_gen,
        "eval/utility/rea": utility_rea,
        "eval/utility/tru": utility_tru,
        "eval/utility/fac": utility_fac,
        "eval/utility/flu": utility_flu,
        "eval/stage_number": stage_number,
        "report/forget/fb": forget_fb * 100,
        "report/forget/qa": forget_qa * 100,
        "report/forget/aa": forget_aa * 100,
        "report/forget/all": (forget_fb * 3268 + forget_qa * 2879 + forget_aa * 6984)
        / (3268 + 2879 + 6984)
        * 100,
        "report/neighbor/fb": neighbor_fb * 100,
        "report/neighbor/qa": neighbor_qa * 100,
        "report/neighbor/all": (neighbor_fb * 5846 + neighbor_qa * 5533)
        / (5846 + 5533)
        * 100,
        "report/mia/fm": mia_fm * -100,
        "report/mia/rt": mia_rt * -100,
        "report/utility/gen": utility_gen * 100,
        "report/utility/rea": utility_rea * 100,
        "report/utility/tru": utility_tru * 100,
        "report/utility/fac": utility_fac * 100,
        "report/utility/flu": utility_flu * 100,
        "report/stage_number": stage_number,
    }

    log.info(f"Results: {results}")
    return results
