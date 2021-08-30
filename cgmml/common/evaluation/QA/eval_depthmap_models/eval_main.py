import logging
import shutil
import sys
import time
from pathlib import Path

from azureml.core import Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.run import Run
from azureml.core.script_run_config import ScriptRunConfig

sys.path.append(Path(__file__).parent)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

CWD = Path(__file__).parent
REPO_DIR = Path(__file__).absolute().parents[5]
EVAL_EXPERIMENT_NAME = 'QA-pipeline'
CLUSTER_NAME = 'gpu-cluster-M60'
OVERALL_SAVE_PATH = './outputs'


def copy_dir(src: Path, tgt: Path, glob_pattern: str, should_touch_init: bool = False):
    logger.info("Creating temp folder")
    if tgt.exists():
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True, exist_ok=True)
    if should_touch_init:
        (tgt / '__init__.py').touch(exist_ok=False)

    paths_to_copy = list(src.glob(glob_pattern))
    logger.info(f"Copying to {tgt} the following files: {str(paths_to_copy)}")
    for p in paths_to_copy:
        destpath = tgt / p.relative_to(src)
        destpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, destpath)


if __name__ == "__main__":
    # Copy QA src/ dir
    temp_path = CWD / "temp_eval"
    copy_dir(src=CWD / "src", tgt=temp_path, glob_pattern='*.py')

    # Copy common/ folder
    common_dir_path = REPO_DIR / "cgmml/common"
    temp_common_dir = temp_path / "cgmml/common"
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)

    from temp_eval.cgmml.common.model_utils.environment import cgm_environment  # noqa: E402, F401

    workspace = Workspace.from_config()
    run = Run.get_context()

    experiment = Experiment(workspace=workspace, name=EVAL_EXPERIMENT_NAME)

    # Find/create a compute target.
    try:
        # Compute cluster exists. Just connect to it.
        compute_target = ComputeTarget(workspace=workspace, name=CLUSTER_NAME)
        logger.info("Found existing compute target.")
    except ComputeTargetException:
        logger.info("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', max_nodes=4)
        compute_target = ComputeTarget.create(workspace, CLUSTER_NAME, compute_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    logger.info("Compute target: %s", compute_target)

    # parameters used in the evaluation
    script_params = {}
    logger.info("script_params: %s", script_params)
    tags = script_params

    start = time.time()

    cgm_env = cgm_environment(workspace, curated_env_name="cgm-QA-v01", env_exist=True,
                              fpath_env_yml=CWD / "environment.yml")

    script_run_config = ScriptRunConfig(source_directory=temp_path,
                                        compute_target=compute_target,
                                        script='evaluate.py',
                                        arguments=[str(item) for sublist in script_params.items() for item in sublist],
                                        environment=cgm_env)

    # Set compute target.
    script_run_config.run_config.target = compute_target

    # Run the experiment.
    run = experiment.submit(config=script_run_config, tags=tags)

    # Show run.
    logger.info("Run: %s", run)

    # Check the logs of the current run until is complete
    run.wait_for_completion(show_output=True)

    # Print Completed when run is completed
    logger.info("Run status: %s", run.get_status())

    end = time.time()
    logger.info("Total time for evaluation experiment: %d sec", end - start)

    # Download the evaluation results of the model
    GET_CSV_FROM_EXPERIMENT_PATH = '.'
    run.download_files(OVERALL_SAVE_PATH, GET_CSV_FROM_EXPERIMENT_PATH)
    logger.info("Downloaded the result.csv")

    # Delete temp folder
    shutil.rmtree(temp_path)
