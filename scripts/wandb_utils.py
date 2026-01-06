import wandb
from typing import Optional, Dict, Any, List


def init_wandb_run(
    project: str,
    job_type: str,
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Starts a W&B run with the given parameters.
    (Backwards compatible: existing calls without group/tags still work.)
    """
    run = wandb.init(
        project=project,
        job_type=job_type,
        config=config,
        name=run_name,
        group=group,
        tags=tags,
    )
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Logs the given metrics to W&B.
    """
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish_wandb_run() -> None:
    """
    Ends the current W&B run.
    """
    wandb.finish()
