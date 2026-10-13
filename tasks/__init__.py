import copy
from .training_tasks.TrainingTask_Object_Detection_DCT import TrainingTask_Object_Detection_DCT
from .training_tasks.TrainingTask_FOMO_DCT import TrainingTask_FOMO_DCT


def build_task(config):
    training_task_config = copy.deepcopy(config)
    task_name = training_task_config.pop("task_name")
    if task_name == "TrainingTask_Object_Detection_DCT":
        task = TrainingTask_Object_Detection_DCT(cfg=training_task_config)
    elif task_name == "TrainingTask_FOMO_DCT":
        task = TrainingTask_FOMO_DCT(cfg=training_task_config)
    else:
        raise NotImplementedError

    return task
