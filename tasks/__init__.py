import copy
from .training_tasks.TrainingTask_NanoDet_DCT import TrainingTask_NanoDet_DCT
from .training_tasks.TrainingTask_FOMO_DCT import TrainingTask_FOMO_DCT
from .training_tasks.TrainingTask_FOMO_DCT_KD import TrainingTask_FOMO_DCT_KD
from .testing_tasks.TestingTask_NanoDet_DCT import TestingTask_NanoDet_DCT
from .testing_tasks.TestingTask_FOMO_DCT import TestingTask_FOMO_DCT


# Checkpoints saved before the rename still carry the old task_name.
NANODET_TASK_NAMES = ("TrainingTask_NanoDet_DCT", "TrainingTask_Object_Detection_DCT")


def build_task(config):
    training_task_config = copy.deepcopy(config)
    task_name = training_task_config.pop("task_name")
    if task_name in NANODET_TASK_NAMES:
        task = TrainingTask_NanoDet_DCT(cfg=training_task_config)
    elif task_name == "TrainingTask_FOMO_DCT":
        task = TrainingTask_FOMO_DCT(cfg=training_task_config)
    elif task_name == "TrainingTask_FOMO_DCT_KD":
        task = TrainingTask_FOMO_DCT_KD(cfg=training_task_config)
    else:
        raise NotImplementedError

    return task


def build_test_task(config, checkpoint, **options):
    """The test task matching the training task_name saved in the run's config.yml."""
    testing_task_config = copy.deepcopy(config)
    task_name = testing_task_config.pop("task_name")
    if task_name in NANODET_TASK_NAMES:
        task = TestingTask_NanoDet_DCT(cfg=testing_task_config, checkpoint=checkpoint, **options)
    elif task_name in ("TrainingTask_FOMO_DCT", "TrainingTask_FOMO_DCT_KD"):
        # A distilled student checkpoint holds only the student model.
        task = TestingTask_FOMO_DCT(cfg=testing_task_config, checkpoint=checkpoint, **options)
    else:
        raise NotImplementedError

    return task
