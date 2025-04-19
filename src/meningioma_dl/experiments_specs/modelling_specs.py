from dataclasses import dataclass

from meningioma_dl.experiments_specs.augmentation_specs import AugmentationSpecs
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.scheduler_specs import SchedulerSpecs


@dataclass
class ModellingSpecs:
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs()
    augmentation_specs: AugmentationSpecs = AugmentationSpecs()
    scheduler_specs: SchedulerSpecs = SchedulerSpecs()
    model_specs: ModelSpecs = ModelSpecs()
