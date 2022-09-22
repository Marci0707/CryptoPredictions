from dataclasses import dataclass, asdict
from typing import Sequence, Optional

import keras.optimizers


@dataclass
class TrainingConfig:
    training_id : str
    regression_days: int
    classifier_borders: Optional[Sequence[float]]
    manual_invalidation_percentile: int
    window_size: int
    optimizer: keras.optimizers.Optimizer

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}