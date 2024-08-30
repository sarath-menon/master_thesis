from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineState:
    image_ids: List[int] = field(default_factory=list)
    dataset_sample: Optional['DatasetSample'] = None
    processed_prompts: Optional['ProcessedPrompts'] = None
    localization_results: Optional['LocalizationResults'] = None
    segmentation_results: Optional['SegmentationResults'] = None

# Use string annotations to avoid circular imports