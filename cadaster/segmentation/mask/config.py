from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Optional

from sacred import Ingredient


@dataclass
class ClassLabel:
    name: str
    color: str
    filter: Union[str, Dict[str, str]]


@dataclass(eq=True, frozen=True)
class Annotation:
    raster: str
    osm: str
    basename: Optional[str] = None


mask_config = Ingredient("mask")


@mask_config.config
def config():
    annotation_data: List[Annotation]
    classes_out: str
    mask_out_dir: str
    overwrite: bool = False
    cache_masks: bool = True
    progress: bool = True

    edges = {"use": True, "width": 3, "color": "#7c4dff"}

    nodes = {"use": True, "radius": 3, "color": "#00695c"}

    no_class_overlap: bool = True
    multilabel: bool = True

    classes: List[ClassLabel] = [
        asdict(
            ClassLabel(
                "road", "#FFF8E1", filter={"highway": "pedestrian", "area": "yes"}
            )
        ),
        asdict(ClassLabel("water", "#80D8FF", filter="water")),
        asdict(ClassLabel("sottoportico", "#FFD600", filter={"building": "roof"})),
        asdict(ClassLabel("courtyard", "#FCE4EC", filter={"landuse": "residential"})),
        asdict(ClassLabel("building", "#F06292", filter={"building": "yes"})),
    ]

    classes_offset = edges["use"] + nodes["use"]

    num_classes = classes_offset + len(classes)


def transform_annotation_data_config(annotation_data):
    return [Annotation(**annotation) for annotation in annotation_data]
