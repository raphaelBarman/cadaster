from collections import namedtuple


class BaseConfig:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict):
        result = cls()
        keys = result.to_dict().keys()
        for k, v in d.items():
            assert k in keys, k
            setattr(result, k, v)
        result.check_config()
        return result

    def check_config(self):
        pass


class MaskConfig(BaseConfig):

    edges_config_tuple = namedtuple(
        "edges_config", ["color", "use", "width"], defaults=["#7c4dff", True, 3]
    )
    nodes_config_tuple = namedtuple(
        "nodes_config", ["color", "use", "radius"], defaults=["#00695c", True, 3]
    )
    class_config_tuple = namedtuple("class_config", ["name", "color", "filter"])

    def __init__(self, **kwargs):
        general_config = kwargs.get("general", {})
        edges_config = general_config.get("edges", {})
        self.edges = self.edges_config_tuple(*edges_config)
