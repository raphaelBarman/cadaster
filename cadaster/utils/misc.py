import os


class DefaultFactoryDict(dict):
    def __init__(self, factory, memory_efficient=False):
        super().__init__()
        self.factory = factory
        self.memory_efficient = memory_efficient

    def __missing__(self, key):
        result = self.factory(key)
        if self.memory_efficient:
            return result
        else:
            self[key] = result
            return self[key]


def path_to_basename(path: str) -> str:
    return "".join(os.path.basename(path).split(".")[-1:])


def is_older_modified(path1, path2):
    return os.path.getmtime(path1) < os.path.getmtime(path2)


def is_newer_modified(path1, path2):
    return not is_older_modified(path1, path2)
