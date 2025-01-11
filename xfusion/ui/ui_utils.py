class UIMixin:
    overrides = ["load_ui"]

    def load_ui(self,**kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'load_ui'")


def lists_append(element,lists):
    for _list in lists:
        _list.append(element)
