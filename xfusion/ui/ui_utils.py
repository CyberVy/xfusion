class UIMixin:
    overrides = ["load_ui"]

    def load_ui(self):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'load_ui'")
