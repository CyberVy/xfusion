def inherit(instance):
    """
    :param instance:
    :return: cls
    """

    class I(instance.__class__):

        def __init__(self):
            object.__setattr__(self,"__oins__",instance)
            try:
                self.__dict__.update(instance.__dict__)
            except AttributeError:
                ...

        def __getattribute__(self, item):
            try:
                return object.__getattribute__(self,item)
            except AttributeError:
                return instance.__getattribute__(item)

        def __setattr__(self, key, value):
            return instance.__setattr__(key,value)

    I.__qualname__ = instance.__class__.__qualname__
    I.__name__ = instance.__class__.__name__
    return I
