class EasyInitSubclass:
    """
    A utility class to delegate most behavior to `__oins__`,
    while allowing certain attributes and methods to be overridden in subclasses.
    Example:
        from coffusers.utils import EasyInitSubclass
        class Extended(EasyInitSubclass):
            extended_value = []
            overrides = ["extended_method", "extended_value"]
            overrides.extend(EasyInitSubclass.overrides)

            def extended_method(self):
                return self.extended_value

            def __init__(self,obj):
                self.extended_value = []
                EasyInitSubclass.__init__(self,obj)

        obj = 1
        extended_obj = Extended(obj)
        extended_obj.extended_method()

    """

    overrides = ["overrides","__oins__"]

    def __init__(self,__oins__):
        self.__oins__ = __oins__

    def __setattr__(self, key, value):
        if key in self.overrides:
            return object.__setattr__(self,key,value)
        else:
            return self.__oins__.__setattr__(key,value)

    def __getattribute__(self, item):
        if item in object.__getattribute__(self,"overrides"):
            return object.__getattribute__(self,item)
        else:
            return object.__getattribute__(self, "__oins__").__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return self.__oins__.__call__(*args,**kwargs)

    def __repr__(self):
        return self.__oins__.__repr__()

    def __str__(self):
        return self.__oins__.__str__()

    def __getitem__(self, item):
        return self.__oins__.__getitem__(item)

    def __len__(self):
        return self.__oins__.__len__()

    def __contains__(self, item):
        return self.__oins__.__contains__(item)

    def __iter__(self):
        return self.__oins__.__iter__()

    def __next__(self):
        return self.__oins__.__next__()
