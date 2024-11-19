class EasyInitSubclass:
    # get these attributes directly, get the other attributes from __oins__
    overrides = ["overrides","__oins__"]

    def __init__(self,__oins__):
        self.__oins__ = __oins__

    def __setattr__(self, key, value):
        if key in self.overrides:
            return object.__setattr__(self,key,value)
        else:
            return self.__oins__.__setattr__(key,value)

    def __getattribute__(self, item):
        if item in object.__getattribute__(self,"__class__").overrides:
            return object.__getattribute__(self,item)
        else:
            return object.__getattribute__(self, "__oins__").__getattribute__(item)

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
