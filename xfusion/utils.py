import threading,os,shutil,gc


class EasyInitSubclass:
    """
        A utility class for delegating most behavior to an internal object (`__oins__`),
        while allowing specific attributes and methods to be overridden in subclasses.

        Purpose:
            - Simplifies the extension of complex objects by delegating their behaviors to `__oins__`.
            - Allows subclasses to define their own attributes and methods by explicitly listing them in the `overrides` attribute.

        Important Notes:
            - Make sure to include all directly defined attributes and methods of your subclass in the `overrides` list.
            - Init EasyInitSubclass first when subclassed.
            - Do not put the variable names starting with "__" but not ending with "__" into attribute overrides (It,s a python feature called Name Mangling).
            - If an `AttributeError` occurs for an overridden attribute, verify that the attribute is correctly listed in `overrides`.

        Example:
            >>> class Extended(EasyInitSubclass):
            >>>    # Custom attributes and methods
            >>>    extended_value = []
            >>>    overrides = ["extended_method", "extended_value"]
            >>>
            >>>    def extended_method(self):
            >>>        return self.extended_value
            >>>
            >>>    def __init__(self, obj):
            >>>        EasyInitSubclass.__init__(self, obj)
            >>>        # Initialize custom attributes
            >>>        self.extended_value = []
            >>> obj = 1  # Original object
            >>> extended_obj = Extended(obj)
            >>> print(extended_obj.extended_method())

        Attributes:
            - `overrides` (list): A list of attribute and method names that should not be delegated to `__oins__`.

        """
    overrides = ["__oins__"]

    def __init__(self,__oins__):
        self.__oins__ = __oins__
        self.__oinstype__ = __oins__.__class__

    def __init_subclass__(cls):
        for item in cls.__bases__:
            if hasattr(item, "overrides"):
                cls.overrides.extend(item.overrides)
        cls.overrides = list(set(cls.overrides))

    def __setattr__(self, key, value):
        if key in object.__getattribute__(self,"overrides"):
            return object.__setattr__(self,key,value)
        else:
            return self.__oins__.__setattr__(key,value)

    def __getattribute__(self, item):
        if item in object.__getattribute__(self,"overrides"):
            return object.__getattribute__(self,item)
        else:
            # when AttributeError: __oins__.__class__ object has no attribute '{item}' found
            # please check the if {item} is in overrides
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

    def __enter__(self):
        return self.__oins__.__enter__(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.__oins__.__exit__(self,exc_type,exc_val,exc_tb)

    def __del__(self):
        # final step before self is completely deleted from RAM
        print(f"{object.__getattribute__(self,'__class__').__name__}:{id(self)} has been deleted from RAM.")

def delete(obj):
    """
        Example:
        >>> a = [1,2,3]
        >>> b = a
        >>> delete(a)
    """
    if obj is None:
        return 0, [], 0
    i = 0
    _i = 0
    referrers = []
    _referrers = []
    for item in gc.get_referrers(obj):
        if hasattr(item, "__dict__"):
            __dict__ = item.__dict__
        elif isinstance(item, dict):
            __dict__ = item
        elif isinstance(item, list):
            for index, _ in enumerate(item):
                if _ is obj:
                    item[index] = None
                    referrers.append(f"list.{index}")
                    i += 1
            continue
        else:
            _referrers.append(id(item))
            _i += 1
            continue

        target_keys = []
        for key, value in __dict__.items():
            if value is obj:
                target_keys.append(key)
                referrers.append(f"dict.{key}")
                i += 1
        for target_key in target_keys:
            __dict__.update({target_key: None})

    return i, referrers, _i, _referrers

class EditableImage(list):

    def __init__(self,image):
        list.__init__(self,[image])

    def edit(self,image):
        self.append(image)
        return self

    @property
    def now(self):
        return self[-1]

    def back(self,n=1):
        if n > len(self) - 1:
            n = len(self) - 1
        for item in range(n):
            self.pop(-1)
        return self

    def reset(self):
        for _ in range(len(self) - 1):
            self.pop(-1)
        return self

def threads_execute(f,args,_await=True):
    threads = []
    if _await:
        for arg in args[1:]:
            thread = threading.Thread(target=f,args=(arg,))
            threads.append(thread)
            thread.start()
        f(args[0])
        for thread in threads:
            thread.join()
    else:
        for arg in args:
            thread = threading.Thread(target=f, args=(arg,))
            threads.append(thread)
            thread.start()
    return threads

def delete_all_contents_of_path(folder_path):
    if os.path.exists(folder_path):
        for file_or_dir in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_or_dir)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
