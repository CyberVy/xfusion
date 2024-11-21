class EasyInitSubclass:
    """
        A utility class for delegating most behavior to an internal object (`__oins__`),
        while allowing specific attributes and methods to be overridden in subclasses.

        **Purpose**:
            - Simplifies the extension of complex objects by delegating their behaviors to `__oins__`.
            - Allows subclasses to define their own attributes and methods by explicitly listing them in the `overrides` attribute.

        **How It Works**:
            - Any attribute or method not listed in `overrides` is delegated to the internal object `__oins__`.
            - Subclasses should list all their custom attributes and methods in the `overrides` attribute.

        **Important**:
            - Make sure to include all directly defined attributes and methods of your subclass in the `overrides` list.
            - If an `AttributeError` occurs for an overridden attribute, verify that the attribute is correctly listed in `overrides`.

        **Example**:
            Extending a simple object with additional behavior:

            ```python
            from coffusers.utils import EasyInitSubclass

            class Extended(EasyInitSubclass):
                # Custom attributes and methods
                extended_value = []
                overrides = ["extended_method", "extended_value"]
                overrides.extend(EasyInitSubclass.overrides)  # Include parent overrides

                def extended_method(self):
                    return self.extended_value

                def __init__(self, obj):
                    # Initialize custom attributes
                    self.extended_value = []
                    EasyInitSubclass.__init__(self, obj)

            # Example usage
            obj = 1  # Original object
            extended_obj = Extended(obj)
            print(extended_obj.extended_method())  # Outputs: []
            ```

        **Attributes**:
            - `overrides` (list): A list of attribute and method names that should not be delegated to `__oins__`.

        **Subclassing Notes**:
            - Always include custom attributes and methods in `overrides` to ensure they are accessed and modified directly.
            - For delegation, the internal object (`__oins__`) must implement the corresponding attributes or methods.

        **Error Handling**:
            - If you encounter `AttributeError: 'NoneType' object has no attribute '{item}'`, ensure that `{item}` is correctly listed in `overrides` if it is directly defined in the subclass.
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
            # when AttributeError: 'NoneType' object has no attribute '{item}' found
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