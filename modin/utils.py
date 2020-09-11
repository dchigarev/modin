# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import pandas
import numpy as np


def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or (
                not callable(parent_obj) and not isinstance(parent_obj, property)
            ):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)
        return cls

    return decorator


def to_pandas(modin_obj):
    """Converts a Modin DataFrame/Series to a pandas DataFrame/Series.

    Args:
        obj {modin.DataFrame, modin.Series}: The Modin DataFrame/Series to convert.

    Returns:
        A new pandas DataFrame or Series.
    """
    return modin_obj._to_pandas()


def hashable(obj):
    """Return whether the object is hashable."""
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def try_cast_to_pandas(obj):
    """
    Converts obj and all nested objects from modin to pandas if it is possible,
    otherwise returns obj

    Parameters
    ----------
        obj : object,
            object to convert from modin to pandas

    Returns
    -------
        Converted object
    """
    if hasattr(obj, "_to_pandas"):
        return obj._to_pandas()
    if isinstance(obj, (list, tuple)):
        return type(obj)([try_cast_to_pandas(o) for o in obj])
    if isinstance(obj, dict):
        return {k: try_cast_to_pandas(v) for k, v in obj.items()}
    if callable(obj):
        module_hierarchy = getattr(obj, "__module__", "").split(".")
        fn_name = getattr(obj, "__name__", None)
        if fn_name and module_hierarchy[0] == "modin":
            return (
                getattr(pandas.DataFrame, fn_name, obj)
                if module_hierarchy[-1] == "dataframe"
                else getattr(pandas.Series, fn_name, obj)
            )
    return obj


def wrap_udf_function(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # if user accidently returns modin DataFrame or Series
        # casting it back to pandas to properly process
        return try_cast_to_pandas(result)

    wrapper.__name__ = func.__name__
    return wrapper


def safe_index_creator(
    values, like=None, level=0, positions=None, discard_bottom_levels=True
):
    """
    Creates new index from `values` with the same parameters
    as `like` (number of levels and its names).

    Parameters
    ----------
    values : list of str, callable
        list of labels of new index. If callable is passed, then
        it function will be aplied to every label in `like` (considering
        `level` and `positions`)

    like : Index, optional default None

    level : int, optional default 0
        if `like` is MultiIndex `level` parameter could be passed to
        specify the level to where labels from `values` should be inserted

    positions : list of int, optional
        positions where transformation with indices required (applying transform
        function from `values` or inserting labels at specified level in `like`)
        if not specifed applies transofrmation to all `values`

    discard_bottom_levels : bool, optional default True
        in case of MultiIndex `like` should we or not discard levels
        which number is greater than `level` parameter

    Returns
    -------
        New created Index

    Examples
    --------
    >>> indices
    MultiIndex([('a', 'one', 'foo', 'zoo'),
            ('a', 'one', 'foo', 'lol'),
            ('a', 'one', 'bar', 'zoo'),
            ('a', 'one', 'bar', 'lol'),
            ('a', 'two', 'foo', 'zoo')],
           names=['first', 'second', 'third', 'fourth'])

    >>> _safe_index_creator(['margin_name'], like=indices, positions=[0], level=1)
    MultiIndex([('a', 'margin_name', '', '')],
            names=['first', 'second', 'third', 'fourth'])

    >>> _safe_index_creator(
            ['margin_name'],
            like=indices,
            positions=[0],
            level=1,
            discard_bottom_levels=False
        )
    MultiIndex([('a', 'margin_name', 'foo', 'zoo')],
            names=['first', 'second', 'third', 'fourth'])

    >>> _safe_index_creator(lambda label: label[0], like=indices, level=2, positions=[0, 2, 4])
    MultiIndex([('a', 'one',   'f',    ''),
            ('a', 'one', 'foo', 'lol'),
            ('a', 'one',   'b',    ''),
            ('a', 'one', 'bar', 'lol'),
            ('a', 'two',   'f',    '')],
           names=['first', 'second', 'third', 'fourth'])

    """
    if like is None:
        return pandas.Index(list(values))

    if positions is None:
        positions = np.arange(len(like))

    is_miltiindex = isinstance(like, pandas.MultiIndex)

    def dummy_func(x):
        return x

    if callable(values):
        transform_func = values
        values = list(like)
        if is_miltiindex:
            for i in positions:
                values[i] = values[i][level]
    elif is_miltiindex:
        transform_func = dummy_func
    else:
        transform_func = None

    if is_miltiindex:
        for i in positions:
            values[i] = (
                (
                    *like[i][:level],
                    transform_func(values[i]),
                    *([""] * (len(like[i]) - level - 1)),
                )
                if discard_bottom_levels
                else (
                    *like[i][:level],
                    transform_func(values[i]),
                    *like[i][(level + 1) :],
                )
            )
    elif transform_func:
        for i in positions:
            values[i] = transform_func(values[i])

    new_index = pandas.Index(list(values))
    new_index.names = like.names
    return new_index
