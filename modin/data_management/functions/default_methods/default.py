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

from modin.data_management.functions.function import Function
from modin.utils import try_cast_to_pandas

from pandas.core.dtypes.common import is_list_like
import pandas


class DataFrameDefault(Function):
    OBJECT_TYPE = "DataFrame"

    @classmethod
    def register(cls, func, inplace=False, fn_name=None):
        """
        Build function that do fallback to default pandas implementation for passed `func`.

        Parameters
        ----------
        func: callable,
            Function to apply to the casted to pandas frame or its property accesed
            by `frame_wrapper`.
        inplace: bool (default False),
            If True return an object to which `func` was applied, otherwise return
            the result of `func`.
        fn_name: str (optional),
            Function name which will be shown in default-to-pandas warning message.
            If not specified, name will be deducted from `func`.

        Returns
        -------
        Callable,
            Method that does fallback to pandas and applies `func` to the pandas frame
            or its property accesed by `frame_wrapper`.
        """
        if fn_name is None:
            fn_name = getattr(func, "__name__", str(func))

        if type(func) == property:
            func = cls.build_property_wrapper(func)

        func = cls.build_applier(func, inplace)
        func.__name__ = f"<function {cls.OBJECT_TYPE}.{fn_name}>"

        def wrapper(self, *args, **kwargs):
            args = try_cast_to_pandas(args)
            kwargs = try_cast_to_pandas(kwargs)
            return self.default_to_pandas(func, *args, **kwargs)

        return wrapper

    @classmethod
    def build_property_wrapper(cls, prop):
        """Build function that access specified property of the frame"""

        def property_wrapper(df):
            return prop.fget(df)

        return property_wrapper

    @classmethod
    def build_applier(cls, func, inplace):
        """
        Build function that will be defaulted to pandas.

        Parameters
        ----------
        func: callable,
            Function to apply to the casted to pandas frame or its property accesed
            by `frame_wrapper`.
        inplace: bool (default False),
            If True return an object to which `func` was applied, otherwise return
            the result of `func`.

        Returns
        -------
        Callable,
            Function that executes `func` under casted to pandas frame.
        """

        def applier(df, *args, **kwargs):
            """
            This function is directly applied to the casted to pandas frame,
            executes target function under it and processes result so it be possible to create a valid
            query compiler from it.
            """
            df = cls.frame_wrapper(df)
            result = func(df, *args, **kwargs)

            if (
                not isinstance(result, pandas.Series)
                and not isinstance(result, pandas.DataFrame)
                and func != "to_numpy"
                and func != pandas.DataFrame.to_numpy
            ):
                result = (
                    pandas.DataFrame(result)
                    if is_list_like(result)
                    else pandas.DataFrame([result])
                )
            if isinstance(result, pandas.Series):
                if result.name is None:
                    result.name = "__reduced__"
                result = result.to_frame()

            method_scoped_inplace = inplace or kwargs.get("inplace", False)
            return result if not method_scoped_inplace else df

        return applier

    @classmethod
    def frame_wrapper(cls, df):
        """
        Do nothing with passed frame.

        Note
        ----
        This method is executed under casted to pandas frame right before applying
        a function passed to `register`, which gives an ability to transform frame somehow
        or access its property, by overriding this method in a child classes.
        """
        return df
