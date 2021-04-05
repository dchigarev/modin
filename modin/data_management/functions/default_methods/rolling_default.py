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

from .default import DataFrameDefault


class RollingDefault(DataFrameDefault):
    OBJECT_TYPE = "Rolling"

    @classmethod
    def register(cls, func):
        """
        Build default to pandas function that provides a rolling window and executes
        `func` on it.

        Parameters
        ----------
        func: callable,
            Function to execute on a rolling window.

        Returns
        -------
        Callable,
            Method that does fallback to pandas and applies rolling function.
        """

        def fn(df, rolling_args, *args, **kwargs):
            roller = df.rolling(*rolling_args)

            if type(func) == property:
                return func.fget(roller)

            return func(roller, *args, **kwargs)

        return super().register(fn, fn_name=func.__name__)
