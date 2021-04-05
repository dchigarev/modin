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


class BinaryDefault(DataFrameDefault):
    """Build default-to-pandas methods which executes binary functions"""

    @classmethod
    def build_applier(cls, fn, inplace):
        """
        Build binary function that will be defaulted to pandas.

        Parameters
        ----------
        func: callable,
            Binary function to apply to the casted to pandas frame and the other operand.
        inplace: bool (default False),
            If True return an object to which `func` was applied, otherwise return
            the result of `func`.

        Returns
        -------
        Callable,
            Function that executes binary `func`.
        """

        def bin_ops_wrapper(df, other, *args, **kwargs):
            squeeze_other = kwargs.pop("broadcast", False) or kwargs.pop(
                "squeeze_other", False
            )
            squeeze_self = kwargs.pop("squeeze_self", False)

            if squeeze_other:
                other = other.squeeze(axis=1)

            if squeeze_self:
                df = df.squeeze(axis=1)

            return fn(df, other, *args, **kwargs)

        return super().build_applier(bin_ops_wrapper, inplace)
