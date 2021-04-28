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

"""Module houses Reduction functions builder class."""

from .function import Function


class ReductionFunction(Function):
    """Builder class for Reduction functions."""

    @classmethod
    # FIXME: spread `**call_kwds` into an actual function arguments.
    def call(cls, reduction_function, **call_kwds):  # noqa: PR02
        """
        Build Reduction function that will be performed across rows/columns.

        It's used if `func` reduces the dimension of partitions in contrast to `FoldFunction`.

        Parameters
        ----------
        reduction_function : callable(pandas.DataFrame) -> scalar
            Source function.
        axis : int, optional
            Axis to apply function along.
            (If specified, have to be passed via `kwargs`).
        **call_kwds : kwargs
            Additional parameters in the glory of compatibility. Does not affect the result.

        Returns
        -------
        callable
            Function that takes query compiler and executes Reduction function.
        """

        def caller(query_compiler, *args, **kwargs):
            """Execute Reduction function against passed query compiler."""
            axis = call_kwds.get("axis", kwargs.get("axis"))
            return query_compiler.__constructor__(
                query_compiler._modin_frame._fold_reduce(
                    cls.validate_axis(axis),
                    lambda x: reduction_function(x, *args, **kwargs),
                )
            )

        return caller
