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


class ResampleDefault(DataFrameDefault):
    OBJECT_TYPE = "Resampler"

    @classmethod
    def register(cls, func, squeeze_self=False):
        """
        Build default to pandas function that resamples time-series data in a casted
        to pandas frame and executes aggregation `func` on it.

        Parameters
        ----------
        func: callable,
            Aggregation function to execute under resampled frame.
        squeeze_self: bool,
            Whether or not to squeeze frame before resampling.

        Returns
        -------
        callable,
            Default to pandas function that applies aggregation to resampled time-series data.
        """

        def fn(df, resample_args, *args, **kwargs):
            if squeeze_self:
                df = df.squeeze(axis=1)
            resampler = df.resample(*resample_args)

            if type(func) == property:
                return func.fget(resampler)

            return func(resampler, *args, **kwargs)

        return super().register(fn, fn_name=func.__name__)
