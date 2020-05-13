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

import pytest
import pandas
import numpy as np
import modin.pandas as pd
from modin.pandas.utils import from_pandas, to_pandas
from .utils import df_equals

pd.DEFAULT_NPARTITIONS = 4


def modin_df_almost_equals_pandas(modin_df, pandas_df):
    difference = to_pandas(modin_df) - pandas_df
    diff_max = difference.max().max()
    assert (
        to_pandas(modin_df).equals(pandas_df)
        or diff_max < 0.0001
        or (all(modin_df.isna().all()) and all(pandas_df.isna().all()))
    )


def modin_groupby_equals_pandas(modin_groupby, pandas_groupby):
    for g1, g2 in zip(modin_groupby, pandas_groupby):
        assert g1[0] == g2[0]
        df_equals(g1[1], g2[1])


@pytest.mark.parametrize("as_index", [True, False])
def test_mixed_dtypes_groupby(as_index):
    frame_data = np.random.randint(97, 198, size=(2 ** 6, 2 ** 4))
    pandas_df = pandas.DataFrame(frame_data).add_prefix("col")
    # Convert every other column to string
    for col in pandas_df.iloc[
        :, [i for i in range(len(pandas_df.columns)) if i % 2 == 0]
    ]:
        pandas_df[col] = [str(chr(i)) for i in pandas_df[col]]
    modin_df = from_pandas(pandas_df)

    n = 1

    by_values = [
        ("col1",),
        (lambda x: x % 2,),
        (modin_df["col0"].copy(), pandas_df["col0"].copy()),
    ]

    for by in by_values:
        modin_groupby = modin_df.groupby(by=by[0], as_index=as_index)
        pandas_groupby = pandas_df.groupby(by=by[-1], as_index=as_index)

        modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
        eval_ngroups(modin_groupby, pandas_groupby)
        eval_ffill(modin_groupby, pandas_groupby)
        eval_sem(modin_groupby, pandas_groupby)
        eval_mean(modin_groupby, pandas_groupby)
        eval_any(modin_groupby, pandas_groupby)
        eval_min(modin_groupby, pandas_groupby)
        eval_idxmax(modin_groupby, pandas_groupby)
        eval_ndim(modin_groupby, pandas_groupby)
        eval_cumsum(modin_groupby, pandas_groupby)
        eval_pct_change(modin_groupby, pandas_groupby)
        eval_cummax(modin_groupby, pandas_groupby)

        # TODO Add more apply functions
        apply_functions = [lambda df: df.sum(), min]
        for func in apply_functions:
            eval_apply(modin_groupby, pandas_groupby, func)

        eval_dtypes(modin_groupby, pandas_groupby)
        eval_first(modin_groupby, pandas_groupby)
        eval_backfill(modin_groupby, pandas_groupby)
        eval_cummin(modin_groupby, pandas_groupby)
        eval_bfill(modin_groupby, pandas_groupby)
        eval_idxmin(modin_groupby, pandas_groupby)
        eval_prod(modin_groupby, pandas_groupby)
        if as_index:
            eval_std(modin_groupby, pandas_groupby)
            eval_var(modin_groupby, pandas_groupby)
            eval_skew(modin_groupby, pandas_groupby)

        agg_functions = ["min", "max"]
        for func in agg_functions:
            eval_agg(modin_groupby, pandas_groupby, func)
            eval_aggregate(modin_groupby, pandas_groupby, func)

        eval_last(modin_groupby, pandas_groupby)
        eval_mad(modin_groupby, pandas_groupby)
        eval_max(modin_groupby, pandas_groupby)
        eval_len(modin_groupby, pandas_groupby)
        eval_sum(modin_groupby, pandas_groupby)
        eval_ngroup(modin_groupby, pandas_groupby)
        eval_nunique(modin_groupby, pandas_groupby)
        eval_median(modin_groupby, pandas_groupby)
        eval_head(modin_groupby, pandas_groupby, n)
        eval_cumprod(modin_groupby, pandas_groupby)
        eval_cov(modin_groupby, pandas_groupby)

        transform_functions = [lambda df: df, lambda df: df + df]
        for func in transform_functions:
            eval_transform(modin_groupby, pandas_groupby, func)

        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(modin_groupby, pandas_groupby, func)

        eval_corr(modin_groupby, pandas_groupby)
        eval_fillna(modin_groupby, pandas_groupby)
        eval_count(modin_groupby, pandas_groupby)
        eval_tail(modin_groupby, pandas_groupby, n)
        eval_quantile(modin_groupby, pandas_groupby)
        eval_take(modin_groupby, pandas_groupby)
        eval___getattr__(modin_groupby, pandas_groupby, "col2")
        eval_groups(modin_groupby, pandas_groupby)


@pytest.mark.parametrize(
    "by", [[1, 2, 1, 2], lambda x: x % 3, "col1", ["col1"], ["col1", "col2"]]
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("col1_categories", [True, False])
def test_simple_row_groupby(by, as_index, col1_categories):
    data = {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [3, 8, 12, 10],
            "col4": [17, 13, 16, 15],
            "col5": [-4, -5, -6, -7],
        }
    
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    if col1_categories:
        pandas_df.astype({"col1": "category"}, copy=False)
        modin_df.astype({"col1": "category"}, copy=False)

    n = 1
    modin_groupby = modin_df.groupby(by=by, as_index=as_index)
    pandas_groupby = pandas_df.groupby(by=by, as_index=as_index)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_ffill(modin_groupby, pandas_groupby)
    eval_sem(modin_groupby, pandas_groupby)
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_idxmax(modin_groupby, pandas_groupby)
    eval_ndim(modin_groupby, pandas_groupby)
    eval_cumsum(modin_groupby, pandas_groupby)
    eval_pct_change(modin_groupby, pandas_groupby)
    eval_cummax(modin_groupby, pandas_groupby)

    # pandas is inconsistent between test environment and here, more investigation is
    # required to understand why this is a mismatch because we default to pandas for
    # this particular case.
    if by != ["col1", "col2"]:
        apply_functions = [lambda df: df.sum(), min]
        for func in apply_functions:
            eval_apply(modin_groupby, pandas_groupby, func)

    eval_dtypes(modin_groupby, pandas_groupby)
    eval_first(modin_groupby, pandas_groupby)
    eval_backfill(modin_groupby, pandas_groupby)
    eval_cummin(modin_groupby, pandas_groupby)
    eval_bfill(modin_groupby, pandas_groupby)
    eval_idxmin(modin_groupby, pandas_groupby)
    eval_prod(modin_groupby, pandas_groupby)
    if as_index:
        eval_std(modin_groupby, pandas_groupby)
        eval_var(modin_groupby, pandas_groupby)
        eval_skew(modin_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(modin_groupby, pandas_groupby, func)
        eval_aggregate(modin_groupby, pandas_groupby, func)

    eval_last(modin_groupby, pandas_groupby)
    eval_mad(modin_groupby, pandas_groupby)
    eval_rank(modin_groupby, pandas_groupby)
    eval_max(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_ngroup(modin_groupby, pandas_groupby)
    eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_head(modin_groupby, pandas_groupby, n)
    eval_cumprod(modin_groupby, pandas_groupby)
    eval_cov(modin_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_corr(modin_groupby, pandas_groupby)
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_tail(modin_groupby, pandas_groupby, n)
    eval_quantile(modin_groupby, pandas_groupby)
    eval_take(modin_groupby, pandas_groupby)
    eval___getattr__(modin_groupby, pandas_groupby, "col3")
    eval_groups(modin_groupby, pandas_groupby)


def test_single_group_row_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 36, 7],
            "col3": [3, 8, 12, 10],
            "col4": [17, 3, 16, 15],
            "col5": [-4, 5, -6, -7],
        }
    )

    modin_df = from_pandas(pandas_df)

    by = ["1", "1", "1", "1"]
    n = 6

    modin_groupby = modin_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_ffill(modin_groupby, pandas_groupby)
    eval_sem(modin_groupby, pandas_groupby)
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_idxmax(modin_groupby, pandas_groupby)
    eval_ndim(modin_groupby, pandas_groupby)
    eval_cumsum(modin_groupby, pandas_groupby)
    eval_pct_change(modin_groupby, pandas_groupby)
    eval_cummax(modin_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_dtypes(modin_groupby, pandas_groupby)
    eval_first(modin_groupby, pandas_groupby)
    eval_backfill(modin_groupby, pandas_groupby)
    eval_cummin(modin_groupby, pandas_groupby)
    eval_bfill(modin_groupby, pandas_groupby)
    eval_idxmin(modin_groupby, pandas_groupby)
    eval_prod(modin_groupby, pandas_groupby)
    eval_std(modin_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(modin_groupby, pandas_groupby, func)
        eval_aggregate(modin_groupby, pandas_groupby, func)

    eval_last(modin_groupby, pandas_groupby)
    eval_mad(modin_groupby, pandas_groupby)
    eval_rank(modin_groupby, pandas_groupby)
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_ngroup(modin_groupby, pandas_groupby)
    eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_head(modin_groupby, pandas_groupby, n)
    eval_cumprod(modin_groupby, pandas_groupby)
    eval_cov(modin_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_corr(modin_groupby, pandas_groupby)
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_tail(modin_groupby, pandas_groupby, n)
    eval_quantile(modin_groupby, pandas_groupby)
    eval_take(modin_groupby, pandas_groupby)
    eval___getattr__(modin_groupby, pandas_groupby, "col2")
    eval_groups(modin_groupby, pandas_groupby)


def test_large_row_groupby():
    pandas_df = pandas.DataFrame(
        np.random.randint(0, 8, size=(100, 4)), columns=list("ABCD")
    )

    modin_df = from_pandas(pandas_df)

    by = [str(i) for i in pandas_df["A"].tolist()]
    n = 4

    modin_groupby = modin_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_ffill(modin_groupby, pandas_groupby)
    eval_sem(modin_groupby, pandas_groupby)
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_idxmax(modin_groupby, pandas_groupby)
    eval_ndim(modin_groupby, pandas_groupby)
    eval_cumsum(modin_groupby, pandas_groupby)
    eval_pct_change(modin_groupby, pandas_groupby)
    eval_cummax(modin_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_dtypes(modin_groupby, pandas_groupby)
    eval_first(modin_groupby, pandas_groupby)
    eval_backfill(modin_groupby, pandas_groupby)
    eval_cummin(modin_groupby, pandas_groupby)
    eval_bfill(modin_groupby, pandas_groupby)
    eval_idxmin(modin_groupby, pandas_groupby)
    # eval_prod(modin_groupby, pandas_groupby) causes overflows
    eval_std(modin_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(modin_groupby, pandas_groupby, func)
        eval_aggregate(modin_groupby, pandas_groupby, func)

    eval_last(modin_groupby, pandas_groupby)
    eval_mad(modin_groupby, pandas_groupby)
    eval_rank(modin_groupby, pandas_groupby)
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_ngroup(modin_groupby, pandas_groupby)
    eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_head(modin_groupby, pandas_groupby, n)
    # eval_cumprod(modin_groupby, pandas_groupby) causes overflows
    eval_cov(modin_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_corr(modin_groupby, pandas_groupby)
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_tail(modin_groupby, pandas_groupby, n)
    eval_quantile(modin_groupby, pandas_groupby)
    eval_take(modin_groupby, pandas_groupby)
    eval_groups(modin_groupby, pandas_groupby)


def test_simple_col_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 3, 2, 3],
            "col2": [4, 1, 6, 7],
            "col3": [3, 8, 2, 10],
            "col4": [1, 13, 6, 15],
            "col5": [-4, 5, 6, -7],
        }
    )

    modin_df = from_pandas(pandas_df)

    by = [1, 2, 3, 2, 1]

    modin_groupby = modin_df.groupby(axis=1, by=by)
    pandas_groupby = pandas_df.groupby(axis=1, by=by)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_ffill(modin_groupby, pandas_groupby)
    eval_sem(modin_groupby, pandas_groupby)
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_ndim(modin_groupby, pandas_groupby)

    eval_idxmax(modin_groupby, pandas_groupby)
    eval_idxmin(modin_groupby, pandas_groupby)
    eval_quantile(modin_groupby, pandas_groupby)

    # https://github.com/pandas-dev/pandas/issues/21127
    # eval_cumsum(modin_groupby, pandas_groupby)
    # eval_cummax(modin_groupby, pandas_groupby)
    # eval_cummin(modin_groupby, pandas_groupby)
    # eval_cumprod(modin_groupby, pandas_groupby)

    eval_pct_change(modin_groupby, pandas_groupby)
    apply_functions = [lambda df: -df, lambda df: df.sum(axis=1)]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_first(modin_groupby, pandas_groupby)
    eval_backfill(modin_groupby, pandas_groupby)
    eval_bfill(modin_groupby, pandas_groupby)
    eval_prod(modin_groupby, pandas_groupby)
    eval_std(modin_groupby, pandas_groupby)
    eval_last(modin_groupby, pandas_groupby)
    eval_mad(modin_groupby, pandas_groupby)
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)

    # Pandas fails on this case with ValueError
    # eval_ngroup(modin_groupby, pandas_groupby)
    # eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_cov(modin_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_corr(modin_groupby, pandas_groupby)
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_take(modin_groupby, pandas_groupby)
    eval_groups(modin_groupby, pandas_groupby)


@pytest.mark.parametrize(
    "by", [np.random.randint(0, 100, size=2 ** 8), lambda x: x % 3, None]
)
@pytest.mark.parametrize("as_index", [True, False])
def test_series_groupby(by, as_index):
    series_data = np.random.randint(97, 198, size=2 ** 8)
    modin_series = pd.Series(series_data)
    pandas_series = pandas.Series(series_data)
    n = 1

    try:
        pandas_groupby = pandas_series.groupby(by, as_index=as_index)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.groupby(by, as_index=as_index)
    else:
        modin_groupby = modin_series.groupby(by, as_index=as_index)

        modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
        eval_ngroups(modin_groupby, pandas_groupby)
        eval_ffill(modin_groupby, pandas_groupby)
        eval_sem(modin_groupby, pandas_groupby)
        eval_mean(modin_groupby, pandas_groupby)
        eval_any(modin_groupby, pandas_groupby)
        eval_min(modin_groupby, pandas_groupby)
        eval_idxmax(modin_groupby, pandas_groupby)
        eval_ndim(modin_groupby, pandas_groupby)
        eval_cumsum(modin_groupby, pandas_groupby)
        eval_pct_change(modin_groupby, pandas_groupby)
        eval_cummax(modin_groupby, pandas_groupby)

        apply_functions = [lambda df: df.sum(), min]
        for func in apply_functions:
            eval_apply(modin_groupby, pandas_groupby, func)

        eval_first(modin_groupby, pandas_groupby)
        eval_backfill(modin_groupby, pandas_groupby)
        eval_cummin(modin_groupby, pandas_groupby)
        eval_bfill(modin_groupby, pandas_groupby)
        eval_idxmin(modin_groupby, pandas_groupby)
        eval_prod(modin_groupby, pandas_groupby)
        if as_index:
            eval_std(modin_groupby, pandas_groupby)
            eval_var(modin_groupby, pandas_groupby)
            eval_skew(modin_groupby, pandas_groupby)

        agg_functions = ["min", "max"]
        for func in agg_functions:
            eval_agg(modin_groupby, pandas_groupby, func)
            eval_aggregate(modin_groupby, pandas_groupby, func)

        eval_last(modin_groupby, pandas_groupby)
        eval_mad(modin_groupby, pandas_groupby)
        eval_rank(modin_groupby, pandas_groupby)
        eval_max(modin_groupby, pandas_groupby)
        eval_len(modin_groupby, pandas_groupby)
        eval_sum(modin_groupby, pandas_groupby)
        eval_ngroup(modin_groupby, pandas_groupby)
        eval_nunique(modin_groupby, pandas_groupby)
        eval_median(modin_groupby, pandas_groupby)
        eval_head(modin_groupby, pandas_groupby, n)
        eval_cumprod(modin_groupby, pandas_groupby)

        transform_functions = [lambda df: df + 4, lambda df: -df - 10]
        for func in transform_functions:
            eval_transform(modin_groupby, pandas_groupby, func)

        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(modin_groupby, pandas_groupby, func)

        eval_fillna(modin_groupby, pandas_groupby)
        eval_count(modin_groupby, pandas_groupby)
        eval_tail(modin_groupby, pandas_groupby, n)
        eval_quantile(modin_groupby, pandas_groupby)
        eval_take(modin_groupby, pandas_groupby)
        eval_groups(modin_groupby, pandas_groupby)


def test_multi_column_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": np.random.randint(0, 100, size=1000),
            "col2": np.random.randint(0, 100, size=1000),
            "col3": np.random.randint(0, 100, size=1000),
            "col4": np.random.randint(0, 100, size=1000),
            "col5": np.random.randint(0, 100, size=1000),
        },
        index=["row{}".format(i) for i in range(1000)],
    )

    modin_df = from_pandas(pandas_df)
    by = ["col1", "col2"]

    df_equals(modin_df.groupby(by).count(), pandas_df.groupby(by).count())

    with pytest.warns(UserWarning):
        for k, _ in modin_df.groupby(by):
            assert isinstance(k, tuple)

    by = ["row0", "row1"]
    with pytest.raises(KeyError):
        modin_df.groupby(by, axis=1).count()


def eval_ngroups(modin_groupby, pandas_groupby):
    assert modin_groupby.ngroups == pandas_groupby.ngroups


def eval_skew(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.skew(), pandas_groupby.skew())


def eval_ffill(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            modin_groupby.ffill()
        except Exception:
            pass


def eval_sem(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.sem()


def eval_mean(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.mean(), pandas_groupby.mean())


def eval_any(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.any(), pandas_groupby.any())


def eval_min(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.min(), pandas_groupby.min())


def eval_idxmax(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.idxmax()


def eval_ndim(modin_groupby, pandas_groupby):
    assert modin_groupby.ndim == pandas_groupby.ndim


def eval_cumsum(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cumsum(axis=axis), pandas_groupby.cumsum(axis=axis))


def eval_pct_change(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            modin_groupby.pct_change()
        except Exception:
            pass


def eval_cummax(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cummax(axis=axis), pandas_groupby.cummax(axis=axis))


def eval_apply(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.apply(func), pandas_groupby.apply(func))


def eval_dtypes(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.dtypes, pandas_groupby.dtypes)


def eval_first(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.first()


def eval_backfill(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            modin_groupby.backfill()
        except Exception:
            pass


def eval_cummin(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cummin(axis=axis), pandas_groupby.cummin(axis=axis))


def eval_bfill(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            modin_groupby.bfill()
        except Exception:
            pass


def eval_idxmin(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.idxmin()


def eval_prod(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.prod(), pandas_groupby.prod())


def eval_std(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.std(), pandas_groupby.std())


def eval_aggregate(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.aggregate(func), pandas_groupby.aggregate(func))


def eval_agg(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.agg(func), pandas_groupby.agg(func))


def eval_last(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.last()


def eval_mad(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.mad()


def eval_rank(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.rank(), pandas_groupby.rank())


def eval_max(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.max(), pandas_groupby.max())


def eval_var(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.var(), pandas_groupby.var())


def eval_len(modin_groupby, pandas_groupby):
    assert len(modin_groupby) == len(pandas_groupby)


def eval_sum(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.sum(), pandas_groupby.sum())


def eval_ngroup(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.ngroup(), pandas_groupby.ngroup())


def eval_nunique(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.nunique(), pandas_groupby.nunique())


def eval_median(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.median(), pandas_groupby.median())


def eval_head(modin_groupby, pandas_groupby, n):
    with pytest.warns(UserWarning):
        modin_groupby.head()


def eval_cumprod(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cumprod(), pandas_groupby.cumprod())
    df_equals(modin_groupby.cumprod(axis=axis), pandas_groupby.cumprod(axis=axis))


def eval_cov(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.cov()


def eval_transform(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.transform(func), pandas_groupby.transform(func))


def eval_corr(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        modin_groupby.corr()


def eval_fillna(modin_groupby, pandas_groupby):
    df_equals(
        modin_groupby.fillna(method="ffill"), pandas_groupby.fillna(method="ffill")
    )


def eval_count(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.count(), pandas_groupby.count())


def eval_pipe(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.pipe(func), pandas_groupby.pipe(func))


def eval_tail(modin_groupby, pandas_groupby, n):
    with pytest.warns(UserWarning):
        modin_groupby.tail()


def eval_quantile(modin_groupby, pandas_groupby):
    try:
        pandas_result = pandas_groupby.quantile(q=0.4)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_groupby.quantile(q=0.4)
    else:
        df_equals(modin_groupby.quantile(q=0.4), pandas_result)


def eval_take(modin_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            modin_groupby.take()
        except Exception:
            pass


def eval___getattr__(modin_groupby, pandas_groupby, item):
    try:
        pandas_result = pandas_groupby[item].count()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_groupby[item].count()
    else:
        df_equals(modin_groupby[item].count(), pandas_result)


def eval_groups(modin_groupby, pandas_groupby):
    for k, v in modin_groupby.groups.items():
        assert v.equals(pandas_groupby.groups[k])


def eval_shift(modin_groupby, pandas_groupby):
    assert modin_groupby.groups == pandas_groupby.groups


def test_groupby_on_index_values_with_loop():
    length = 2 ** 6
    data = {
        "a": np.random.randint(0, 100, size=length),
        "b": np.random.randint(0, 100, size=length),
        "c": np.random.randint(0, 100, size=length),
    }
    idx = ["g1" if i % 3 != 0 else "g2" for i in range(length)]
    modin_df = pd.DataFrame(data, index=idx, columns=list("aba"))
    pandas_df = pandas.DataFrame(data, index=idx, columns=list("aba"))
    modin_groupby_obj = modin_df.groupby(modin_df.index)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.index)

    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}

    for k in modin_dict:
        df_equals(modin_dict[k], pandas_dict[k])

    modin_groupby_obj = modin_df.groupby(modin_df.columns, axis=1)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.columns, axis=1)

    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}

    for k in modin_dict:
        df_equals(modin_dict[k], pandas_dict[k])


def test_groupby_multiindex():
    frame_data = np.random.randint(0, 100, size=(2 ** 6, 2 ** 4))
    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    new_columns = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df.columns], names=["four", "two", "one"]
    )
    modin_df.columns = new_columns
    pandas_df.columns = new_columns
    modin_df.groupby(level=1, axis=1).sum()

    modin_df = modin_df.T
    pandas_df = pandas_df.T
    df_equals(modin_df.groupby(level=1).count(), pandas_df.groupby(level=1).count())
    df_equals(modin_df.groupby(by="four").count(), pandas_df.groupby(by="four").count())

    by = ["one", "two"]
    df_equals(modin_df.groupby(by=by).count(), pandas_df.groupby(by=by).count())
