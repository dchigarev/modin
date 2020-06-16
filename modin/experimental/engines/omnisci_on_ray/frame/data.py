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

from modin.engines.base.frame.data import BasePandasFrame
from modin.experimental.backends.omnisci.query_compiler import DFAlgQueryCompiler
from .partition_manager import OmnisciOnRayFrameManager

from pandas.core.index import ensure_index, Index, MultiIndex
from pandas.core.dtypes.common import _get_dtype
import pandas as pd

from .df_algebra import (
    MaskNode,
    FrameNode,
    GroupbyAggNode,
    TransformNode,
    UnionNode,
    JoinNode,
)
from .expr import (
    InputRefExpr,
    LiteralExpr,
    build_if_then_else,
    build_dt_expr,
    _get_common_dtype,
    _agg_dtype,
    DirectMapper,
    TransformMapper,
    InputMapper,
)
from collections import OrderedDict

import ray
import numpy as np


class OmnisciOnRayFrame(BasePandasFrame):

    _query_compiler_cls = DFAlgQueryCompiler
    _frame_mgr_cls = OmnisciOnRayFrameManager

    _next_id = [1]

    def __init__(
        self,
        partitions=None,
        index=None,
        columns=None,
        row_lengths=None,
        column_widths=None,
        dtypes=None,
        op=None,
        index_cols=None,
    ):
        assert dtypes is not None

        self.id = str(type(self)._next_id[0])
        type(self)._next_id[0] += 1

        if index is not None:
            index = ensure_index(index)
        columns = ensure_index(columns)
        self._op = op
        self._index_cols = index_cols
        self._partitions = partitions
        self._index_cache = index
        self._columns_cache = columns
        self._row_lengths_cache = row_lengths
        self._column_widths_cache = column_widths
        if self._op is None:
            self._op = FrameNode(self)

        self._table_cols = columns.tolist()
        if self._index_cols is not None:
            self._table_cols = self._index_cols + self._table_cols

        assert len(dtypes) == len(self._table_cols)
        if isinstance(dtypes, list):
            self._dtypes = pd.Series(dtypes, index=self._table_cols)
        else:
            self._dtypes = dtypes

        if partitions is not None:
            self._filter_empties()

    def id_str(self):
        return f"frame${self.id}"

    def ref(self, col):
        if col == "__rowid__":
            return InputRefExpr(self, col, _get_dtype(int))
        return InputRefExpr(self, col, self._dtypes[col])

    def mask(
        self,
        row_indices=None,
        row_numeric_idx=None,
        col_indices=None,
        col_numeric_idx=None,
    ):
        if col_indices is not None:
            new_columns = col_indices
        elif col_numeric_idx is not None:
            new_columns = self.columns[col_numeric_idx]
        else:
            new_columns = self.columns

        op = MaskNode(
            self,
            row_indices=row_indices,
            row_numeric_idx=row_numeric_idx,
            col_indices=new_columns,
        )
        dtypes = self._dtypes_for_cols(self._index_cols, new_columns)

        return self.__constructor__(
            columns=new_columns, dtypes=dtypes, op=op, index_cols=self._index_cols
        )

    def _dtypes_for_cols(self, new_index, new_columns):
        if new_index is not None:
            res = self._dtypes[new_index + new_columns]
        else:
            res = self._dtypes[new_columns]
        return res

    def _dtypes_for_exprs(self, new_index, exprs):
        dtypes = []
        if new_index is not None:
            dtypes += self._dtypes[new_index].tolist()
        dtypes += [expr._dtype for expr in exprs.values()]
        return dtypes

    def groupby_agg(self, by, axis, agg, groupby_args, **kwargs):
        # Currently we only expect by to be a projection of the same frame
        if not isinstance(by, DFAlgQueryCompiler):
            raise NotImplementedError("unsupported groupby args")

        if axis != 0:
            raise NotImplementedError("groupby is supported for axis = 0 only")

        base = by._modin_frame._op.input[0]
        if not by._modin_frame._is_projection_of(base):
            raise NotImplementedError("unsupported groupby args")

        if self != base and not self._is_projection_of(base):
            raise NotImplementedError("unsupported groupby args")

        if groupby_args["level"] is not None:
            raise NotImplementedError("levels are not supported for groupby")

        groupby_cols = by.columns.tolist()
        new_columns = []
        index_cols = None

        if groupby_args["as_index"]:
            index_cols = groupby_cols.copy()
        else:
            new_columns = groupby_cols.copy()
        new_dtypes = base._dtypes[groupby_cols].tolist()

        if isinstance(agg, str):
            new_agg = {}
            for col in self.columns:
                if col not in groupby_cols:
                    new_agg[col] = agg
                    new_columns.append(col)
                    new_dtypes.append(_agg_dtype(agg, self._dtypes[col]))
            agg = new_agg
        else:
            assert isinstance(agg, dict), "unsupported aggregate type"
            for k, v in agg.items():
                if isinstance(v, list):
                    # TODO: support levels
                    for item in v:
                        new_columns.append(k + " " + item)
                        new_dtypes.append(_agg_dtype(item, self._dtypes[k]))
                else:
                    new_columns.append(k)
                    new_dtypes.append(_agg_dtype(v, self._dtypes[k]))
        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)

        new_op = GroupbyAggNode(base, groupby_cols, agg, groupby_args)
        new_frame = self.__constructor__(
            columns=new_columns, dtypes=new_dtypes, op=new_op, index_cols=index_cols
        )

        return new_frame

    def fillna(
        self, value=None, method=None, axis=None, limit=None, downcast=None,
    ):
        if axis != 0:
            raise NotImplementedError("fillna is supported for axis = 0 only")

        if limit is not None:
            raise NotImplementedError("fillna doesn't support limit yet")

        if downcast is not None:
            raise NotImplementedError("fillna doesn't support downcast yet")

        if method is not None:
            raise NotImplementedError("fillna doesn't support method yet")

        exprs = OrderedDict()
        if isinstance(value, dict):
            for col in self.columns:
                col_expr = self.ref(col)
                if col in value:
                    value_expr = LiteralExpr(value[col])
                    res_type = _get_common_dtype(value_expr._dtype, col_expr._dtype)
                    exprs[col] = build_if_then_else(
                        col_expr.is_null(), value_expr, col_expr, res_type
                    )
                else:
                    exprs[col] = col_expr
        elif np.isscalar(value):
            value_expr = LiteralExpr(value)
            for col in self.columns:
                col_expr = self.ref(col)
                res_type = _get_common_dtype(value_expr._dtype, col_expr._dtype)
                exprs[col] = build_if_then_else(
                    col_expr.is_null(), value_expr, col_expr, res_type
                )
        else:
            raise NotImplementedError("unsupported value for fillna")

        new_op = TransformNode(self, exprs)
        dtypes = self._dtypes_for_exprs(self._index_cols, exprs)
        new_frame = self.__constructor__(
            columns=self.columns, dtypes=dtypes, op=new_op, index_cols=self._index_cols
        )

        return new_frame

    def dt_year(self):
        exprs = OrderedDict()
        for col in self.columns:
            col_expr = self.ref(col)
            exprs[col] = build_dt_expr("year", col_expr)
        new_op = TransformNode(self, exprs)
        dtypes = self._dtypes_for_exprs(self._index_cols, exprs)
        return self.__constructor__(
            columns=self.columns, dtypes=dtypes, op=new_op, index_cols=self._index_cols
        )

    def join(self, other, how="inner", on=None, sort=False, suffixes=("_x", "_y")):
        assert (
            on is not None
        ), "Merge with unspecified 'on' parameter is not supported in the engine"

        for col in on:
            assert (
                col in self.columns and col in other.columns
            ), "Only cases when both frames contain key column are supported"

        new_columns = on.copy()
        new_dtypes = self._dtypes[on].tolist()

        conflicting_list = list(set(self.columns) & set(other.columns))
        for c in self.columns:
            if c not in on:
                suffix = suffixes[0] if c in conflicting_list else ""
                new_columns.append(c + suffix)
                new_dtypes.append(self._dtypes[c])
        for c in other.columns:
            if c not in on:
                suffix = suffixes[1] if c in conflicting_list else ""
                new_columns.append(c + suffix)
                new_dtypes.append(other._dtypes[c])

        op = JoinNode(self, other, how=how, on=on, sort=sort, suffixes=suffixes,)

        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)
        return self.__constructor__(dtypes=new_dtypes, columns=new_columns, op=op)

    def _index_width(self):
        if self._index_cols is None:
            return 1
        return len(self._index_cols)

    def _union_all(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        # determine output columns
        new_cols_map = OrderedDict()
        for col in self.columns:
            new_cols_map[col] = self._dtypes[col]
        for frame in other_modin_frames:
            if join == "inner":
                for col in list(new_cols_map):
                    if col not in frame.columns:
                        del new_cols_map[col]
            else:
                for col in frame.columns:
                    if col not in new_cols_map:
                        new_cols_map[col] = frame._dtypes[col]
        new_columns = list(new_cols_map.keys())

        if sort:
            new_columns = sorted(new_columns)

        # determine how many index components are going into
        # the resulting table
        if not ignore_index:
            index_width = self._index_width()
            for frame in other_modin_frames:
                index_width = min(index_width, frame._index_width())

        # compute resulting dtypes
        if sort:
            new_dtypes = [new_cols_map[col] for col in new_columns]
        else:
            new_dtypes = list(new_cols_map.values())

        # build projections to align all frames
        aligned_frames = []
        for frame in [self] + other_modin_frames:
            aligned_index = None
            exprs = OrderedDict()

            if not ignore_index:
                if frame._index_cols:
                    aligned_index = frame._index_cols[0 : index_width + 1]
                    aligned_index_dtypes = frame._dtypes[aligned_index].tolist()
                    for i in range(0, index_width):
                        col = frame._index_cols[i]
                        exprs[col] = frame.ref(col)
                else:
                    assert index_width == 1, "unexpected index width"
                    aligned_index = ["__index__"]
                    exprs["__index__"] = frame.ref("__rowid__")
                    aligned_index_dtypes = [_get_dtype(int)]
                aligned_dtypes = aligned_index_dtypes + new_dtypes
            else:
                aligned_dtypes = new_dtypes

            for col in new_columns:
                if col in frame._table_cols:
                    exprs[col] = frame.ref(col)
                else:
                    exprs[col] = LiteralExpr(None)

            aligned_frame_op = TransformNode(frame, exprs, False)
            aligned_frames.append(
                self.__constructor__(
                    columns=new_columns,
                    dtypes=aligned_dtypes,
                    op=aligned_frame_op,
                    index_cols=aligned_index,
                )
            )

        new_op = UnionNode(aligned_frames)
        return self.__constructor__(
            columns=new_columns,
            dtypes=aligned_frames[0]._dtypes,
            op=new_op,
            index_cols=aligned_frames[0]._index_cols,
        )

    def _concat(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        if axis == 0:
            return self._union_all(axis, other_modin_frames, join, sort, ignore_index)

        base = self
        for frame in other_modin_frames:
            base = base._find_common_projections_base(frame)
            if base is None:
                raise NotImplementedError("concat requiring join is not supported yet")

        exprs = OrderedDict()
        for col in self.columns:
            exprs[col] = self.ref(col)
        for frame in other_modin_frames:
            for col in frame.columns:
                if col == "" or col in exprs:
                    new_col = f"__col{len(exprs)}__"
                else:
                    new_col = col
                exprs[new_col] = frame.ref(col)

        exprs = self._translate_exprs_to_base(exprs, base)
        new_columns = Index.__new__(Index, data=exprs.keys(), dtype=self.columns.dtype)
        new_frame = self.__constructor__(
            columns=new_columns,
            dtypes=self._dtypes_for_exprs(self._index_cols, exprs),
            op=TransformNode(self, exprs),
            index_cols=self._index_cols,
        )
        return new_frame

    def bin_op(self, other, op_name, **kwargs):
        if isinstance(other, (int, float)):
            value_expr = LiteralExpr(other)
            exprs = {
                col: self.ref(col).bin_op(value_expr, op_name) for col in self.columns
            }
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(self._index_cols, exprs),
                op=TransformNode(self, exprs),
                index_cols=self._index_cols,
            )
        elif isinstance(other, list):
            if len(other) != len(self.columns):
                raise ValueError(
                    f"length must be {len(self.columns)}: given {len(other)}"
                )
            exprs = {
                col: self.ref(col).bin_op(LiteralExpr(val), op_name)
                for col, val in zip(self.columns, other)
            }
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(self._index_cols, exprs),
                op=TransformNode(self, exprs),
                index_cols=self._index_cols,
            )
        elif isinstance(other, type(self)):
            # For now we only support binary operations on
            # projections of the same frame, because we have
            # no support for outer join.
            base = self._find_common_projections_base(other)
            if base is None:
                raise NotImplementedError(
                    "unsupported binary op args (outer join is not supported)"
                )

            new_columns = self.columns.tolist()
            for col in other.columns:
                if col not in self.columns:
                    new_columns.append(col)
            new_columns = sorted(new_columns)

            fill_value = kwargs.get("fill_value", None)
            if fill_value is not None:
                fill_value = LiteralExpr(fill_value)

            exprs = OrderedDict()
            for col in new_columns:
                lhs = self.ref(col) if col in self.columns else fill_value
                rhs = other.ref(col) if col in other.columns else fill_value
                if lhs is None or rhs is None:
                    exprs[col] = LiteralExpr(None)
                else:
                    exprs[col] = lhs.bin_op(rhs, op_name)

            exprs = self._translate_exprs_to_base(exprs, base)
            return self.__constructor__(
                columns=new_columns,
                dtypes=self._dtypes_for_exprs(self._index_cols, exprs),
                op=TransformNode(base, exprs),
                index_cols=self._index_cols,
            )

    def _find_common_projections_base(self, rhs):
        bases = {self}
        while self._is_projection():
            self = self._op.input[0]
            bases.add(self)

        while rhs not in bases and rhs._is_projection():
            rhs = rhs._op.input[0]

        if rhs in bases:
            return rhs

        return None

    @staticmethod
    def _translate_exprs_to_base(exprs, base):
        new_exprs = dict(exprs)

        frames = set()
        for k, v in new_exprs.items():
            v.collect_frames(frames)
        frames.discard(base)

        while len(frames) > 0:
            mapper = InputMapper()
            new_frames = set()
            for frame in frames:
                frame_base = frame._op.input[0]
                if frame_base != base:
                    new_frames.add(frame_base)
                if isinstance(frame._op, MaskNode):
                    mapper.add_mapper(frame, DirectMapper(frame_base))
                else:
                    assert isinstance(frame._op, TransformNode)
                    mapper.add_mapper(frame, TransformMapper(frame._op.exprs))

            for k, v in new_exprs.items():
                new_expr = new_exprs[k].translate_input(mapper)
                new_expr.collect_frames(new_frames)
                new_exprs[k] = new_expr

            new_frames.discard(base)
            frames = new_frames

        res = OrderedDict()
        for col in exprs.keys():
            res[col] = new_exprs[col]
        return res

    def _is_projection_of(self, base):
        return (
            isinstance(self._op, MaskNode)
            and self._op.input[0] == base
            and self._op.row_indices is None
            and self._op.row_numeric_idx is None
        )

    def _is_projection(self):
        if isinstance(self._op, MaskNode):
            return self._op.row_indices is None and self._op.row_numeric_idx is None
        elif isinstance(self._op, TransformNode):
            return True
        return False

    def _execute(self):
        if isinstance(self._op, FrameNode):
            return

        new_partitions = self._frame_mgr_cls.run_exec_plan(self._op, self._index_cols)
        self._partitions = new_partitions
        self._op = FrameNode(self)

    def _build_index_cache(self):
        assert isinstance(self._op, FrameNode)
        assert self._partitions.size == 1
        self._index_cache = ray.get(self._partitions[0][0].oid).index

    def _get_index(self):
        self._execute()
        if self._index_cache is None:
            self._build_index_cache()
        return self._index_cache

    def _set_index(self, new_index):
        raise NotImplementedError("OmnisciOnRayFrame._set_index is not yet suported")

    def reset_index(self, drop):
        if drop:
            exprs = OrderedDict()
            for c in self.columns:
                exprs[c] = self.ref(c)
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(None, exprs),
                op=TransformNode(self, exprs, False),
                index_cols=None,
            )
        else:
            if self._index_cols is None:
                raise NotImplementedError(
                    "default index reset with no drop is not supported"
                )
            new_columns = Index.__new__(
                Index, data=self._table_cols, dtype=self.columns.dtype
            )
            return self.__constructor__(
                columns=new_columns,
                dtypes=self._dtypes_for_cols(None, new_columns),
                op=self._op,
                index_cols=None,
            )

    def _set_columns(self, new_columns):
        exprs = {new: self.ref(old) for old, new in zip(self.columns, new_columns)}
        return self.__constructor__(
            columns=new_columns,
            dtypes=self._dtypes.tolist(),
            op=TransformNode(self, exprs),
            index_cols=self._index_cols,
        )

    def _get_columns(self):
        return super(OmnisciOnRayFrame, self)._get_columns()

    columns = property(_get_columns)
    index = property(_get_index, _set_index)

    def has_multiindex(self):
        if self._index_cache is not None:
            return isinstance(self._index_cache, MultiIndex)
        return self._index_cols is not None and len(self._index_cols) > 1

    def to_pandas(self):
        self._execute()
        return super(OmnisciOnRayFrame, self).to_pandas()

    # @classmethod
    # def from_pandas(cls, df):
    #    return super().from_pandas(df)