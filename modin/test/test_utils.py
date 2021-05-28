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
import modin.utils

from textwrap import dedent, indent


# Note: classes below are used for purely testing purposes - they
# simulate real-world use cases for _inherit_docstring
class BaseParent:
    def method(self):
        """ordinary method (base)"""

    def base_method(self):
        """ordinary method in base only"""

    @property
    def prop(self):
        """property"""

    @staticmethod
    def static():
        """static method"""

    @classmethod
    def clsmtd(cls):
        """class method"""


class BaseChild(BaseParent):
    """this is class docstring"""

    def method(self):
        """ordinary method (child)"""

    def own_method(self):
        """own method"""

    def no_overwrite(self):
        """another own method"""

    F = property(method)


@pytest.fixture(scope="module")
def wrapped_cls():
    @modin.utils._inherit_docstrings(BaseChild)
    class Wrapped:
        def method(self):
            pass

        def base_method(self):
            pass

        def own_method(self):
            pass

        def no_overwrite(self):
            """not overwritten doc"""

        @property
        def prop(self):
            return None

        @staticmethod
        def static():
            pass

        @classmethod
        def clsmtd(cls):
            pass

        F = property(method)

    return Wrapped


def _check_doc(wrapped, orig):
    assert wrapped.__doc__ == orig.__doc__
    if isinstance(wrapped, property):
        assert wrapped.fget.__doc_inherited__
    else:
        assert wrapped.__doc_inherited__


def test_doc_inherit_clslevel(wrapped_cls):
    _check_doc(wrapped_cls, BaseChild)


def test_doc_inherit_methods(wrapped_cls):
    _check_doc(wrapped_cls.method, BaseChild.method)
    _check_doc(wrapped_cls.base_method, BaseParent.base_method)
    _check_doc(wrapped_cls.own_method, BaseChild.own_method)
    assert wrapped_cls.no_overwrite.__doc__ != BaseChild.no_overwrite.__doc__
    assert not getattr(wrapped_cls.no_overwrite, "__doc_inherited__", False)


def test_doc_inherit_special(wrapped_cls):
    _check_doc(wrapped_cls.static, BaseChild.static)
    _check_doc(wrapped_cls.clsmtd, BaseChild.clsmtd)


def test_doc_inherit_props(wrapped_cls):
    assert type(wrapped_cls.method) == type(BaseChild.method)  # noqa: E721
    _check_doc(wrapped_cls.prop, BaseChild.prop)
    _check_doc(wrapped_cls.F, BaseChild.F)


def test_doc_inherit_prop_builder():
    def builder(name):
        return property(lambda self: name)

    class Parent:
        prop = builder("Parent")

    @modin.utils._inherit_docstrings(Parent)
    class Child(Parent):
        prop = builder("Child")

    assert Parent().prop == "Parent"
    assert Child().prop == "Child"


@pytest.mark.parametrize(
    "source_doc,to_append,expected",
    [
        (
            "One-line doc.",
            "One-line message.",
            "One-line doc.One-line message.",
        ),
        (
            """
            Regular doc-string
                With the setted indent style.
            """,
            """
                    Doc-string having different indents
                        in comparison with the regular one.
            """,
            """
            Regular doc-string
                With the setted indent style.

            Doc-string having different indents
                in comparison with the regular one.
            """,
        ),
    ],
)
def test_append_to_docstring(source_doc, to_append, expected):
    def source_fn():
        pass

    source_fn.__doc__ = source_doc
    result_fn = modin.utils.append_to_docstring(to_append)(source_fn)

    answer = dedent(result_fn.__doc__)
    expected = dedent(expected)

    assert answer == expected


def test_align_indents():
    source = """
    Source string that sets
        the indent pattern."""
    target = indent(source, " " * 5)
    result = modin.utils.align_indents(source, target)
    assert source == result


def test_format_string():
    template = """
            Source template string that has some {inline_placeholder}s.
            Placeholder1:
            {new_line_placeholder1}
            Placeholder2:
            {new_line_placeholder2}
            Placeholder3:
            {new_line_placeholder3}Text text:
                Placeholder4:
                {new_line_placeholder4}
    """

    singleline_value = "Single-line value"
    multiline_value = """
        Some string
            Having different indentation
        From the source one."""
    multiline_value_new_line = multiline_value + "\n"

    expected = """
            Source template string that has some Single-line values.
            Placeholder1:
            Some string
                Having different indentation
            From the source one.
            Placeholder2:
            Single-line value
            Placeholder3:
            Some string
                Having different indentation
            From the source one.
            Text text:
                Placeholder4:
                Some string
                    Having different indentation
                From the source one.
    """

    answer = modin.utils.format_string(
        template,
        inline_placeholder=singleline_value,
        new_line_placeholder1=multiline_value,
        new_line_placeholder2=singleline_value,
        new_line_placeholder3=multiline_value_new_line,
        new_line_placeholder4=multiline_value,
    )
    assert answer == expected
