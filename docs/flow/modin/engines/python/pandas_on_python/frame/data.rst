PandasOnPythonFrame
"""""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.data.PandasFrame`
for ``PandasOnPython`` backend. It serves as an intermediate level between
:py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler` and
:py:class:`~modin.engines.python.pandas_on_python.frame.partition_manager.PandasOnPythonFramePartitionManager`.

Public API
----------

.. autoclass:: modin.engines.python.pandas_on_python.frame.data.PandasOnPythonFrame
  :members: