tfdeploy
========

.. centered:: This page contains only API docs. For more info, visit `tfdeploy on GitHub <https://github.com/riga/tfdeploy>`_.


.. toctree::
   :maxdepth: 2


.. automodule:: tfdeploy


Classes
^^^^^^^

``Model``
---------

.. autoclass:: Model
   :member-order: bysource
   :members:


``Tensor``
----------

.. autoclass:: Tensor
   :member-order: bysource
   :members:


``Operation``
-------------

.. autoclass:: Operation
   :member-order: bysource
   :members:


Functions
^^^^^^^^^

``optimize``
------------

.. autofunction:: optimize


Other Attributes
^^^^^^^^^^^^^^^^

.. py:attribute:: IMPL_NUMPY

   Implementation type for ops that merely rely on numpy (the default).

.. py:attribute:: IMPL_SCIPY

   Implementation type for ops that also use scipy.

.. py:attribute:: IMPL_THEANO

   Implementation type for ops that also use theano.

.. py:attribute:: IMPL_TENSORFLOW

   Implementation type for ops that also use tensorflow.

.. py:attribute:: HAS_SCIPY

   A flag that is *True* when scipy is available on your system.

.. py:attribute:: HAS_THEANO

   A flag that is *True* when theano is available on your system.

.. py:attribute:: HAS_TENSORFLOW

   A flag that is *True* when tensorflow is available on your system.


Exceptions
^^^^^^^^^^

``UnknownOperationException``
-----------------------------

.. autoexception:: UnknownOperationException


``OperationMismatchException``
------------------------------

.. autoexception:: OperationMismatchException


``InvalidImplementationException``
----------------------------------

.. autoexception:: InvalidImplementationException


``UnknownImplementationException``
----------------------------------

.. autoexception:: UnknownImplementationException


``ScipyOperationException``
---------------------------

.. autoexception:: ScipyOperationException


``TensorflowOperationException``
--------------------------------

.. autoexception:: TensorflowOperationException
