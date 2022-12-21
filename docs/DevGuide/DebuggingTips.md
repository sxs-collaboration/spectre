\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Tips for debugging an executable {#runtime_errors}

Learn how to use a debugger such as gdb.

# Useful gdb commands

- To break when an exception is thrown `catch throw`

- To break on a specific exception type `catch throw std::out_of_range`
  (This may not work on all compilers or older versions of gdb.  In this
  case you also try setting a breakpoint on the constructor of the exception
  type, `break std::out_of_range::out_of_range`)

