\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Profiling With Charm++ Projections {#profiling_with_projections}

## Basic Setup and Compilation

To view trace data after a profiling run you must download Charm++'s
Projections software from their [website](http://charm.cs.illinois.edu/).
If you encounter issues it may
be necessary to clone the git repository and build the correct version
from scratch. Note that the version of Charm++ used to compile SpECTRE
should match the version of Projections used to analyze the trace data.
You can collect the trace data on a different machine than the one you
will be analyzing the data on. For example, you can collect the data on
a supercomputer and analyze it on your desktop or laptop.

For profiling you will want to use a production build of Charm++, which
means compiling Charm++ with the `--with-production` flag. To enable trace
collecting you must build with the `--enable-tracing` flag as well. For
example, on a multicore 64-bit Linux machine the build command would be
``` shell
./build charm++ multicore-linux-x86_64 gcc -j8 --with-production \
  --enable-tracing
```

When building SpECTRE for profiling there are several options to control
what is profiled. First, you'll want to compile in Release mode by specifying
`-D CMAKE_BUILD_TYPE=Release`. To enable tracing with Projections you must
specify the CMake variable `-D PROJECTIONS=ON`. By default only Charm++ entry
methods will be profiled. Because we use an entry method template you'll need to
look at the template parameters to determine which action was called for the
specific entry method invocation.

## Running SpECTRE With Trace Output

When running SpECTRE you must specify a directory to output trace data into.
This is done by adding the command line argument `+traceroot DIR` where `DIR` is
the directory to dump the trace data into. Note that `DIR` must already exist,
the application will not create it.
For example, to run the 3D scalar wave executable on a multicore build with
tracing enabled use

```shell
./Evolve3DScalarWave +p4 +traceroot ./traces
```

For more information on runtime options to control trace data see the
[Charm++ Projections manual]
(http://charm.cs.illinois.edu/manuals/html/projections/1.html).

## Visualizing Trace %Data In Projections

See the [Charm++ Projections manual]
(http://charm.cs.illinois.edu/manuals/html/projections/2.html)
for details.
