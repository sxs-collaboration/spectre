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
./build charm++ multicore-linux64 gcc -j8 --with-production --enable-tracing
```

When building SpECTRE for profiling there are several options to control
what is profiled. First, you'll want to compile in Release mode by specifying
`-D CMAKE_BUILD_TYPE=Release`. To enable tracing with Projections you must
specify the CMake variable `-D PROJECTIONS=ON`. By default only Charm++ entry
methods will be profiled, which will not provide very much insight because
the AlgorithmChare infrastructure has only one entry method, which happens to
be a function template. You must specify a typelist of which Actions you would
like to profile before the first time `Evolution/EvolveSystem.hpp` is included
in the main executable. This typically means right after the
`the_ordered_actions_list` list is defined in the main executable. For example,

```cpp
using the_ordered_actions_list = tmpl::list<
    Algorithms::CheckTriggers<the_system, the_volume_dim>,
    Algorithms::SendDataForFluxes<the_system, the_volume_dim>,
    Algorithms::ComputeVolumeDtU<the_system, the_volume_dim>,
    Algorithms::ComputeBoundaryFlux<the_system, the_volume_dim>,
    Algorithms::ImposeExternalBoundaryConditions<the_system, the_volume_dim>,
    Algorithms::ComputeU<the_system, the_volume_dim>,
    Algorithms::AdvanceSlab<the_system, the_volume_dim>,
    Algorithms::UpdateInstance<the_system, the_volume_dim>>;

using the_trace_actions_list = tmpl::list<
    Algorithms::CheckTriggers<the_system, the_volume_dim>,
    Algorithms::ComputeVolumeDtU<the_system, the_volume_dim>,
    Algorithms::ComputeBoundaryFlux<the_system, the_volume_dim>,
    Algorithms::ImposeExternalBoundaryConditions<the_system, the_volume_dim>,
    Algorithms::ComputeU<the_system, the_volume_dim>>;
```

Actions not in `the_trace_actions_list` will not be traced but will still be
executed. The code also records time in the AlgorithmChare's `receive_data`
method that is
not taken up by Actions (typically checking if all data needed for the next
Action has been received) and also time spent saving the received data into
local memory.

## Using PAPI With Projections

\warning Using PAPI with Projections's stat counters requires Charm++ v6.8.0
or newer.

It is possible to collect information from hardware counters using PAPI
for the Actions by specifying
`-D PROJECTIONS_PAPI_COUNTERS="PAPI_L1_DCM,PAPI_L2_DCM"`. That is, a comma
separated list of PAPI counters to record. To see the list of available counters
on your hardware run `papi_avail -a`. The recorded user statistics can then be
analyzed inside Charm++ Projections. It is also possible to record custom user
statistics using Charm++ by specifying `-D PROJECTIONS_USER_STATS=ON` and
defining the variable `user_stat_names` as

```cpp
static constexpr std::array<const char*, 2> user_stat_names{
    {"name_1", "name_2"}};
```

The variable needs to be defined before `Evolution/EvolveSystem.hpp` is
included. To record statistics of PAPI counters it is recommended you disable
counters for Actions (you can still time profile them, though) and specify the
CMake variable `-D USE_PAPI=ON`. An example of how to record statistics from
PAPI counters is given below.

To provide a concrete example of tracing and analyzing hardware PMUs using PAPI
for only a subset of functions in an Action we will profile the function
`ScalarWaveEquations<Dim>::%compute_volume_dt_u`. There are several helper
functions provided in `Utilities/PAPI.hpp` that will come in useful. To start
the PAPI counters and record the time at the beginning of the function we call

```cpp
start_papi_counters(std::array<int, 2>{{PAPI_L1_DCM, PAPI_L2_DCM}});
const double start_time = get_time_from_papi();
```

and wherever we want to finish recording we run

```cpp
const double stop_time = get_time_from_papi();
auto counters = stop_papi_counters<2>();
```

The template parameter to `stop_papi_counters` must be the number of PAPI
counters being recorded, two in our example. The function `get_time_from_papi`
returns the time in seconds and `stop_time - start_time` gives the elapsed time
in seconds between the two calls. Using `get_time_from_papi` it is possible to
compute FLOP/s, or any other metric related to time. Finally, to store the
counter read outs into Projections's stat counter we use

```cpp
updateStat(projections_user_stat_offset, counters[0]);
updateStat(projections_user_stat_offset + 1, counters[1]);
```

The variable `projections_user_stat_offset` is used to ensure that the stat
numbers used by Charm++ internally do not collide with any used in the Actions,
unless you have over 9000 Actions.

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

For more information on runtime options to
control trace data see the
[Charm++ Projections manual](http://charm.cs.illinois.edu/manuals/html/projections/1.html).

## Visualizing Trace %Data In Projections

See the [Charm++ Projections manual](http://charm.cs.illinois.edu/manuals/html/projections/2.html)
for details.
