\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Getting Started With SpECTRE {#spectre_getting_started}

# Building SpECTRE {#spectre_building}

SpECTRE uses the CMake build system and can be built with a variety of different
configurations. Most of the CMake flags to configure the build are documented
\ref common_cmake_flags "here". Below is a list of the most important
CMake configuration options:
- You must specify the Charm++ directory. For example, in the container you
  would do this by specifying
  `-D CHARM_ROOT=/work/charm/multicore-linux-x86_64-gcc`
- You can use the clang compiler by specifying `-D CMAKE_C_COMPILER=clang` and
  `-D CMAKE_CXX_COMPILER=clang++`. If you have a specific version of clang or
  GCC you can specify that as well, just use e.g. `clang-10` or `gcc-8`.
- Building in debug mode reduces build time and is fine as long as you aren't
  going to be doing long 3d simulations. To build in debug mode specify
  `-D CMAKE_BUILD_TYPE=Debug` (this is the default).
- SpECTRE requires around 12GB of memory to build, though this can be reduced to
  around 3-4GB by disabling debug symbols by specifying
  `-D DEBUG_SYMBOLS=OFF`. This will mean that running through a debugger is no
  longer useful, but our experience has been that memory errors (seg faults) are
  very rare in SpECTRE.
- In addition to using a lot of RAM, SpECTRE also requires quite a bit of disk
  space (~10GB by default). You can decrease this in a few different
  ways. First, you can disable debug symbols by specifying
  `-D DEBUG_SYMBOLS=OFF`. You can also strip remaining symbols by specify
  `-D STRIP_SYMBOLS=ON`. Finally, building with shared libraries will further
  reduce the size on disk. Shared libraries are enabled by specify
  `-D BUILD_SHARED_LIBS=ON`. Note that the python bindings don't yet fully work
  with shared library builds. All of the above together reduce the built total
  size to approximately 1.5GB with Clang-11.
- You can use the Ninja build generator by passing `-GNinja` to CMake. Ninja is
  an alternative to Make that gives up some functionality but the simplification
  allows for much faster dependency analysis leading to extremely quick
  rebuilds.
- Using ccache (compiler cache) allows for quick rebuilds when switching
  branches. ccache caches the object files during a build allowing the cached
  object to be used if the target is later triggered for a rebuild but a valid
  cache is stored. The default ccache size is around 5GB. Increasing this to
  20GB will help reduce build times if you manage several branches
  simultaneously. The ccache cache size is specified by adding `max_size = 20G`
  in `~/.ccache/ccache.conf`. ccache is enabled by default and can be disabled
  using `-D USE_CCACHE=OFF`. Note that if ccache is not installed it cannot be
  used.

With over ~250,000 lines of C++, SpECTRE is no longer a small code base.
Different parts of SpECTRE can be built separately to reduce how long it
takes to get the functionality you need. For example, if you want to run a 1d
plane wave scalar wave solution, you can only build the target
`EvolveScalarWavePlaneWave1D`. To build only the unit tests, you can build the
target `unit-tests`. To get a list of all the available targets, build
`list`. You will get an alphabetical list of all the targets. All evolution
executables are prefixed with `Evolve`, while elliptic solvers are prefixed with
`Solve`. When building targets you should build with almost all of your cores or
hyperthreads. For example, if you have a hyperthreaded dual-core CPU, then you
can reasonable build on 3 threads in parallel. This leaves one thread for you to
do other things while the code is building. Specifically, to build the unit
tests on a hyperthreaded dual-core CPU you would run
```
make -j3 unit-tests
```
or
```
ninja -j3 unit-tests
```
if you are using Ninja instead of Make.

Once you have built the unit tests you can run them using CTest as follows
```
ctest -L unit -j3
```
Notice that you can run the tests in parallel as well to help reduce wait time.

You can build all the executables used in the tests by building the target
`test-excutables`. This will build several evolution and elliptic solver
executables and test that the example input files work. Note that building the
executables can take a while and a single executable build can use 8GB of memory
if you haven't disabled debug symbols. You can run all the tests using
```
ctest -j3
```
While the input file tests use a lot of memory to build, they don't require much
memory to run, so running them in parallel is fine.

# Running your first 1d evolution {#spectre_running_first_1d_evolution}

As a concrete example we will evolve a 1d plane wave solution to the flat space
scalar wave system in this section. Later on we will evolve a 3d plane wave
solution to the flat space scalar wave system. Assuming you have configured a
SpECTRE build, build the `EvolveScalarWavePlaneWave1D` target:
```
make -j2 EvolveScalarWavePlaneWave1D
```
If you have more than 2 cores/threads, feel free to build on more. Once the
executable is built, copy the example input file from the SpECTRE repo
`tests/InputFiles/ScalarWave/PlaneWave1DObserveExample.yaml` into your build
directory (any location is fine, but for concreteness we will work in the build
directory). From here on out we will assume the current working directory is the
build directory and all paths will be specified relative to that unless
explicitly stated otherwise.

### Command line arguments and input files

You can see the help string for the executable by running:
```
./bin/EvolveScalarWavePlaneWave1D --help
```
There are three main parts to the help message. The first are the arguments that
Charm++ (the parallel runtime system SpECTRE uses) will accept. This section
will start with a string along the lines of:
```
Converse Machine Command-line Parameters:
```
Note that this string may depend on the Charm++ version and specific
build. Around 166 lines later, the SpECTRE options start. You will find the
message
```
SpECTRE Build Information
```
and about 8 lines below that the command line arguments will be listed. The most
important one is `--input-file` which we will use to specify the input
file. Note that SpECTRE executables have the capability of validating the input
file without running a full simulation by passing the flag
`--check-options`. This is extremely useful for avoiding silly input file typos
when submitting a large simulation on a supercomputer since nobody wants to wait
3 days in a queue only to be told they have a typo.

The last section of the SpECTRE help text is a description of the highest-level
(or root) options in the input file. This section starts with the string:
```
==== Description of expected options:
```
A brief description of the executable follows, and the actual input options are
listed under the
```
Options:
```
heading. For example, in the scalar wave executable we see options like
```
  AnalyticSolution:
    Analytic solution used for the initial data and errors
  DomainCreator:
    type=DomainCreator
    The domain to create initially
```

If you open up the `PlaneWave1DObserveExample.yaml` input file you will find the
same options listed there, including sub-options for each of them. A reasonable
way of figuring out what the sub-options of an option are is to cause an input
file error. For example, to figure out what the sub-options of
`AnalyticSolution` are I could specify:
```
AnalyticSolution:
  SomeInvalidOptionString:
```
If we now run the executable:
```
./bin/EvolveScalarWavePlaneWave1D --check-options \
                                  --input-file ./PlaneWave1DObserveExample.yaml
```
we get an error message that will look something like:
```
In ./PlaneWave1DObserveExample.yaml:
In group AnalyticSolution:
At line 11 column 3:
Option 'SomeInvalidOptionString' is not a valid option.

==== Description of expected options:
Analytic solution used for the initial data and errors

Options:
  PlaneWave:
    type=PlaneWave
    Options for the analytic solution


############ ERROR ############
```
From this we see that the valid options for `AnalyticSolution` are `PlaneWave`
and nothing else. This method of determining the options by causing an error can
be applied to any of the options or sub-options.

For now we will ignore most of the options in the input file, but you should
familiarize yourself with them before running production simulations. The
choices you make in the input file can have a major impact on how long the
simulation takes to run and how accurate your result is. Let's focus on the
`EventsAndTriggers` section. Events and triggers allow specific things to occur
(the event) when a criterion (the trigger) is met. The syntax is as follows:
```
EventsAndTriggers:
  ? Trigger:
      OptionsForTrigger
  : - EventA:
        OptionsForEventA
    - EventB:
        OptionsForEventB
```
For example, the trigger and event pairing:
```
  ? Slabs:
      EvenlySpaced:
        Interval: 3
        Offset: 5
  : - ObserveErrorNorms:
        SubfileName: Errors
```
writes the error norms of the evolve variables every Slabs starting on the fifth
Slab. A Slab in SpECTRE is a collection of time steps. When using global time
stepping Slabs and time steps are the same. This is not true when using local
time stepping. The events and triggers in the example input file are:

\snippet PlaneWave1DObserveExample.yaml observe_event_trigger

The `ObserveFields` events will write data into the `VolumeFileName` file with
the `.%h5` extension, while the `ObserveErrorNorms` events will write into the
`ReductionFileName` file with the extension `.%h5`. Each event specifies a
`SubfileName`, which is the subfile in the HDF5 file that it is written to. For
example, the error norms are written into
`ReductionFileName.%h5:/Errors.dat`. The extension of the subfile name is
determined by the event and type of data written. Error norms are written into
`h5::Dat` files, which is effectively a CSV file but in binary format. The
volume data is written into an `h5::VolumeData` file that contains information
about how the domain is divided up, where the grid points are, and the values of
the quantities or fields being observed/written to disk.
The last event that we need to be aware of is `Completion`. This ends the
simulation when it is run. We can see that (as of this writing) the simulation
ends after the 100th Slab.

### Running the executable and visualizing the output

We are now ready to run the executable. We do so using:
```
./bin/EvolveScalarWavePlaneWave1D --input-file ./PlaneWave1DObserveExample.yaml
```
If you run `ls` you should see
```
ScalarWavePlaneWave1DObserveExampleReductions.h5
ScalarWavePlaneWave1DObserveExampleVolume0.h5
```
The `0` after the `VolumeFileName` is because we write one volume file per
node. Since we are running on only one node we only get one volume file, the one
from node `0`. If we ran on two nodes we would also get a volume file from node
`1`. If you ran the simulation on a remote machine it's best to copy the data to
your computer for visualization. The data can be visualized remotely but we will
not discuss that in this tutorial.

We will use the `Render1D` application to visualize the data. This is a python
script that renders the data using matplotlib. To see what options are
available, run
```
./bin/Render1D --help
```
Let's see what data was written into `VolumePsiPiPhiEvery50Slabs`. We do this
using
```
./bin/Render1D --file-prefix ScalarWavePlaneWave1DObserveExampleVolume \
               --subfile-name VolumePsiPiPhiEvery50Slabs.vol \
               --list-vars
```
The `--file-prefix` is the name of the volume data HDF5 file without the node
number and `.%h5` extension. The `--subfile-name` is the one specified for the
event in the input file, excluding the `.vol` extension. Finally, `--list-vars`
means all the variables available for visualization are printed. You should get
something like
```
['Error(Phi)_x', 'Error(Pi)', 'Error(Psi)', 'Phi_x', 'Pi', 'Psi']
```
Now let's visualize the data from a single time step. You can do so by running
```
./bin/Render1D --file-prefix ScalarWavePlaneWave1DObserveExampleVolume \
               --subfile-name VolumePsiPiPhiEvery50Slabs  --var Psi --time 1
 ```
The `--time` is i'th time we observed, not the associated time. The time is
printed as the title of the plot. You can save the plot to disk by specifying
the `-o NameOfFile` option. For example, to save to `Psi.pdf` we would run
```
./bin/Render1D --file-prefix ScalarWavePlaneWave1DObserveExampleVolume \
               --subfile-name VolumePsiPiPhiEvery50Slabs  --var Psi --time 1 \
               -o Psi
 ```
Note that we omit the `.pdf` extension from the output file name.

Now let's look at the error norms that we observed. For this we will use the
`PlotDatFiles` application. We can see the options by running
```
./bin/PlotDatFiles --help
```
Let's see what we can plot from the `Errors` subfile. Running
```
./bin/PlotDatFiles --file ScalarWavePlaneWave1DObserveExampleReductions.h5 \
                   --subfile Errors --legend-only
```
we get the legend from the Dat file, which will be something like:
```
The legend in the dat subfile is:
['Time', 'NumberOfPoints', 'Error(Pi)', 'Error(Phi)', 'Error(Psi)']
```
Let's start off by plotting the `Error(Pi)` as a function of time. We do this as
follows:
```
./bin/PlotDatFiles --file ScalarWavePlaneWave1DObserveExampleReductions.h5 \
                   --subfile Errors --x-axis Time --functions "Error(Pi)" \
                   -o ErrorPi
```
We can now look at `ErrorPi.pdf` to see the output. There are a couple of things
that aren't very nice about this plot. First, the y-axis isn't log scale while
error plots usually use a log scale. We can fix this using the `--y-logscale`
flag. Additionally, `Pi` is written out, while ideally we would like the Greek
letter to be used. We can fix this by using the `--labels` flag as follows:
```
./bin/PlotDatFiles --file ScalarWavePlaneWave1DObserveExampleReductions.h5 \
                   --subfile Errors --x-axis Time --functions "Error(Pi)" \
                   --y-logscale \
 --labels '{"Error(Pi)":"$L_2(\\Pi_{\\mathrm{num}}-\\Pi_{\\mathrm{exact}})$"}' \
                   -o ErrorPi
```
The `--labels` flag takes a JSON dictionary that maps the labels in the Dat file
legend to labels that you want used in the legend of the plot. Finally, we can
also display the error in `Psi` as well:

```
./bin/PlotDatFiles --file ScalarWavePlaneWave1DObserveExampleReductions.h5 \
                   --subfile Errors --x-axis Time \
                   --functions "Error(Pi)" "Error(Psi)" \
                   --y-logscale \
 --labels '{"Error(Pi)":"$L_2(\\Pi_{\\mathrm{num}}-\\Pi_{\\mathrm{exact}})$",
 "Error(Psi)":"$L_2(\\Psi_{\\mathrm{num}}-\\Psi_{\\mathrm{exact}})$"}' \
                   -o ErrorPiAndPsi
```

# Running your first 3d evolution {#spectre_running_first_3d_evolution}

Assuming you have configured a SpECTRE build, build the
EvolveScalarWavePlaneWave3D target:
```
make -j2 EvolveScalarWavePlaneWave3D
```
Once the executable is built, copy the example input file from the SpECTRE repo
`tests/InputFiles/ScalarWave/PlaneWave3D.yaml` into your build
directory (any location is fine, but for concreteness we will work in the build
directory). From here on out we will assume the current working directory is the
build directory and all paths will be specified relative to that unless
explicitly stated otherwise.

This input file does not have any observers enabled so let's start by copying
the ones over from the 1d evolution. You will also need to increase the number
of slabs for the completion event from 5 to 100. Since 3d evolutions can be
quite expensive let's run this one on 4 cores (2 if you don't have 4) by
specifying the `+p4` parameter
```
./bin/EvolveScalarWavePlaneWave3D +p4 \
        --input-file ./PlaneWave1DObserveExample.yaml
```
Let's start this time by plotting the reduction data by running
```
./bin/PlotDatFiles --file ScalarWavePlaneWave3DReductions.h5 \
                   --subfile Errors --x-axis Time \
                   --functions "Error(Pi)" "Error(Psi)" \
                   --y-logscale \
 --labels '{"Error(Pi)":"$L_2(\\Pi_{\\mathrm{num}}-\\Pi_{\\mathrm{exact}})$",
 "Error(Psi)":"$L_2(\\Psi_{\\mathrm{num}}-\\Psi_{\\mathrm{exact}})$"}' \
                   -o ErrorPiAndPsi
```
You'll notice that the errors are quite large and quickly approaching
unity. This is because we actually only have 8 DG elements in this
simulation. Let's increase that to 64 by having 4 DG elements in each
direction. To do this, change the `InitialRefinement:` option in the
`%DomainCreator` from
```
    InitialRefinement: [1, 1, 1]
```
to
```
    InitialRefinement: [2, 2, 2]
```
Before rerunning the simulation we need to delete the HDF5 files:
```
rm ScalarWavePlaneWave3DVolume0.h5 ScalarWavePlaneWave3DReductions.h5
```
The simulation will take a little bit longer to run since we now have more DG
elements. Plotting the errors again you'll notice they are much smaller. We
could keep increasing the number of DG elements to decrease the error, but we
need to be mindful that eventually we may need to take smaller time steps
because we have reached the CFL limit. We won't discuss the CFL condition in
this tutorial.

Now let's visualize the 3d volume data. For that we will use
[ParaView](http://paraview.org/). The `GenerateXdmf` application generates an
XDMF file that tells ParaView how to read data out of the HDF5 file so you can
visualize it. Running
```
./bin/GenerateXdmf --help
```
you will see the available options for generating an XDMF file. Run
```
./bin/GenerateXdmf --file-prefix ScalarWavePlaneWave3DVolume \
                   --subfile-name VolumePsiPiPhiEvery50Slabs \
                   -o Sw3d
```
This will generate a file called `Sw3d.xdmf`. Now open ParaView, select
`File -> Open`, navigate to your build directory, open `Sw3d.xdmf` and when
asked which XDMF reader to use choose `XDMF Reader` (not any that are XDMF3).You
can the select which variables to visualize on the left and click `Apply` to
have ParaView generate the visualization. You can learn more about ParaView by
following the ParaView tutorials: https://www.paraview.org/tutorials/
