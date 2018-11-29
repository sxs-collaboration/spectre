\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Running and Visualizing {#tutorial_visualization}

### Building an Executable from an Existing Source File

SpECTRE source files for evolution executables are located in
`src/Evolution/Executables`. Executables can be compiled by running the command
`make EXECUTABLE_NAME` where `EXECUTABLE_NAME` is the name of the executable
as defined in the `CMakeLists.txt` file located in the same directory as the
source file. For example, to compile the executable that evolves a scalar wave
using a three-dimensional domain, one runs the command:
`make EvolveScalarWave3D`, which then results in an executable of the same name
in the `bin` directory of the user's build directory.

### Running an Evolution Executable

Each SpECTRE executable reads a user-provided YAML file that specifies the
runtime settings of the evolution. This is where options such as the Domain,
AnalyticSolution, TimeStepper, etc are supplied. Example input files are kept
in `tests/InputFiles`, and are written to provide a functional base input file
which the user can then modify as desired. Copy the executable and YAML file
to a directory of your choice. The YAML file is then passed as an argument to
the executable using the flag `--input-file`. For example, for a scalar wave
evolution, run the command:
`./EvolveScalarWave3D --input-file Input3DPeriodic.yaml`.
By default, the example input files do not produce any output. This can be
changed by modifying the options passed to `ObserveNSlabs` and `ObserveAtT0`.
A successful observation will result in the creation of H5 files whose names
can be specified in the YAML file under the options `VolumeFileName` and
`ReductionFileName`. One volume data file will be produced from each Charm++
node that is used to run the executable. Each volume data file will have its
corresponding node number appended to its file name. Visualization of the
volume data will be described in the next section.

### 3D %Data Volume %Data In ParaView

A SpECTRE executable with observers produces volume and/or reduced data h5
files. An XDMF file must be created from the volume data in order to do
visualization using ParaView. To this end we provide the python executable
`GenerateXdmf.py` in the `tools` directory. `GenerateXdmf.py` takes two
arguments which are passed to `--file-prefix` and `--output`. The argument
passed to `--file-prefix` is the name of the H5 volume data, leaving out the
node number and extension. The argument passed to `--output` is the desired
.xmf file name, also without filename extension. Use `--help` to see a further
description of possible arguments. Open the .xmf file in ParaView and select
the `Xdmf Reader`, *not* the version 3 readers. On the left hand side of the
main ParaView window is a section named `Properties`, here you must click the
highlighted `Apply` button. ParaView will now render your volume data. If you
only wish to visualize a few datasets out of a large set, we recommended
unchecking the boxes for the datasets you wish to ignore under `Point Arrays`
before clicking `Apply`.

### Helpful ParaView Filters

Here we describe the usage of filters we've found to better visualize our data.
Feel free to contribute to this section!

#### Removing Mesh Imprinting
You may notice what appears to be mesh imprinting on the data. The imprinting
effect can be removed by applying the `Tetrahedralize` filter. To apply the
filter select the `Filters` menu item, then `Alphabetical` and finally
`Tetrahedralize`.

#### Creating Derived Volume Data
New volume data can be created from the existing volume data using the
`Calculator` filter. In the `Calculator`'s text box, input a numerical
expression in terms of existing datasets evaluating to the desired
quantity. For example, a vector-valued velocity dataset can be created
from three scalar velocity component datasets using the expression
`velocity_x * iHat + velocity_y * jHat + velocity_z * kHat` and hitting
the `Apply` button. By default, this will create a new dataset `Result`.
The name of the new dataset can be changed by changing the name provided
to `Result Array Name` above the `Calculator` text box.

#### Visualizing Vector Fields Derived From Scalar Fields
Use the `Calculator` filter described above in "Creating Derived Volume Data"
to create a new vector-valued dataset. Once this is created, use the `Glyph`
filter and set the `Active Attributes` to the vector you wish to visualize.
Make sure that `Scale Mode` is set to `vector`.
