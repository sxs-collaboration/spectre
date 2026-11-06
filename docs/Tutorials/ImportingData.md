\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Importing data into other programs {#tutorial_importing_data}

\tableofcontents

SpECTRE writes volume data to H5 files. To use the data in other programs, e.g.
to import initial data into another evolution code, interpolate it to the points
you need. All interfaces below use the same implementation, so they share these
features:

- Select the tensor components to interpolate, e.g. `Lapse` or `Shift_x`. List
  the available components with `spectre interpolate-to-points -l VOLFILES...`.
- Select the observation by its step, where 0 is the first observation and -1
  is the last.
- Parallelize the interpolation with OpenMP (configure SpECTRE with
  `-D ENABLE_OPENMP=ON`).
- Extrapolate into excision regions, e.g. to fill black hole interiors with
  smooth (but constraint-violating) data for moving-puncture codes.
- Points outside the domain are filled with NaN, unless you request an error.

## Command line

The `spectre interpolate-to-points` command interpolates to points listed on the
command line or in a text file and writes the result to a text file:

```sh
spectre interpolate-to-points VolumeData*.h5 -d VolumeData --step -1 \
  -y Lapse -y 'Shift_*' -t points.txt -o interpolated.txt
```

See `spectre interpolate-to-points --help` for all options and \ref tutorial_cli
for how to use the CLI.

## Python

```py
import numpy as np
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr
from spectre.IO.Exporter import ObservationStep, interpolate_to_points

# Target points with shape (3, num_points)
target_points = tnsr.I[DataVector, 3](np.array([x, y, z]))
lapse, shift_x = interpolate_to_points(
    "VolumeData*.h5",
    subfile_name="VolumeData",
    observation=ObservationStep(-1),
    tensor_components=["Lapse", "Shift_x"],
    target_points=target_points,
)
```

The `PointwiseInterpolator3DInertial` class (and its variants for other
dimensions and frames) in the same module loads the volume data into memory
once, which is faster if you need to interpolate repeatedly. See
\ref spectre_using_python for how to use SpECTRE's Python modules.

## C++, C, and other languages

Compile the `BundledExporter` library to link into your program:

```sh
make BundledExporter
```

This creates `lib/libBundledExporter` and the headers in `include/spectre/` in
the build directory. The test project in
`tests/Unit/IO/Exporter/BundledExporter` shows how to find and link them with
CMake.

The C++ interface is `spectre::Exporter::interpolate_to_points` in
`spectre/Exporter.hpp`:

```cpp
#include <spectre/Exporter.hpp>

const std::vector<std::vector<double>> result =
    spectre::Exporter::interpolate_to_points<3>(
        "VolumeData*.h5", "VolumeData", spectre::Exporter::ObservationStep{-1},
        {"Lapse", "Shift_x"}, {{x, y, z}});
```

The C interface is `spectre_interpolate_to_points` in `spectre/Exporter.h`. It
reads the target points and writes the result in place, so you can pass your
program's arrays directly without copying:

```c
#include <spectre/Exporter.h>

const char* volume_files[] = {"VolumeData0.h5", "VolumeData1.h5"};
const char* tensor_components[] = {"Lapse", "Shift_x"};
const double* target_points[] = {x, y, z};  // Each has num_points elements
double* out[] = {lapse, shift_x};  // Each has num_points elements
if (spectre_interpolate_to_points(out, volume_files, 2, "VolumeData", -1,
                                  tensor_components, 2, target_points, 3,
                                  num_points, false, false, 0) != 0) {
  fprintf(stderr, "%s\n", spectre_exporter_last_error());
}
```

The C interface can be called from any language with a C foreign-function
interface. For example, Fortran programs can declare it in a `bind(C)`
interface block using the `iso_c_binding` module.

## Mathematica

The `SpectreExporter.wl` package calls the C interface via
`ForeignFunctionLoad`. After compiling the `BundledExporter` library (see
above), load the package from the build directory:

```
Get["/path/to/spectre/build/include/spectre/SpectreExporter.wl"];

(* Target points with shape {3, numPoints} *)
{lapse, shiftX} = SpectreInterpolate[{"VolumeData0.h5", "VolumeData1.h5"},
  "VolumeData", -1, {"Lapse", "Shift_x"}, {x, y, z}]
```

Points outside the domain are `Indeterminate`. Run `?SpectreInterpolate` for
all options.
