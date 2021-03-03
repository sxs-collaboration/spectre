\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# %Running CCE {#tutorial_cce}

The basic instructions for getting up and running with a stand-alone
CCE using external data are:
- Clone spectre and build the CharacteristicExtract target
- At this point (provided the build succeeds) you should have the
  executable `bin/CharacteristicExtract` in the build directory; You can now
  run it using an input file to provide options. The input file
  `tests/InputFiles/Cce/CharacteristicExtract.yaml` from the spectre
  source tree can help you get started with writing your input file.
  There are a few important notes there:
  - For resolution, the example input file has lmax (`Cce.LMax`) of 12, and
    filter lmax (`Filtering.FilterLMax`) of 10; that'll run pretty fast but
    might be a little low for full precision. lmax 16, filter 14 should be
    pretty good, and typically precision doesn't improve above lmax 24,
    filter 22 (be sure to update the filter as you update lmax -- it should
    generally be around 2 lower than the maximum l to reduce possible aliasing).
  - If you want to just run through the end of the provided worldtube data,
    you can just omit the `EndTime` option and the executable will figure it
    out from the worldtube file.
  - The `ScriOutputDensity` adds extra interpolation points to the output,
    which is useful for finite-difference derivatives on the output data, but
    otherwise it'll just unnecessarily inflate the output files, so if you
    don't need the extra points, best just set it to 1.
  - If you're extracting at 100M or less, best to reduce the `TargetStepSize`,
    to around .5 at 100M and lower yet for nearer extraction.
  - The `InitializeJ` in the example file uses `InverseCubic` which is a pretty
    primitive scheme, but early tests indicate that it gives the best results
    for most systems.
    If initial data is a concern, you can also try replacing the `InverseCubic`
    entry with:
  ```
    NoIncomingRadiation:
      AngularCoordTolerance: 1.0e-13
      MaxIterations: 500
      RequireConvergence: true
  ```
  which are probably pretty good choices for those parameters,
  and the `RequireConvergence: true` will cause the iterative solve in
  this version to error out if it doesn't find a good frame.

- An example of an appropriate submission command for slurm systems is:
  ```
  srun -n 1 -c 1 path/to/build/bin/CharacteristicExtract ++ppn 3 \
 --input-file path/to/input.yaml
  ```
  CCE doesn't currently scale to more than 4 cores, so those slurm options are
  best.
- CCE will work faster if the input worldtube hdf5 file is chunked in small
  numbers of complete rows.
  This is relevant because by default, SpEC writes its worldtube files
  chunked along full time-series columns, which is efficient for
  compression, but not for reading in to SpECTRE -- in that case,
  it is recommended to rechunk the input file before running CCE
  for maximum performance. This can be done, for instance, using h5py
  (you will need to fill in filenames appropriate to your case in place
  of "BondiCceR0050.h5" and "RechunkBondiCceR0050.h5"):
  ```py
  import h5py
  input_file = "BondiCceR0050.h5"
  output_file = "RechunkBondiCceR0050.h5"
  with h5py.File(input_file,'r') as input_h5,\
    h5py.File(output_file, 'w') as output_h5:
    for dset in input_h5:
        if("Version" in dset):
            output_h5[dset] = input_h5[dset][()]
            continue
        number_of_columns = input_h5[dset][()].shape[1]
        output_h5.create_dataset(dset, data=input_h5[dset],
                                 maxshape=(None, number_of_columns),
                                 chunks=(4, number_of_columns), dtype='d')
        for attribute in input_h5[dset].attrs.keys():
            output_h5[dset].attrs[attribute] = input_h5[dset].attrs[attribute]
  ```
- The output data will be written as spin-weighted spherical harmonic
  modes, one physical quantity per dataset, and each row will
  have the time value followed by the real and imaginary parts
  of the complex modes in m-varies-fastest order.

Once you have the volume data output file from a successful CCE run, you can
confirm the integrity of the h5 file and its contents by running
```
h5ls CharacteristicExtractVolumeData0.h5
```

For the volume file produced by a successful run, the output of the `h5ls`
should resemble
```
EthInertialRetardedTime.dat Dataset {3995/Inf, 163}
News.dat                 Dataset {3995/Inf, 163}
Psi0.dat                 Dataset {3995/Inf, 163}
Psi1.dat                 Dataset {3995/Inf, 163}
Psi2.dat                 Dataset {3995/Inf, 163}
Psi3.dat                 Dataset {3995/Inf, 163}
Psi4.dat                 Dataset {3995/Inf, 163}
Strain.dat               Dataset {3995/Inf, 163}
src.tar.gz               Dataset {3750199}
```

The `Strain.dat` represents the asymptotic transverse-traceless contribution
to the metric scaled by the Bondi radius (to give the asymptotically leading
part), the `News.dat` represents the first derivative of the strain, and each
of the `Psi...` datasets represent the Weyl scalars, each scaled by the
appropriate factor of the Bondi-Sachs radius to retrieve the asymptotically
leading contribution.

The `EthInertialRetardedTime.dat` is a diagnostic dataset that represents the
angular derivative of the inertial retarded time, which determines the
coordinate transformation that is performed at scri+.

The following python script will load data from a successful CCE run and
construct a plot of all of the physical waveform data.
```py
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5


def spectre_real_mode_index(l, m):
    return 2 * (l**2 + l + m)


def spectre_imag_mode_index(l, m):
    return 2 * (l**2 + l + m) + 1


def get_modes_from_block_output(filename, dataset, modes=[[2, 2], [3, 3]]):
    with h5.File(filename, "r") as h5_file:
        timeseries_data = (h5_file[dataset + ".dat"][()][:, [0] + list(
            np.array([[
                spectre_real_mode_index(x[0], x[1]),
                spectre_imag_mode_index(x[0], x[1])
            ] for x in modes]).flatten())])
    return timeseries_data


plot_quantities = ["Strain", "News", "Psi0", "Psi1", "Psi2", "Psi3", "Psi4"]
mode_set = [[2, 2], [3, 3]]
filename = "CharacteristicExtractVolume0.h5"
output_plot_filename = "CCE_plot.pdf"

legend = []
for (mode_l, mode_m) in mode_set:
    legend = np.append(legend, [
        r"Re $Y_{" + str(mode_l) + r"\," + str(mode_m) + r"}$",
        r"Im $Y_{" + str(mode_l) + r"\," + str(mode_m) + r"}$"
    ])

plt.figure(figsize=(8, 1.9 * len(plot_quantities)))
for i in range(len(plot_quantities)):
    ax = plt.subplot(len(plot_quantities), 1, i + 1)
    timeseries = np.transpose(
        get_modes_from_block_output(filename, plot_quantities[i], mode_set))
    for j in range(len(mode_set)):
        plt.plot(timeseries[0], timeseries[j + 1], linestyle='--', marker='')
    ax.set_xlabel("Simulation time (M)")
    ax.set_ylabel("Mode coefficient")
    ax.set_title(plot_quantities[i])
plt.tight_layout()
plt.savefig(output_plot_filename, dpi=400)
plt.clf()
```

### Input data formats

The worldtube data must be constructed as spheres of constant coordinate
radius, and (for the time being) written to a filename of the format
`...CceRXXXX.h5`, where the `XXXX` is to be replaced by the integer for which
the extraction radius is equal to `XXXX`M. For instance, a 100M extraction
should have filename `...CceR0100.h5`. This scheme of labeling files with the
extraction radius is constructed for compatibility with SpEC worldtube data.

There are two possible formats of the input data, one based on the Cauchy metric
at finite radius, and one based on Bondi data. The metric data format must be
provided as spherical harmonic modes with the following datasets:
- `gxx.dat`, `gxy.dat`, `gxz.dat`, `gyy.dat`, `gyz.dat`, `gzz.dat`
- `Drgxx.dat`, `Drgxy.dat`, `Drgxz.dat`, `Drgyy.dat`, `Drgyz.dat`, `Drgzz.dat`
- `Dtgxx.dat`, `Dtgxy.dat`, `Dtgxz.dat`, `Dtgyy.dat`, `Dtgyz.dat`, `Dtgzz.dat`
- `Shiftx.dat`, `Shifty.dat`, `Shiftz.dat`
- `DrShiftx.dat`, `DrShifty.dat`, `DrShiftz.dat`
- `DtShiftx.dat`, `DtShifty.dat`, `DtShiftz.dat`
- `Lapse.dat`
- `DrLapse.dat`
- `DtLapse.dat`

In this format, each row must start with the time stamp, and the remaining
values are the complex modes in m-varies-fastest format.

The second format is Bondi-Sachs metric component data.
This format is far more space-efficient (by around a factor of 4), and SpECTRE
provides a separate executable for converting to the Bondi-Sachs worldtube
format, `ReduceCceWorldtube`.
The `ReduceCceWorldtube` executable should be run on a Cauchy metric worldtube
file, and will produce a corresponding 'reduced' Bondi-Sachs worldtube file.
The basic command-line arguments for the executable are:
```
ReduceCceWorldtube --input-file CceR0050.h5 --output-file BondiCceR0050.h5\
 --lmax_factor 3
```
The argument `--lmax_factor` determines the factor by which the resolution at
which the boundary computation that is run will exceed the resolution of the
input and output files.
Empirically, we have found that `lmax_factor` of 3 is sufficient to achieve
roundoff precision in all boundary data we have attempted, and an `lmax_factor`
of 2 is usually sufficient to vastly exceed the precision of the simulation that
provided the boundary dataset.

The format is similar to the metric components, except in spin-weighted
spherical harmonic modes, and the real (spin-weight-0) quantities omit the
redundant negative-m modes and imaginary parts of m=0 modes.
The quantities that must be provided by the Bondi-Sachs metric data format are:
- `Beta.dat`
- `DrJ.dat`
- `DuR.dat`
- `H.dat`
- `J.dat`
- `Q.dat`
- `R.dat`
- `U.dat`
- `W.dat`

The Bondi-Sachs data may also be used directly for CCE input data.
To specify that the input type is in 'reduced' Bondi-Sachs form, use:
```
...
Cce:
  H5IsBondiData: True
...
```
Otherwise, for the Cauchy metric data format, use:
```
...
Cce:
  H5IsBondiData: False
...
```
