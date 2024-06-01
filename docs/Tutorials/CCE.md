\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# %Running CCE {#tutorial_cce}

\tableofcontents

## Acquiring the CCE module {#acquiring_the_cce_module}

There are a couple different ways to acquire the CCE module/executable.

### From a release

Starting from late May 2024, in every
[Release of SpECTRE](https://github.com/sxs-collaboration/spectre/releases) we
offer a tarball that contains everything needed to run CCE on a large number of
different systems. Inside this tarball is

- the CCE executable `CharacteristicExtract`
- an example set of Bondi-Sachs worldtube data (see
   [Input worldtube data formats](#input_worldtube_data_formats) section)
- an example YAML input file
- example output from CCE
- a `ReduceCceWorldtube` executable for converting between
   [worldtube data formats](#input_worldtube_data_formats)

See [Running the CCE executable](#running_the_cce_executable) for how to run
CCE.

We have tested that this executable works natively on the following machines:

- Expanse
- Anvil
- Stampede3
- Delta (if you add
  `LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/lib64/:$LD_LIBRARY_PATH`
  before running CCE)
- Perlmutter
- Ubuntu 18.04 LTS or later (LTS version only)

We have also tested that this executable works inside our
[`dev` Docker container](https://hub.docker.com/r/sxscollaboration/spectre/tags)
on the following machines (in addition to the ones above):

- Frontera
- Delta

### From source

You can clone the [spectre repo](https://github.com/sxs-collaboration/spectre)
and follow the instructions on the \ref installation page to obtain an
environment to configure and build SpECTRE. Once you have a configured `build`
directory, build the CCE executable with

```
make CharacteristicExtract
```

\note You may want to add the `-j4` flag to speed up compilation. However, be
warned that this executable will need several GB of memory to build.

## Input worldtube data formats {#input_worldtube_data_formats}

The worldtube data must be constructed as spheres of constant coordinate
radius, and (for the time being) written to a filename of the format
`...CceRXXXX.h5`, where the `XXXX` is to be replaced by the integer for which
the extraction radius is equal to `XXXX`M. For instance, a 100M extraction
should have filename `...CceR0100.h5`. This scheme of labeling files with the
extraction radius is constructed for compatibility with SpEC worldtube data.

Currently CCE is able to read in worldtube data in two different formats.

### Cartesian metric and derivatives {#cartesian_metric_and_derivatives}

This metric data format must be provided as spherical harmonic modes of the
following datasets:
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
values are the complex modes in m-varies-fastest format. That is,
```
"time", "Lapse_Re(0,0)", "Lapse_Im(0,0)",
"Lapse_Re(1,1)", "Lapse_Im(1,1)", "Lapse_Re(1,0)", "Lapse_Im(1,0)",
"Lapse_Re(1,-1)", "Lapse_Im(1,-1)",
"Lapse_Re(2,2)", "Lapse_Im(2,2)", "Lapse_Re(2,1)", "Lapse_Im(2,1)",
"Lapse_Re(2,0)", "Lapse_Im(2,0)", "Lapse_Re(2,-1)", "Lapse_Im(2,-1)",
"Lapse_Re(2,-2)", "Lapse_Im(2,-2)"
```
Each dataset in the file must also have an attribute named "Legend" which
is an ASCII-encoded null-terminated variable-length string. That is, the HDF5
type is:
```
DATATYPE  H5T_STRING {
  STRSIZE H5T_VARIABLE;
  STRPAD H5T_STR_NULLTERM;
  CSET H5T_CSET_ASCII;
  CTYPE H5T_C_S1;
}
```
This can be checked for a dataset by running
```
h5dump -a DrLapse.dat/Legend CceR0150.h5
```

### Bondi-Sachs {#bondi_sachs}

The second format is Bondi-Sachs metric component data.
This format is far more space-efficient (by around a factor of 4) than the
[cartesian_metric](#cartesian_metric_and_derivatives) format.

The format is similar to the
[cartesian_metric](#cartesian_metric_and_derivatives) format, except in
spin-weighted spherical harmonic modes, and the real (spin-weight-0) quantities
omit the redundant negative-m modes and imaginary parts of m=0 modes.
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

An example of the columns of these dat files is

```
"time", "Re(0,0)", "Re(1,0)", "Re(1,1)", "Im(1,1)", "Re(2,0)",
"Re(2,1)", "Im(2,1)", "Re(2,2)", "Im(2,2)", "Re(3,0)", "Re(3,1)",
"Im(3,1)", "Re(3,2)", "Im(3,2)", "Re(3,3)", "Im(3,3)", ...
```

See \cite Moxon2020gha for a description of these quantities.

\note The columns of the legend of the
[cartesian_metric](#cartesian_metric_and_derivatives) format are different from
the [Bondi-Sachs](#bondi_sachs) format. In the
[cartesian_metric](#cartesian_metric_and_derivatives), the name of the quantity
is in the legend, while for [Bondi-Sachs](#bondi_sachs) it isn't.

### Converting data formats

Since the [Bondi-Sachs](#bondi_sachs) format is far more space-efficient,
SpECTRE provides a separate executable for converting from the
[cartesian_metric](#cartesian_metric_and_derivatives) format to the
[Bondi-Sachs](#bondi_sachs) worldtube format called `ReduceCceWorldtube`.
The `ReduceCceWorldtube` executable should be run on a
[cartesian_metric](#cartesian_metric_and_derivatives) worldtube file, and will
produce a corresponding 'reduced' Bondi-Sachs worldtube file.
The basic command-line arguments for the executable are:

```
ReduceCceWorldtube --input-file CceR0050.h5 --output-file BondiCceR0050.h5\
 --lmax_factor 3
```

The argument `--lmax_factor` determines the factor by which the resolution of
the boundary computation that is run will exceed the resolution of the
input and output files.
Empirically, we have found that `lmax_factor` of 3 is sufficient to achieve
roundoff precision in all boundary data we have attempted, and an `lmax_factor`
of 2 is usually sufficient to vastly exceed the precision of the simulation that
provided the boundary dataset.

### What Worldtube data "should" look like

While no two simulations will look exactly the same, there are some general
trends in the worldtube data to look for. Here is a plot of some modes of the
Bondi variable `J` from the [Bondi-Sachs](#bondi_sachs) worldtube format.

\image html worldtube_J.png "Bondi variable J on the Worldtube"

The 2,2 modes are oscillatory and capture the orbits of the two objects. The
real part of the 2,0 mode contains the gravitational memory of the system. Then
for this system, all the other modes are subdominant.

If you are using the [cartesian metric](#cartesian_metric_and_derivatives)
worldtube format, here is a plot of the imaginary part of the 2,2 mode of the
lapse and its radial and time derivative during inspiral.

\image html lapse.png "2,2 component of lapse and its radial and time derivative"

One thing to keep in mind with this plot is that it was produced using the
Generalized Harmonic formulation of Einstein's equations using the damped
harmonic gauge. Therefore, if you are using a different formulation and gauge
(like BSSN + moving punctures), the lapse may look different than this. One way
to sanity check your data (regardless of what type it is or where you got it
from) is to look and how the mode amplitude for a given quantity decays as you
increase (l,m). Here is a plot of the amplitude of the modes for the lapse in
the above plot.

\image html amp_lapse.png "Amplitude of modes of Lapse"

You'll notice that most modes are around machine precision and only the first
few have any real impact. This is expected.

## Input file for CCE

Input files for CCE are commonly named `CharacteristicExtract.yaml`. An example
input file with comments explaining some of the options can be found in
`$SPECTRE_HOME/tests/InputFiles/Cce/CharacteristicExtract.yaml`. Here we expand
a bit on why we chose some of those parameters.

### General options

- For resolution, the example input file has lmax (`Cce.LMax`) of 20, and
  filter lmax (`Filtering.FilterLMax`) of 18; that may run a bit slow for
  basic tests, but this value yields the best precision-to-run-time ratio
  for a typical BBH system. Note that precision doesn't improve above lmax 24,
  filter 22 (be sure to update the filter as you update lmax -- the filter
  should generally be around 2 lower than the maximum l to reduce possible
  aliasing).
- If you want to just run over all times in the worldtube H5 file, you can
  set both the `StartTime` and `EndTime` to `Auto` and it will automatically
  figure it out based on the data in the worldtube file.
- The `ScriOutputDensity` adds extra interpolation points to the output,
  which is useful for finite-difference derivatives on the output data, but
  otherwise it'll just unnecessarily inflate the output files, so if you
  don't need the extra points, best just set it to 1.
- For production level runs, it's recommended to have the
  `Cce.Evolution.StepChoosers.Constant` option set to 0.1 for an accurate time
  evolution. However, if you're just testing, this can be increased to 0.5 to
  speed things up.
- We generally do not recommend extracting at less than 100M due to the CCE junk
  radiation being much worse at these smaller worldtube radii. That being said,
  we also recommend running CCE over several worldtube radii and checking which
  is the best based on the Bianchi identity violations. There isn't necessarily
  a "best radius" to extract waveforms at.
- If the worldtube data is in the [Bondi-Sachs](#bondi_sachs) form, set
  `Cce.H5IsBondiData` to `True`. If the worldtube data is the
  [cartesian_metric](#cartesian_metric_and_derivatives) form, set
  `Cce.H5IsBondiData` to `False`.

### Initial data on the null hypersurface

Choosing initial data on the initial null hypersurface is a non-trivial task and
is an active area of research. We want initial data that will reduce the amount
of CCE junk radiation as much as possible, while also having the initial data
work for as many cases as possible.

SpECTRE currently has four different methods to choose the initial data on
the null hypersurface. In order from most recommended to least recommended,
these are:

- `ConformalFactor`: Try to make initial time coordinate as inertial as
  possible at \f$\mathscr{I}^+\f$ with a smart choice of the conformal factor.
  This will work for many cases, but not all. But will produce the best initial
  data when it does work.
- `InverseCubic`: Ansatz where \f$J = A/r + B/r^3\f$. This is very robust and
  almost never fails, but contains a lot of CCE junk radiation compared to
  `ConformalFactor`.
- `ZeroNonSmooth`: Make `J` vanish. Like the name says, it's not smooth.
- `NoIncomingRadiation`: Make \f$\Psi_0 = 0\f$; this does not actually lead
  to no incoming radiation, since \f$\Psi_0\f$ and \f$\Psi_4\f$ both include
  incoming and outgoing radiation.

### Rechunking worldtube data

\note This section is less important than the others and really only matters if
you will be doing a large number of CCE runs like for a catalog. For only a
couple runs or just for testing, this part is unnecessary and can be skipped.

CCE will run faster if the input worldtube hdf5 file is chunked in small numbers
of complete rows. This is relevant because by default, SpEC and SpECTRE write
their worldtube  files chunked along full time-series columns, which is
efficient for writing and compression, but not for reading in to CCE. In that
case, you can rechunk the input file before running CCE for maximum performance.
This can be done, for instance, using h5py (you will need to fill in filenames
appropriate to your case in place of "BondiCceR0050.h5" and
"RechunkBondiCceR0050.h5"):

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

The rechunked data will still be in the same
[format](#input_worldtube_data_formats) as before, but will just have a
different underlying structure in the H5 file that makes it faster to read in.

## Running the CCE executable {#running_the_cce_executable}

Once you have [acquired an executable](#acquiring_the_cce_module), running CCE
in a supported environment is a simple command:

```
./CharacteristicExtract --input-file CharacteristicExtract.yaml
```

You may notice at the beginning you get some warnings that look like

```
Warning: iterative angular solve did not reach target tolerance 1.000000e-13.
Exited after 300 iterations, achieving final maximum over collocation points
 for deviation from target of 2.073455e-08
Proceeding with evolution using the partial result from partial angular solve.
```

This is normal and expected. All it means is that initially an angular solve
didn't hit a tolerance. We've found that it never really reaches the tolerance
of 1e-13, but we still keep this tolerance so it gets as low as possible.

After this, you'll likely see some output like

```
Simulation time: 10.000000
  Wall time: 00:00:44
Simulation time: 20.000000
  Wall time: 00:01:14
```

which tells you that the simulation is proceeding as expected. When the run
finished, you'll see something like

```
Done!
Wall time: 06:01:41
Date and time at completion: Thu May 23 22:31:27 2024

[Partition 0][Node 0] End of program
```

In terms of runtime, we've found that for a ~5000M long cauchy simulation, CCE
takes about 6 hours to run. This will vary based on a number of factors like how
long the cauchy evolution actually is, the desired error tolerance of your
characteristic timestepper, and also how much data you output at future null
infinity. Therefore, take these numbers with a grain of salt and only use them
as a rough estimate for how long a job will take.

\note CCE can technically run on two (2) cores by adding the option `++ppn 2` to
the above command, however, we have found in practice that this makes
little-to-no difference in the runtime of the executable.

## Output from CCE

Once you have the reduction data output file from a successful CCE run, you can
confirm the integrity of the h5 file and its contents by running

```
h5ls -r CharacteristicExtractReduction.h5
```

For the reduction file produced by a successful run, the output of the `h5ls`
should resemble

```
/SpectreR0100.cce                 Group
/SpectreR0100.cce/EthInertialRetardedTime Dataset {26451/Inf, 163}
/SpectreR0100.cce/News            Dataset {26451/Inf, 163}
/SpectreR0100.cce/Psi0            Dataset {26451/Inf, 163}
/SpectreR0100.cce/Psi1            Dataset {26451/Inf, 163}
/SpectreR0100.cce/Psi2            Dataset {26451/Inf, 163}
/SpectreR0100.cce/Psi3            Dataset {26451/Inf, 163}
/SpectreR0100.cce/Psi4            Dataset {26451/Inf, 163}
/SpectreR0100.cce/Strain          Dataset {26451/Inf, 163}
/src.tar.gz              Dataset {7757329}
```

Notice that the worldtube radius will be encoded into the subfile name.

\note Prior to
[this Pull Request](https://github.com/sxs-collaboration/spectre/pull/5985),
merged May 15, 2024, the output of `h5ls` looked like this
```
/                        Group
/Cce                     Group
/Cce/EthInertialRetardedTime.dat Dataset {3995/Inf, 163}
/Cce/News.dat                 Dataset {3995/Inf, 163}
/Cce/Psi0.dat                 Dataset {3995/Inf, 163}
/Cce/Psi1.dat                 Dataset {3995/Inf, 163}
/Cce/Psi2.dat                 Dataset {3995/Inf, 163}
/Cce/Psi3.dat                 Dataset {3995/Inf, 163}
/Cce/Psi4.dat                 Dataset {3995/Inf, 163}
/Cce/Strain.dat               Dataset {3995/Inf, 163}
/src.tar.gz               Dataset {3750199}
```

### Raw CCE output

The `Strain` represents the asymptotic transverse-traceless contribution
to the metric scaled by the Bondi radius (to give the asymptotically leading
part), the `News` represents the first time derivative of the strain, and each
of the `Psi...` datasets represent the Weyl scalars, each scaled by the
appropriate factor of the Bondi-Sachs radius to retrieve the asymptotically
leading contribution.

The `EthInertialRetardedTime` is a diagnostic dataset that represents the
angular derivative of the inertial retarded time, which determines the
coordinate transformation that is performed at future null infinity.

If you'd like to visualize the output of a CCE run, we offer a
[CLI](py/cli.html) that will produce a plot of all of quantities except
`EthInertialRetardedTime`. To see how to use this CLI, run

```
spectre plot cce -h
```

If you'd like to do something more complicated than just make a quick plot,
you'll have to load in the output data yourself using `h5py` or our
`spectre.IO.H5` bindings.

\note The CLI can also plot the "old" version of CCE output, described above.
Pass `--cce-group %Cce` to the CLI. This option is only for backwards
compatibility with the old CCE output and is not supported for the current
version of output. This options is deprecated and will be removed in the future.

### Frame fixing

You may notice some odd features in some of the output quantities if you try and
plot them. This is not a bug in CCE (not that we know of at least). This occurs
because the data at future null infinity is in the wrong Bondi-Metzner-Sachs
(BMS) frame. In order to put this data into the correct BMS frame, the
[SXS Collaboration](https://github.com/sxs-collaboration) offers a python/numba
code called [scri](https://github.com/moble/scri) to do these transformations.
See their documentation for how to install/run/plot a waveform at future null
infinity in the correct BMS frame.

Below is a plot of the imaginary part of the 2,2 component for the strain when
plotted using the raw output from SpECTRE CCE and BMS frame-fixing with scri.
You'll notice that there is a non-zero offset, and this component of the strain
doesn't decay back down to zero during the ringdown. This is because of the
improper BMS frame that the SpECTRE CCE waveform is in. A supertranslation must
be applied to transform the waveform into the correct BMS frame. See
\cite Mitman2024review for a review of BMS transformations and gravitational
memory. To perform the frame fixing, see
[this tutorial](https://scri.readthedocs.io/en/latest/tutorial_abd.html) in
scri.

\image html im_h22.png "Imaginary part of 2,2 component of the strain"

The discrepancy is even more apparent if you plot the amplitude of the two
waveforms. It's pretty clear which waveform is the more "physical" one.

\image html amp_im_h22.png "Amplitude of imaginary part of 2,2 component of the strain"

Notice that there are still some oscillations in the strain output by scri
towards the beginning of the waveform (up to ~1500M). This is caused by CCE junk
radiation from imperfect initial data on the null hypersurface. In order to use
this waveform in analysis, the CCE junk must be cut off from the beginning.

You are also able to see gravitational memory effects with SpECTRE CCE! This
shows up in the real part of the 2,0 mode of the strain. Though you can see the
memory effects in the SpECTRE CCE waveform, in order to do any analysis, you
must also transform the waveform to a more physically motivated BMS frame.

\image html re_h20.png "Real part of 2,0 component of the strain

## Citing CCE

If you use the SpECTRE CCE module to extract your waveforms at future null
infinity, please cite the following:

- [SpECTRE DOI](https://zenodo.org/doi/10.5281/zenodo.4290404) (This link
  defaults to the latest release of SpECTRE. From there, you can find links to
  past releases of SpECTRE as well)
- [SpECTRE CCE paper](https://doi.org/10.1103/PhysRevD.107.064013)
- [CCE paper](https://doi.org/10.1103/PhysRevD.102.044052)

If you used scri to perform any frame-fixing, please also consult the
[scri GitHub](https://github.com/moble/scri) for how you should cite it.

You can also consult our \ref publication_policies page for further questions or
contact [spectre-devel@black-holes.org](mailto:spectre-devel@black-holes.org).
