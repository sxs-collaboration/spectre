\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Versioning and releases {#versioning_and_releases}

\tableofcontents

# Version format

We employ a date-based versioning scheme with the format `YYYY.0M.0D[.TWEAK]`,
where `YYYY` is the year, `0M` the zero-padded month and `0D` the zero-padded
day of the month when the version is released. If additional versions are
released on a single day, the `TWEAK` component enumerates them. For example,
the first version released on May 4th, 2021 carries the name `2021.05.04`.
Additional versions released on May 4th, 2021 would carry the names
`2021.05.04.1`, `2021.05.04.2`, etc.

The reason we use a date-based versioning scheme over, for example, semantic
versioning is to avoid implying a notion of compatibility between releases (see
section below).

# Guarantees and non-guarantees of releases

Each release is **guaranteed** to pass all automated tests laid out in the
\ref travis_guide "guide on automated tests".

We provide **no guarantee** that releases satisfy any notion of "compatibility"
with each other. In particular, the interface of any C++ library, Python module
or executable may change between releases. When writing code for SpECTRE we
recommend regularly rebasing on the latest `develop` branch of the
[`sxs-collaboration/spectre`](https://github.com/sxs-collaboration/spectre)
repository and aiming to contribute your code upstream by following our \ref
contributing_to_spectre "contributing guide" in a timely manner. That said, we
make an effort to retain compatibility between releases for the following
aspects of the code:

- Simulation data: We try to ensure that data produced by a release remains
  compatible with future releases, in the sense that the data can be read,
  processed or converted to newer formats.
- Python bindings: We try to offer deprecation warnings when changing the Python
  interfaces so developers of external packages that use the Python bindings
  have time to update their packages.

We may establish guarantees for specific interfaces related to these aspects.
These guarantees will be defined in the documentation of the respective
interface.

We explicitly provide **no guarantee** that input files remain compatible with
executables built on different releases. Instead, we try to make input files as
explicit as possible by not supporting default values, so simulations don't
continue with subtly changed parameters part way through. We try to reflect
changes that would affect running simulations in the input files, so they can
(and often must) be reviewed before the simulation continues. Please note that
low-level changes or bugs may always alter results, so be advised to stick with
a particular compiled executable if a high level of reproducibility is crucial
for your project.

# Schedule

We typically publish a release at the beginning of each month, and may publish
additional releases irregularly. If a major bug is discovered and fixed we will
likely create an unscheduled release.

# How to find releases

You can find all releases on GitHub:

- [sxs-collaboration/spectre releases on GitHub](https://github.com/sxs-collaboration/spectre/releases)

Each release is tagged with the version name and prefixed `v` in the repository.
For example, this is how you can obtain the source code for the `2020.12.07`
release:

```sh
git clone -b v2020.12.07 https://github.com/sxs-collaboration/spectre
```

The latest release is always available on the `release` branch. This is how you
can obtain the source code for the latest release:

```sh
git clone -b release https://github.com/sxs-collaboration/spectre
```
