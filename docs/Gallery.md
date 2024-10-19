\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

\cond NEVER
Instructions for adding images or videos:

1. Send your media files to a SpECTRE core dev (see CONTRIBUTING.md for details
   and communication channels). They will place the files in this public Google
   Drive folder:
   https://drive.google.com/drive/folders/1xvLplF_pPlkADGfgFNaW_ByVATFv61m-
2. Open the media files in Google Drive and get their file IDs. To do this,
   select [...] > Open in new window > Copy the <FILE_ID> in the URL of the form
   https://drive.google.com/file/d/<FILE_ID>/view.
3. Add a section to this page.
   - Embed images like this:
     <img src="https://drive.google.com/thumbnail?id=<FILE_ID>&sz=w<WIDTH>"/>
     Replace <WIDTH> with the width you want your image to have on the page.
     Google Drive will automatically provide an image of this size.
   - Embed videos like this:
     \htmlonly
     <iframe src="https://drive.google.com/file/d/18KIEoK8oH6WDifx0cyFkl2ncKijNtaxs/preview" width="640" height="480" allow="autoplay" allowfullscreen></iframe>
     \endhtmlonly
4. Open a pull request to contribute your changes (see CONTRIBUTING.md for
   details).
\endcond

# Gallery {#gallery}

This page highlights some visualizations of SpECTRE simulations.

\note We're always happy to feature images and videos created by the SpECTRE
community on this page. See [this page on
GitHub](https://github.com/sxs-collaboration/spectre/blob/develop/docs/Gallery.md)
for instructions.

## Binary black hole mergers {#gallery_bbh}

SpECTRE can simulate black holes that orbit each other and eventually merge,
emitting gravitational waves.
SpECTRE uses the Generalized Harmonic formulation of
Einstein's equations of general relativity to solve this problem. Since we
expect the solution of Einstein's equations to be smooth for the BBH problem,
we represent our solution using the Discontinuous Galerkin (DG) method because
of its ability to represent smooth functions to high accuracy. Also, DG allows
SpECTRE to parallelize the BBH problem.

Publications:

- Lovelace, Nelli, et al. (2024). Simulating binary black hole mergers using
  discontinuous Galerkin methods. \cite Lovelace2024wra

\htmlonly
<figure>
<img src="https://drive.google.com/thumbnail?id=1Ccz_ZDPTpIsLXh_lGotGBdSLEnc7a__f&sz=w1200"/>
<figcaption>
Apparent horizons of an equal mass non-spinning BBH. The
colorful surface represents the lapse value in the equatorial plane and the
arrows depict the shift vector. Image credit: Alex Carpenter, CSUF
</figcaption>
</figure>
\endhtmlonly

\htmlonly
<figure>
<img src="https://drive.google.com/thumbnail?id=1Hf1lkmm9oqZ27Qhbi-McUMbSuP4yR7FW&sz=w800"/>
<figcaption>
Overview of a binary black hole simulation in SpECTRE. Upper left panel:
computational grid and shape of black hole horizons during merger. A common
horizon has formed (blue) that envelops the two original horizons (black). Upper
right panel: trajectories of the black holes until merger.
This inspiral is 18 orbits long and approximately circular.
Bottom panel: gravitational waveform extracted with CCE (see below).
Image credit: Geoffrey Lovelace, CSUF
</figcaption>
</figure>
\endhtmlonly

## Binary black hole initial data

SpECTRE can generate initial data to start simulations of merging black holes.
This problem involves solving the elliptic constraint sector of the Einstein
equations for a slice of spacetime that contains two black holes with the
requested parameters. SpECTRE uses the \ref ::Xcts "XCTS" formulation with a
non-conformally-flat background defined by the
\ref ::Xcts::AnalyticData::Binary "superposed"
\ref gr::Solutions::KerrSchild "Kerr-Schild" formalism
to reach high spins. Black holes are represented by excisions and
\ref ::Xcts::BoundaryConditions::ApparentHorizon "boundary conditions".

Publications:

- Vu et al. (2022). A scalable elliptic solver with task-based parallelism for
  the SpECTRE numerical relativity code. \cite Vu2021coj
- Vu (2024). A discontinuous Galerkin scheme for elliptic equations on
  extremely stretched grids. \cite Vu2024cgf

\htmlonly
<figure>
<img src="https://drive.google.com/thumbnail?id=1lruXf3G4KCZ57CSjkuTE6B-bGPQY1ZyQ&sz=w800"/>
<figcaption>
An initial slice of spacetime containing two black holes in orbit around each
other. Shown is the lapse variable. The two black holes are represented as
boundary conditions on excised regions of the computational domain.
Image credit: Nils Vu, Caltech
</figcaption>
</figure>
\endhtmlonly

## Cauchy-characteristic evolution (CCE) {#gallery_cce}

SpECTRE implements a novel Cauchy-characteristic evolution (CCE) system for
extracting gravitational waveforms from our simulations. It evolves the
Einstein equations on null slices to infinity, which is more accurate than
extrapolation and allows us to extract the gravitational memory effect.
The CCE waveform extraction is publicly available as a standalone module.

Tutorial:

- \ref tutorial_cce

Publications:

- Moxon et al. (2023). SpECTRE Cauchy-characteristic evolution system for rapid,
  precise waveform extraction. \cite Moxon2021gbv
- Moxon et al. (2020). Improved Cauchy-characteristic evolution system for
  high-precision numerical relativity waveforms. \cite Moxon2020gha

## Binary neutron star mergers

SpECTRE can simulate merging neutron stars and other general-relativistic
magneto-hydrodynamic (GRMHD) problems with dynamic gravity. Our DG-FD hybrid
scheme accelerates smooth regions of the grid with high-order spectral methods
(see \ref gallery_dgfd).

Publications:

- Deppe et al. (2024). Binary neutron star mergers using a discontinuous
  Galerkin-finite difference hybrid method. \cite Deppe2024ckt

\htmlonly
<figure>
<iframe src="https://drive.google.com/file/d/1kLx00pjDBmD49bpUcfHRcZNXR9jE8Sxb/preview" width="640" height="480" allow="autoplay" allowfullscreen></iframe>
<figcaption>
Simulation of two merging neutron stars. The colors show density contours.
Video credit: Nils Vu, Caltech
</figcaption>
</figure>
\endhtmlonly

## Curved and moving meshes with control systems

Our computational domains in SpECTRE are designed to adapt to the geometry of
the problems we want to solve. They can be curved, e.g. to wrap around excision
regions in binary black hole problems (see \ref gallery_bbh) or to resolve the
wavezone in binary neutron star merger. They can also rotate and deform in time
using control systems, which reactively adjust coordinate maps to track the
position and shape of the black hole excisions or neutron stars.

## Adaptive mesh refinement

Our discontinuous Galerkin methods allow two types of mesh refinement: splitting
elements in half along any dimension (h-refinement) or increasing their
polynomial expansion order (p-refinement). The former allows us to distribute
computational cost to supercomputers, while the latter allows us to use these
resources efficiently by decreasing the numerical error exponentially with the
number of grid points where the solution is smooth. Our adaptive mesh refinement
technology decides which type of refinement to apply in each region of the
domain.

## DG-FD hybrid method {#gallery_dgfd}

Our hydrodynamical simulations use a discontinuous Galerkin-finite difference
(DG-FD) hybrid method: smooth regions of the simulation are evolved with an
efficient DG scheme and non-smooth regions fall back to a robust FD method.
Shocks and discontinuities on the grid are tracked with a troubled-cell
indicator (TCI) to switch between DG and FD. This approach accelerates our
simulations by reducing the computational resources spent on smooth regions of
the grid, e.g. when evolving inspiral binary neutron stars and their
gravitational radiation.

Publications:

- Deppe et al. (2022). A high-order shock capturing discontinuous
  Galerkin-finite difference hybrid method for GRMHD. \cite Deppe2021ada
- Deppe et al. (2022). Simulating magnetized neutron stars with discontinuous
  Galerkin methods. \cite Deppe2021bhi

\htmlonly
<figure>
<iframe src="https://drive.google.com/file/d/1gSRfo2-V6HLK1XwpZ-twrp4f-4G8n08V/preview" width="640" height="480" allow="autoplay" allowfullscreen></iframe>
<figcaption>
Simulation of the Kelvin-Helmholtz instability (KHI). Squares indicate cells
that have switched to a finite-difference method. They track shocks and
discontinuities in the solution. The rest of the domain uses an efficient
DG method.
Video credit: Nils Deppe, Cornell University
</figcaption>
</figure>
\endhtmlonly

## Binary black holes in scalar Gauss-Bonnet gravity

SpECTRE can generate initial data for binary black holes in scalar Gauss-Bonnet
gravity, evolve the modified Einstein equations, and extract the gravitational
and scalar radiation.

Publications:

- Nee et al. (2024). Quasistationary hair for binary black hole initial data in
  scalar Gauss-Bonnet gravity \cite Nee2024bur
- Lara et al. (2024). Scalarization of isolated black holes in scalar
  Gauss-Bonnet theory in the fixing-the-equations approach. \cite Lara2024rwa

\htmlonly
<figure>
<img src="https://drive.google.com/thumbnail?id=1uhmrprFOE9xOWaUQntgE_4J9DgniFBLe&sz=w800"/>
<figcaption>
Binary black hole initial data in scalar Gauss-Bonnet gravity, in a
configuration where the two black holes have opposite charge.
The scalar field is solved such that it is in equilibrium with the gravity
background, minimizing initial transients in the evolution and giving control
over the simulation parameters.
Image credit: Peter James Nee, MPI for Gravitational Physics Potsdam, Germany
</figcaption>
</figure>
\endhtmlonly

## Thermal noise in gravitational wave detectors

We have applied the SpECTRE technology to an interdisciplinary problem,
simulating the Brownian thermal noise in the mirrors of interferometric
gravitational-wave detectors at unprecedented accuracy. It uses the SpECTRE
elliptic solver \cite Vu2021coj to solve an elasticity problem, which connects
to the thermal noise problem through the fluctuation dissipation theorem.

Publications:

- Vu et al. (2024). High-accuracy numerical models of Brownian thermal noise in
  thin mirror coatings. \cite Vu2023thn
