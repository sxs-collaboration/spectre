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

## Adaptive mesh refinement

Here's an image of an adaptively refined mesh:

<img src="https://drive.google.com/thumbnail?id=1ivR2Wbu_RYvHih098waKi6jqx637pSUn&sz=w600"/>

## Binary Black Holes

### Overview

Simulations of two black holes orbiting each other and eventually merging using
the SpECTRE code. SpECTRE uses the Generalized Harmonic formulation of
Einstein's equations of general relativity to solve this problem. Since we
expect the solution of Einstein's equations to be smooth for the BBH problem,
we represent our solution using the Discontinuous Galerkin (DG) method because
of it's ability to represent smooth functions to high accuracy. Also, DG allows
SpECTRE to parallelize the BBH problem to exascale.

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
