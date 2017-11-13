\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Visualization {#tutorial_visualization}

### 1D %Data Volume %Data

Volume data from one dimensional simulations is visualized using the python
script `Support/bin/Render1dData.py`. Run
`python /path/to/Render1dData.py --help` to view the help text for the script.
Let us consider a Burgers evolution where we want to visualize the solution,
`U`, then in the run directory you can use

### 2D & 3D %Data Volume %Data In ParaView

Once you have generated volume data you must run the
executable `ApplyObservers` in the run directory to generate an XDMF
(`.xmf` extension) file. Open the `VisVolumeData.xmf` file in ParaView
and selet the `Xdmf Reader`, *not* the version 3 readers. On the left hand side
of the main ParaView window is a section named `Properties`, here you must
click the highlighted `Apply` button. ParaView will now render your volume data.
You may notice what appears to be mesh imprinting on the data. The imprinting
effect can be removed by applying the `Tetrahedralize` filter. To apply the
filter select the `Filters` menu item, then `Alphabetical` and finally
`Tetrahedralize`.

