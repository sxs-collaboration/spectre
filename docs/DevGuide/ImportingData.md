\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Importing data {#dev_guide_importing}

\tableofcontents

The `importers` namespace holds functionality for importing data into SpECTRE.
We currently support loading volume data files in the same format that is
written by the `observers`.

## Importing volume data

The `importers::ElementDataReader` parallel component is responsible for loading
volume data, interpolating it, and distributing it to elements of one or
multiple array parallel components. As a first step, make sure you have added
the `importers::ElementDataReader` to your `Metavariables::component_list`. Also
make sure you have a `Parallel::Phase` in which you will perform
registration with the importer, and another in which you want to load the data.
Here's an example for such `Metavariables`:

\snippet Test_VolumeDataReaderAlgorithm.hpp metavars

To load volume data from a file, write an action in which you invoke
`importers::Actions::ReadAllVolumeDataAndDistribute` on the
`importers::ElementDataReader`. For simple use cases we provide
`importers::Actions::ReadVolumeData`, which can be added to the
`phase_dependent_action_list` of your element array and which will generate
input-file options for you. Here's an example that will be explained in more
detail below:

\snippet Test_VolumeDataReaderAlgorithm.hpp import_actions

- The `importers::Actions::ReadVolumeData` action specifies input-file options
  and dispatches to `importers::Actions::ReadAllVolumeDataAndDistribute` on the
  `importers::ElementDataReader` nodegroup component. It loads the volume data
  file once per node on its first invocation. Subsequent invocations of these
  actions, e.g. from all other elements on the node, will do nothing. The data
  is distributed into the inboxes of all elements on the node under the
  `importers::Tags::VolumeData` tag using `Parallel::receive_data`.
- The `importers::Actions::ReceiveVolumeData` action waits for the volume data
  to be available and directly moves it into the DataBox. If you wish to verify
  or post-process the data before populating the DataBox, use your own
  specialized action in place of `importers::Actions::ReceiveVolumeData`.
- You need to register the elements of your array parallel component for
  receiving volume data. To do so, invoke the
  `importers::Actions::RegisterWithElementDataReader` action in an earlier
  phase, as shown in the example above.

The parameters passed to `importers::Actions::ReadAllVolumeDataAndDistribute`
specify the volume data to load. See the documentation of
`importers::Actions::ReadAllVolumeDataAndDistribute` for details. In the example
above, we use `importers::Actions::ReadVolumeData` to generate the input-file
options for us and place them in an option group:

\snippet Test_VolumeDataReaderAlgorithm.hpp option_group

This results in a section in the input file that may look like this:

\snippet Test_VolumeDataReaderAlgorithm2D.yaml importer_options
