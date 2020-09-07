\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Importing data {#dev_guide_importing}

The `importers` namespace holds functionality for importing data into SpECTRE.
We currently support loading volume data files in the same format that is
written by the `observers`.

## Importing volume data

The `importers::VolumeDataReader` parallel component is responsible for loading
volume data and distributing it to elements of one or multiple array parallel
components. As a first step, make sure you have added the
`importers::VolumeDataReader` to your `Metavariables::component_list`. Also make
sure you have a `Metavariables::Phase` in which you will perform registration
with the importer, and another in which you want to load the data. Here's an
example for such `Metavariables`:

\snippet Test_VolumeDataReaderAlgorithm.hpp metavars

To load volume data from a file, invoke the `importers::Actions::ReadVolumeData`
iterable action in the `phase_dependent_action_list` of your array parallel
component. Here's an example that will be explained in more detail below:

\snippet Test_VolumeDataReaderAlgorithm.hpp import_actions

- The `importers::Actions::ReadVolumeData` action will load the volume data file
  once per node on its first invocation by dispatching to
  `importers::Actions::ReadAllVolumeDataAndDistribute` on the
  `importers::VolumeDataReader` nodegroup component. Subsequent invocations of
  these actions, e.g. from all other elements on the node, will do nothing. The
  data is distributed into the inboxes of all elements on the node under the
  `importers::Tags::VolumeData` tag using `Parallel::receive_data`.
- The `importers::Actions::ReceiveVolumeData` action waits for the volume data
  to be available and directly moves it into the DataBox. If you wish to verify
  or post-process the data before populating the DataBox, use your own
  specialized action in place of `importers::Actions::ReceiveVolumeData`.
- You need to register the elements of your array parallel component for
  receiving volume data. To do so, invoke the
  `importers::Actions::RegisterWithVolumeDataReader` action in an earlier phase,
  as shown in the example above.

The template parameters to the actions in the example above specify the volume
data to load. The first template parameter is an option group that determines
the file to read and the second parameter is a typelist of the tags to import
and fill with the volume data. See the documentation of the
`importers::Actions::ReadAllVolumeDataAndDistribute` action for details on these
parameters. In the example above, the `VolumeDataOptions` is an option group
that supplies information such as the file name. You provide an option group
that represents the data you want to import. For example, we have the
`evolution::OptionTags::NumericInitialData` that represents numeric initial data
for an evolution. In our example, we created a new class like this:

\snippet Test_VolumeDataReaderAlgorithm.hpp option_group

This results in a section in the input file that may look like this:

\snippet Test_VolumeDataReaderAlgorithm2D.yaml importer_options
