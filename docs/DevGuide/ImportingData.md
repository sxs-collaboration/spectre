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
with the importer, and another for loading the data. Here's an example for such
`Metavariables`:

\snippet Test_VolumeDataReaderAlgorithm.hpp metavars

To register elements of your array parallel component for receiving volume data,
invoke the `importers::Actions::RegisterWithVolumeDataReader` action in the
`phase_dependent_action_list`. Here's an example:

\snippet Test_VolumeDataReaderAlgorithm.hpp register_action

Now you're all set to load a data file. To do so, invoke the
`importers::ThreadedActions::ReadVolumeData` action on the
`importers::VolumeDataReader`. A good place for this is the `execute_next_phase`
function of the array parallel component:

\snippet Test_VolumeDataReaderAlgorithm.hpp read_data_action

\snippet Test_VolumeDataReaderAlgorithm.hpp invoke_readvoldata

In the first snippet the `read_element_data_action` type alias is set to a
specialization of the `importers::ThreadedActions::ReadVolumeData` template.
Its first template parameter specifies an option group that determines the data
file to load, the second parameter is a typelist of the tags to import and fill
with the volume data, the third parameter is the action that is called upon
receiving the data on the array elements, and the last template parameter is the
parallel component on which the callback action will be invoked. See the
documentation of the `importers::ThreadedActions::ReadVolumeData` action for
details on these parameters. Here is more information on the parameters used in
the example above:

- The `VolumeDataOptions` in this example is an option group that supplies
  information such as the file name. You provide an option group that represents
  the data you want to import. For example, we have the
  `evolution::OptionTags::NumericInitialData` that represents numeric initial
  data for an evolution. In our example, we created a new class like this:

  \snippet Test_VolumeDataReaderAlgorithm.hpp option_group

  This results in a section in the input file that may look like this:

  \snippet Test_VolumeDataReaderAlgorithm2D.yaml importer_options

- The `::Actions::SetData` callback action in the example will be invoked to
  receive the data on each element. You can use this particular action to move
  the received data directly into the DataBox, but it's generally better to
  write a new action that does a few consistency checks on the data before
  moving it into the DataBox.

When the `importers::ThreadedActions::ReadVolumeData` is invoked, the data file
will be read and its data distributed to all elements of the array that have
been registered.
