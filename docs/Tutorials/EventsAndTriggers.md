\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Events and triggers {#tutorial_events_and_triggers}

The "events and triggers" infrastructure (and the related "events and
dense triggers") provide some control of SpECTRE executable execution
from the input file contents by running user-specified code on the
elements.  It does not generally have major effects on the flow of the
\ref dev_guide_parallelization_core_algorithm "algorithm", but is
primarily used for observations, such as writing volume data to H5
files, and minor simulation adjustments.

### Events

An \ref Event "event" is an input-file-creatable object that performs
some task that (with the exception of the \ref Events::Completion
"Completion" event) does not directly affect the execution of the
algorithm on the element.  The effects of events are limited to
sending messages.  The most commonly used events (such as the \ref
dg::Events::ObserveErrorNorms "ObserveErrorNorms" event) send data to
be written to disk, but have no long-term effects on the simulation
state.  Others (such as \ref Events::ChangeSlabSize "ChangeSlabSize")
can have indirect effects when \ref dev_guide_parallelization_actions
"actions" react to the messages they send.

### Triggers

A \ref Trigger "trigger" is an input-file-creatable object that
controls when events are run.  They are checked periodically, once per
Slab in an evolution executable, and once per iteration in an elliptic
solve.  At each check a trigger may fire, causing its associated
events to run or not.  Triggers must give a consistent result over
the entire domain at each check, so their result must always be
independent of element-specific state, such as the local values of the
system variables.

### Dense triggers

A \ref DenseTrigger "dense trigger" is similar to a trigger, but uses
time-stepper dense output (i.e., interpolation or extrapolation) to
provide more precise control of the time that it fires.  As a
time-related feature, dense triggers are only applicable to evolution
executables.  Dense triggers fire at a precise set of times,
independent of the time-step or slab state, by interpolating the
evolved variables to the requested time before running the associated
events.  As with normal triggers, dense triggers are required to fire
consistently over the entire domain.

### Input file syntax

The `%EventsAndTriggers` (and `%EventsAndDenseTriggers`) sections of
the \ref dev_guide_option_parsing "input file" indicate structure
using a collection of `?`, `:`, and `-` characters.  These are
standard, if perhaps somewhat obscure, YAML syntax.  They are defining
a map from triggers to lists of events.  The `?:` syntax defines an
expanded key-value pair, which allows the key to be a complicated
type, unlike the common `Key: Value` syntax.  The `-` is the expanded
form of a list, which allows entries to have complicated types.
Frequently only one event is associated with a trigger, so there will
only be a single `-` after the `:`, but additional events introduced
by more `-` characters can appear afterwards:

\snippet PlaneWave1DEventsAndTriggersExample.yaml multiple_events

In this example, we are using the \ref Triggers::Slabs "Slabs" trigger
to run two events every 10 slabs: ObserveFields and \ref
dg::Events::ObserveErrorNorms "ObserveErrorNorms".

If an event and trigger both took no options, the syntax

\snippet PlaneWave1DEventsAndTriggersExample.yaml no_arguments

could be written in the more familiar form

\snippet PlaneWave1DEventsAndTriggersExample.yaml no_arguments_single_line

but we recommend always using the expanded form for consistency.
