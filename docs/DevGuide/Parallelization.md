\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Parallization, Charm++, and Core Concepts {#dev_guide_parallelization_foundations}

\tableofcontents

# Introduction {#dev_guide_parallelization_introduction}

SpECTRE builds a layer on top of Charm++ that performs various safety checks and
initialization for the user that can otherwise lead to difficult-to-debug
undefined behavior. The central concept is what is called a %Parallel
Component. A %Parallel Component is a struct with several type aliases that
is used by SpECTRE to set up the Charm++ chares and allowed communication
patterns. %Parallel Components are input arguments to the compiler, which then
writes the parallelization infrastructure that you requested for the executable.
There is no restriction on the number of %Parallel Components, though
practically it is best to have around 10 at most.

Here is an overview of what is described in detail in the sections below:

- \ref dev_guide_parallelization_metavariables_class "Metavariables": Provides
  high-level configuration to the compiler, e.g. the physical system to be
  simulated.
- \ref dev_guide_parallelization_phases_of_execution "Phases": Defines distinct
  simulation phases separated by a global synchronization point,
  e.g. `Initialization`, `Evolve` and `Exit`.
- \ref dev_guide_parallelization_core_algorithm "Algorithm": In each phase,
  repeatedly iterates over a list of actions until the current phase ends.
- \ref dev_guide_parallelization_parallel_components "Parallel component":
  Maintains and executes its algorithm.
- \ref dev_guide_parallelization_actions "Action": Performs a computational
  task, e.g. evaluating the right hand side of the time evolution equations. May
  require data to be received from another action potentially being executed on
  a different core or node.

# The Metavariables Class {#dev_guide_parallelization_metavariables_class}

SpECTRE takes a different approach to input options passed to an executable than
is common. SpECTRE not only reads an input file at runtime but also has many
choices made at compile time. The compile time options are specified by what is
referred to as the metavariables. What exactly the metavariables struct
specifies depends on the executable, but all metavariables structs must
specify the following:

- `help`: a `static constexpr OptionString` that will be printed as part of the
  help message. It should describe the executable and basic usage of it, as well
  as any non-standard options that must be specified in the metavariables and
  their current values. An example of a help string for one of the testing
  executables is:
  \snippet Test_AlgorithmCore.cpp help_string_example
- `component_list`: a `tmpl::list` of the parallel components (described below)
  that are to be created. Most evolution executables will have the
  `DgElementArray` parallel component listed. An example of a `component_list`
  for one of the test executables is:
  \snippet Test_AlgorithmCore.cpp component_list_example
- `using const_global_cache_tags`: a `tmpl::list` of tags that are
  used to place items in the GlobalCache.  The alias may be
  omitted if the list is empty.
- `Phase`: an `enum class` that must contain at least `Initialization` and
  `Exit`. Phases are described in the next section.
- `determine_next_phase`: a static function with the signature
  \code
    static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>& cache_proxy)
      noexcept;
  \endcode
  What this function does is described below in the discussion of
  \ref dev_guide_parallelization_phases_of_execution "phases".

There are also several optional members:

- `input_file`: a `static constexpr OptionString` that is the default name of
  the input file that is to be read. This can be overridden at runtime by
  passing the `--input-file` argument to the executable.
- `ignore_unrecognized_command_line_options`: a `static constexpr bool` that
  defaults to `false`. If set to `true` then unrecognized command line options
  are ignored. Ignoring unrecognized options is generally only necessary for
  tests where arguments for the testing framework,
  [Catch](https://github.com/catchorg/Catch2/), are passed to the executable.

# Phases of an Execution {#dev_guide_parallelization_phases_of_execution}

Global synchronization points, where all cores wait for each other, are
undesirable for scalability reasons. However, they are sometimes inevitable for
algorithmic reasons. That is, in order to actually get a correct solution you
need to have a global synchronization. SpECTRE executables can have multiple
phases, where after each phase a global synchronization occurs. By global
synchronization we mean that no parallel components are executing or have more
tasks to execute: everything is waiting to be told what tasks to perform next.

Every executable must have at least two phases, `Initialization` and
`Exit`. The next phase is decided by the static member function
`determine_next_phase` in the metavariables. Currently this function has access
to the phase that is
ending, and also the global cache. In the future we will add support for
receiving data from various components to allow for more complex decision
making. Here is an example of a `determine_next_phase` function and the `Phase`
enum class:
\snippet Test_AlgorithmCore.cpp determine_next_phase_example

In contrast, an evolution executable might have phases
`Initialization`, `SetInitialData`, `Evolve`, and `Exit`, but have a
similar `switch` or `if-else` logic in the `determine_next_phase`
function. The first phase that is entered is always
`Initialization`. During the `Initialization` phase the
`Parallel::GlobalCache` is created, all non-array components are created,
and empty array components are created.  Next, the function
`allocate_array_components_and_execute_initialization_phase` is called
which allocates the elements of each array component, and then starts
the `Initialization` phase on all parallel components. Once all
parallel components' `Initialization` phase is complete, the next
phase is determined and the `execute_next_phase` function is called on
all the parallel components.

At the end of an execution the `Exit` phase has the executable wait to make sure
no parallel components are performing or need to perform any more tasks, and
then exits. An example where this approach is important is if we are done
evolving a system but still need to write data to disk. We do not want to exit
the simulation until all data has been written to disk, even though we've
reached the final time of the evolution.

\warning Currently dead-locks are treated as successful termination. In the
future checks against deadlocks will be performed before terminating.

# The Algorithm {#dev_guide_parallelization_core_algorithm}

Since most numerical algorithms repeat steps until some criterion such as the
final time or convergence is met, SpECTRE's parallel components are designed to
do such iterations for the user. An Algorithm executes an ordered list of
actions until one of the actions cannot be evaluated, typically because it is
waiting on data from elsewhere. When an algorithm can no longer evaluate actions
it passively waits by handing control back to Charm++. Once an algorithm
receives data, typically done by having another parallel component call the
`receive_data` function, the algorithm will try again to execute the next
action. If the algorithm is still waiting on more data then the algorithm will
again return control to Charm++ and passively wait for more data. This is
repeated until all required data is available. The actions that are iterated
over by the algorithm are called iterable actions and are described below.
Since the action list is phase dependent we refer to them generally as
phase-dependent action lists (PDALs, pronounced "pedals").

# Parallel Components {#dev_guide_parallelization_parallel_components}

Each %Parallel Component struct must have the following type aliases:
1. `using chare_type` is set to one of:
   1. `Parallel::Algorithms::Singleton`s have one object in the entire execution
      of the program.
   2. `Parallel::Algorithms::Array`s hold zero or more elements, each of which
      is an object distributed to some core. An array can grow and shrink in
      size dynamically if need be and can also be bound to another array. A
      bound array has the same number of elements as the array it is bound to,
      and elements with the same ID are on the same core. See Charm++'s chare
      arrays for details.
   3. `Parallel::Algorithms::Group`s are arrays with
      one element per core which are not able to be moved around between
      cores. These are typically useful for gathering data from array elements
      on their core, and then processing or reducing the data further. See
      [Charm++'s](http://charm.cs.illinois.edu/help) group chares for details.
   4. `Parallel::Algorithms::Nodegroup`s are similar to
      groups except that there is one element per node. For Charm++ SMP (shared
      memory parallelism) builds, a node corresponds to the usual definition of
      a node on a supercomputer. However, for non-SMP builds nodes and cores are
      equivalent. We ensure that all entry method calls done through the
      Algorithm's `simple_action` and `receive_data` functions are
      threadsafe. User controlled threading is possible by calling the non-entry
      method member function `threaded_action`.
2. `using metavariables` is set to the Metavariables struct that stores the
   global metavariables. It is often easiest to have the %Parallel
   Component struct have a template parameter `Metavariables` that is the
   global metavariables struct. Examples of this technique are given below.
3. `using phase_dependent_action_list` is set to a `tmpl::list` of
   `Parallel::PhaseActions<PhaseType, Phase, tmpl::list<Actions...>>`
   where each `PhaseAction` represents a PDAL that will be executed on
   the parallel component during the specified phase. The %Actions are
   executed in the order that they are given in the `tmpl::list`s of
   the PDALs, but the phases need not be run in linear order. However,
   `db::DataBox` types are constructed assuming the phases are
   performed from first in the `phase_dependent_action_list` to the
   last. Simple actions (described below) can be executed in any
   phase. If there are no iterable actions in a phase then a
   `PhaseAction` need not be specified for that phase. However, at
   least one `PhaseAction`, even if it is empty, must be specified.
4. `using initialization_tags` which is a `tmpl::list` of all the tags
   that will be inserted into the initial `db::DataBox` of each component.
   These tags are db::SimpleTag%s that have have a `using option_tags`
   type alias and a static function `create_from_options` (see the
   example below).  This list can usually be constructed from the
   initialization actions of the component (i.e. the list of actions
   in the `PhaseAction` list for the `Initialization` phase) using the
   helper function `Parallel::get_initialization_tags` (see the
   examples of components below).  Each initialization action may
   specify a type alias `using initialization_tags` which are a
   `tmpl::list` of tags that will be fetched from the db::DataBox by the
   action.  All `initialization_tags` are removed from the db::DataBox of
   the component at the end of the `Initialization` phase, except for
   tags listed in a type alias `using initialization_tags_to_keep` that
   may appear in each initialization action.
5. `using const_global_cache_tags` is set to a `tmpl::list` of tags
   that are required by the `allocate_array` function of an array
   component, or simple actions called on the parallel component.
   These tags correspond to items that are stored in the
   Parallel::GlobalCache (of which there is one copy per Charm++
   node).  The alias can be omitted if the list is empty.  (See
   `array_allocation_tags` below for specifying tags needed for the
   `allocate_array` function, but will not be added to the
   Parallel::GlobalCache.)

\note Array parallel components must also specify the type alias `using
array_index`, which is set to the type that indexes the %Parallel Component
Array. Charm++ allows arrays to be 1 through 6 dimensional or be indexed by a
custom type. The Charm++ provided indexes are wrapped as
`Parallel::ArrayIndex1D` through `Parallel::ArrayIndex6D`. When writing custom
array indices, the [Charm++ manual](http://charm.cs.illinois.edu/help) tells you
to write your own `CkArrayIndex`, but we have written a general implementation
that provides this functionality (see `Parallel::ArrayIndex`); all that you need
to provide is a plain-old-data
([POD](http://en.cppreference.com/w/cpp/concept/PODType)) struct of the size of
at most 3 integers.

%Parallel array components have a static `allocate_array` function
that is used to construct the elements of the array. The
signature of the `allocate_array` functions must be:
\code
static void allocate_array(
    Parallel::CProxy_GlobalCache<metavariables>& global_cache,
    const tuples::tagged_tuple_from_typelist<initialization_tags>&
    initialization_items) noexcept;
\endcode
The `allocate_array` function is called by the Main parallel component
when the execution starts and will typically insert elements into
array parallel components. If the `allocate_array` function depends
upon input options, the array component must specify a `using
array_allocation_tags` type alias that is a `tmpl::list` of tags which
are db::SimpleTag%s that have have a `using option_tags` type alias
and a static function `create_from_options`. An example is:
\snippet DistributedLinearSolverAlgorithmTestHelpers.hpp array_allocation_tag

The `allocate_array` functions of different
array components are called in random order and so it is not safe to
have them depend on each other.

Each parallel component must also decide what to do in the different phases of
the execution. This is controlled by an `execute_next_phase` function with
signature:
\code
static void execute_next_phase(
    const typename metavariables::Phase next_phase,
    const Parallel::CProxy_GlobalCache<metavariables>& global_cache);
\endcode
The `determine_next_phase` function in the Metavariables determines the next
phase, after which the `execute_next_phase` function gets called. The
`execute_next_phase` function determines what the parallel component should do
during the next phase. Typically the `execute_next_phase` function should just
call `start_phase(phase)` on the parallel component. In the future
`execute_next_phase` may be removed.

An example of a singleton parallel component is:
\snippet Test_AlgorithmParallel.cpp singleton_parallel_component

An example of an array parallel component is:
\snippet Test_AlgorithmParallel.cpp array_parallel_component
Elements are inserted into the array by using the Charm++ `insert` member
function of the CProxy for the array. The `insert` function is documented in
the Charm++ manual. In the above Array example `array_proxy` is a `CProxy` and
so all the documentation for Charm++ array proxies applies. SpECTRE always
creates empty arrays with the constructor and requires users to insert however
many elements they want and on which cores they want them to be placed. Note
that load balancing calls may result in array elements being moved.

# Actions {#dev_guide_parallelization_actions}

For those familiar with Charm++, actions should be thought of as effectively
being entry methods. They are functions that can be invoked on a remote object
(chare/parallel component) using a `CProxy` (see the [Charm++
manual](http://charm.cs.illinois.edu/help)), which is retrieved from the
Parallel::GlobalCache using the parallel component struct and the
`Parallel::get_parallel_component()` function. %Actions are structs with a
static `apply` method and come in three variants: simple actions, iterable
actions, and reduction actions. One important thing to note
is that actions cannot return any data to the caller of the remote method.
Instead, "returning" data must be done via callbacks or a callback-like
mechanism.

The simplest signature of an `apply` method is for iterable actions:
\snippet Test_AlgorithmCore.cpp apply_iterative
The return type is discussed at the end of each section describing a particular
type of action. Simple actions can have additional arguments, do not receive
the `inboxes` or `ActionList`, and take the `ParallelComponent` as an
explicit first template parameter. Reduction actions have the same signature as
simple actions except that the additional arguments must be of the types reduced
over.
The `db::DataBox` should be thought of as the member data of the parallel
component while the actions are the member functions. The combination of a
`db::DataBox` and actions allows building up classes with arbitrary member data
and methods using template parameters and invocation of actions. This approach
allows us to eliminate the need for users to work with Charm++'s interface
files, which can be error prone and difficult to use.

The Parallel::GlobalCache is passed to each action so that the
action has access to global data and is able to invoke actions on
other parallel components. The `ParallelComponent` template parameter
is the tag of the parallel component that invoked the action. A proxy
to the calling parallel component can then be retrieved from the
Parallel::GlobalCache. The remote entry method invocations are
slightly different for different types of actions, so they will be
discussed below. However, one thing that is disallowed for all actions
is calling an action locally from within an action on the same
parallel component.  Specifically,

\snippet Test_AlgorithmNestedApply1.cpp bad_recursive_call

Here `ckLocal()` is a Charm++ provided method that returns a pointer
to the local (currently executing) parallel component. See the
[Charm++ manual](http://charm.cs.illinois.edu/help) for more
information.  However, you are able to queue a new action to be
executed later on the same parallel component by getting your own
parallel component from the Parallel::GlobalCache
(`Parallel::get_parallel_component<ParallelComponent>(cache)`).  The
difference between the two calls is that by calling an action through
the parallel component you will first finish the series of actions you
are in, then when they are complete Charm++ will call the next queued
action.

Array, group, and nodegroup parallel components can have actions invoked in two
ways. First is a broadcast where the action is called on all elements of the
array:

\snippet Test_AlgorithmParallel.cpp broadcast_to_group

The second case is invoking an action on a specific array element by using the
array element's index. The below example shows how a broadcast would be done
manually by looping over all elements in the array:

\snippet Test_AlgorithmParallel.cpp call_on_indexed_array

Note that in general you will not know what all the elements in the array are
and so a broadcast is the correct method of sending data to or invoking an
action on all elements of an array parallel component.

The `array_index` argument passed to all `apply` methods is the index into the
parallel component array. If the parallel component is not an array the value
and type of `array_index` is implementation defined and cannot be relied on. The
`ActionList` type is the `tmpl::list` of iterable actions in the current phase.
That is, it is equal to the `action_list` type alias in the current PDAL.

## 1. Simple Actions {#dev_guide_parallelization_simple_actions}

Simple actions are designed to be called in a similar fashion to
member functions of classes. They are the direct analog of entry
methods in Charm++ except that the member data is stored in the
`db::DataBox` that is passed in as the first argument. A simple action
must return void but can use `db::mutate` to change values of items in
the `db::DataBox` if the `db::DataBox` is taken as a non-const
reference. In some cases you will need specific items to be in the
`db::DataBox` otherwise the action won't compile. To restrict which
`db::DataBox`es can be passed you should use `Requires` in the
action's `apply` function template parameter list. For example,
\snippet Test_AlgorithmCore.cpp requires_action
where the conditional checks if any element in the parameter pack `DbTags` is
`CountActionsCalled`.

A simple action that does not take any arguments can be called using a `CProxy`
from the Parallel::GlobalCache as follows:

\snippet Test_AlgorithmCore.cpp simple_action_call

If the simple action takes arguments then the arguments must be as follows:

\snippet Test_AlgorithmNodelock.cpp simple_action_with_args

## 2. Iterable Actions {#dev_guide_parallelization_iterable_actions}

%Actions in the algorithm that are part of the current PDAL are executed one
after the other until one of them cannot be evaluated. Iterable
actions may have an `is_ready` method that returns `true` or `false` depending
on whether or not the action is ready to be evaluated. If no `is_ready` method
is provided then the action is assumed to be ready to be evaluated. The
`is_ready` method typically checks that required data from other parallel
components has been received. For example, it may check that all data from
neighboring elements has arrived to be able to continue integrating in time.
The signature of an `is_ready` method must be:

\snippet Test_AlgorithmCore.cpp is_ready_example

The `inboxes` is a collection of the tags passed to `receive_data` and are
specified in the iterable actions member type alias `inbox_tags`, which must be
a `tmpl::list`. The `inbox_tags` must have two member type aliases, a
`temporal_id` which is used to identify when the data was sent, and a `type`
which is the type of the data to be stored in the `inboxes`. The types are
typically a `std::unordered_map<temporal_id, DATA>`. In the discussed scenario
of waiting for neighboring elements to send their data the `DATA` type would be
a `std::unordered_map<TheElementId, DataSent>`. Inbox tags must also specify
a `static void insert_into_inbox()` function. For example,

\snippet Test_AlgorithmParallel.cpp int_receive_tag

For common types of `DATA`, such as a `map`, a data structure with an `insert`
function, a data structure with a `push_back` function, or copy/move assignment
that is used to insert the received data, inserters are available in
`Parallel::InboxInserters`. For example, there is
`Parallel::InboxInserters::Map` for `map` data structures. The inbox tag can
inherit publicly off the inserters to gain the required insertion capabilities:

\snippet Test_AlgorithmCore.cpp int receive tag insert

The `inbox_tags` type alias for the action is:

\snippet Test_AlgorithmParallel.cpp int_receive_tag_list

and the `is_ready` function is:

\snippet Test_AlgorithmParallel.cpp int_receive_tag_is_ready

Once all of the `int`s have been received, the iterable action is executed, not
before.

\warning
It is the responsibility of the iterable action to remove data from the inboxes
that will no longer be needed. The removal of unneeded data should be done in
the `apply` function.

Iterable actions can change the type of the db::DataBox by adding or
removing elements/tags from the db::DataBox. The only requirement is
that the last action in each PDAL returns a db::DataBox that is the
same type for each iteration. Iterable actions can also request that
the algorithm no longer be executed, and control which action in the
current PDAL will be executed next. This is all done via the return
value from the `apply` function.  The `apply` function for iterable
actions must return a `std::tuple` of one, two, or three elements. The
first element of the tuple is the new db::DataBox, which can be the
same as the type passed in or a db::DataBox with different tags.  Most
iterable actions will simply return:

\snippet Test_AlgorithmParallel.cpp return_forward_as_tuple

By returning the db::DataBox as a reference in a `std::tuple` we avoid
any unnecessary copying of the db::DataBox. The second argument is an
optional bool, and controls whether or not the algorithm is
terminated. If the bool is `true` then the algorithm is terminated, by
default it is `false`. Here is an example of how to return a
db::DataBox with the same type that is passed in and also terminate
the algorithm:

\snippet Test_AlgorithmParallel.cpp return_with_termination

Notice that we again return a reference to the db::DataBox, which is
done to avoid any copying. After an algorithm has been terminated it
can be restarted by passing `false` to the `set_terminate` method or
by calling `receive_data(..., true)`. Since the order in which
messages are received is undefined in most cases the
`receive_data(..., true)` call should be used to restart the
algorithm.

The third optional element in the returned `std::tuple` is a `size_t` whose
value corresponds to the index of the action to be called next in the
PDAL. The metafunction `tmpl::index_of<list, element>` can be used to
get an `tmpl::integral_constant` with the value of the index of the element
`element` in the typelist `list`. For example,

\snippet Test_AlgorithmCore.cpp out_of_order_action

Again a reference to the db::DataBox is returned, while the
termination `bool` and next action `size_t` are returned by value. The
metafunction call `tmpl::index_of<ActionList,
iterate_increment_int0>::%value` returns a `size_t` whose value is
that of the action `iterate_increment_int0` in the PDAL.  The indexing
of actions in the PDAL starts at `0`.

Iterable actions are invoked as part of the algorithm and so the only way
to request they be invoked is by having the algorithm run on the parallel
component. The algorithm can be explicitly evaluated in a new phase by calling
`start_phase(Phase::TheCurrentPhase)`:

\snippet Test_AlgorithmCore.cpp start_phase

Alternatively, to evaluate the algorithm without changing phases the
`perform_algorithm()` method can be used:

\snippet Test_AlgorithmParallel.cpp perform_algorithm

By passing `true` to `perform_algorithm` the algorithm will be restarted if it
was terminated.

The algorithm is also evaluated by calling the `receive_data` function, either
on an entire array or singleton (this does a broadcast), or an on individual
element of the array. Here is an example of a broadcast call:

\snippet Test_AlgorithmParallel.cpp broadcast_to_group

and of calling individual elements:

\snippet Test_AlgorithmParallel.cpp call_on_indexed_array

The `receive_data` function always takes a `ReceiveTag`, which is set in the
actions `inbox_tags` type alias as described above.  The first argument is the
temporal identifier, and the second is the data to be sent.

Normally when remote functions are invoked they go through the Charm++ runtime
system, which adds some overhead. The `receive_data` function tries to elide
the call to the Charm++ RTS for calls into array components. Charm++ refers to
these types of remote calls as "inline entry methods". With the Charm++ method
of eliding the RTS, the code becomes susceptible to stack overflows because
of infinite recursion. The `receive_data` function is limited to at most 64 RTS
elided calls, though in practice reaching this limit is rare. When the limit is
reached the remote method invocation is done through the RTS instead of being
elided.

## 3. Reduction Actions {#dev_guide_parallelization_reduction_actions}

Finally, there are reduction actions which are used when reducing data over an
array. For example, you may want to know the sum of a `int` from every
element in the array. You can do this as follows:

\snippet Test_AlgorithmReduction.cpp contribute_to_reduction_example

This reduces over the parallel component
`ArrayParallelComponent<Metavariables>`, reduces to the parallel component
`SingletonParallelComponent<Metavariables>`, and calls the action
`ProcessReducedSumOfInts` after the reduction has been performed. The reduction
action is:

\snippet Test_AlgorithmReduction.cpp reduce_sum_int_action

As you can see, the last argument to the `apply` function is of type `int`, and
is the reduced value.

You can also broadcast the result back to an array, even yourself. For example,

\snippet Test_AlgorithmReduction.cpp contribute_to_broadcast_reduction

It is often necessary to reduce custom data types, such as `std::vector` or
`std::unordered_map`. Charm++ supports such custom reductions, and so does our
layer on top of Charm++.
Custom reductions require one additional step to calling
`contribute_to_reduction`, which is writing a reduction function to reduce the
custom data. We provide a generic type that can be used in custom reductions,
`Parallel::ReductionData`, which takes a series of `Parallel::ReductionDatum` as
template parameters and `ReductionDatum::value_type`s as the arguments to the
constructor. Each `ReductionDatum` takes up to four template parameters (two
are required). The first is the type of data to reduce, and the second is a
binary invokable that is called at each step of the reduction to combine two
messages. The last two template parameters are used after the reduction has
completed. The third parameter is an n-ary invokable that is called once the
reduction is complete, whose first argument is the result of the reduction. The
additional arguments can be any `ReductionDatum::value_type` in the
`ReductionData` that are before the current one. The fourth template parameter
of `ReductionDatum` is used to specify which data should be passed. It is a
`std::index_sequence` indexing into the `ReductionData`.

The action that is invoked with the result of the reduction is:

\snippet Test_AlgorithmReduction.cpp custom_reduction_action

Note that it takes objects of the types that the reduction was done over as
additional arguments.

\warning
All elements of the array must call the same reductions in the same order. It is
*defined* behavior to do multiple reductions at once as long as all contribute
calls on all array elements occurred in the same order. It is **undefined**
behavior if the contribute calls are made in different orders on different array
elements.

# Charm++ Node and Processor Level Initialization Functions {#dev_guide_parallelization_charm_node_processor_level_initialization}

Charm++ allows running functions once per core and once per node before the
construction of any parallel components. This is commonly used for setting up
error handling and enabling floating point exceptions. Other functions could
also be run. Which functions are run on each node and core is set by specifying
a `std::vector<void (*)()>` called `charm_init_node_funcs` and
`charm_init_proc_funcs` with function pointers to the functions to be called.
For example,
\snippet Test_AlgorithmCore.cpp charm_init_funcs_example

Finally, the user must include the `Parallel/CharmMain.tpp` file at the end of
the main executable cpp file. So, the end of an executables main cpp file will
then typically look as follows:
\snippet Test_AlgorithmParallel.cpp charm_include_example
