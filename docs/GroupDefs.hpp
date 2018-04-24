// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines all group definitions

#pragma once

/*!
 * \defgroup ActionsGroup Actions
 * \brief A collection of steps used in algorithms.
 */

/*!
 * \defgroup BoundaryConditionsGroup Boundary Conditions
 * A collection of boundary conditions used for evolutions.
 */

/*!
 * \defgroup CharmExtensionsGroup Charm++ Extensions
 * \brief Classes and functions used to make Charm++ easier and safer to use.
 */

/*!
 * \defgroup ComputationalDomainGroup  Computational Domain
 * \brief The building blocks used to describe the computational domain.
 *
 * ### Description
 * The VolumeDim-dimensional computational Domain is constructed from a set of
 * non-overlapping Block%s. Each Block is a distorted VolumeDim-dimensional
 * hyperrcube  Each codimension-1 boundary of a Block is either part of the
 * external boundary of the computational domain, or is identical to a boundary
 * of one other Block.  Each Block is subdivided into one or more Element%s
 * that may be changed dynamically if AMR is enabled.
 */

/*!
 * \defgroup ConstantExpressionsGroup Constant Expressions
 * \brief Contains an assortment of constexpr functions
 *
 * ### Description
 * Contains an assortment of constexpr functions that are useful for
 * metaprogramming, or efficient mathematical computations, such as
 * exponentiating to an integer power, where the power is known at compile
 * time.
 */

/*!
 * \defgroup ControlSystemGroup Control System
 * \brief Contains control system elements
 */

/*!
 * \defgroup CoordinateMapsGroup  Coordinate Maps
 * \brief Functions for mapping coordinates between different frames
 *
 * Coordinate maps provide the maps themselves, the inverse maps, along
 * with the Jacobian and inverse Jacobian of the maps.
 */

/*!
 * \defgroup CoordMapsTimeDependentGroup  Coordinate Maps, Time-dependent
 * \brief Functions for mapping time-dependent coordinates between different
 * frames
 *
 * Coordinate maps provide the maps themselves, the inverse maps, the Jacobian
 * and inverse Jacobian of the maps, and the frame velocity (time derivative of
 * the map)
 */

/*!
 * \defgroup DataBoxGroup DataBox
 * \brief Contains (meta)functions used for manipulating DataBoxes
 */

/*!
 * \defgroup DataBoxTagsGroup DataBox Tags
 * \brief Structures and metafunctions for labeling the contents of DataBoxes
 */

/*!
 * \defgroup DataStructuresGroup Data Structures
 * \brief Various useful data structures used in SpECTRE
 */

/*!
 * \defgroup DiscontinuousGalerkinGroup Discontinuous Galerkin
 * \brief Functions and classes specific to the Discontinuous Galerkin
 * algorithm.
 */

/*!
 * \defgroup DomainCreatorsGroup Domain Creators
 * A collection of domain creators for specifying the initial computational
 * domain geometry.
 */

/*!
 * \defgroup EinsteinSolutionsGroup Einstein Solutions
 * \brief Classes which implement analytic solutions to Einstein's equations
 */

/*!
 * \defgroup EquationsOfStateGroup Equations of State
 * \brief The various available equations of state
 */

/*!
 * \defgroup ErrorHandlingGroup Error Handling
 * Macros and functions used for handling errors
 */

/*!
 * \defgroup EventsAndTriggersGroup Events and Triggers
 * \brief Classes and functions related to events and triggers
 */

/*!
 * \defgroup EvolutionSystemsGroup Evolution Systems
 * \brief Contains the namespaces of all the available evolution systems.
 */

/*!
 * \defgroup ExecutablesGroup Executables
 * \brief A list of executables and how to use them
 *
 * <table class="doxtable">
 * <tr>
 * <th>Executable Name </th><th>Description </th>
 * </tr>
 * <tr>
 * <td> \ref ParallelInfoExecutablePage "ParallelInfo" </td>
 * <td> Executable for checking number of nodes, cores, etc.</td>
 * </tr>
 * </table>
 */

/*!
 * \defgroup FileSystemGroup File System
 * \brief A light-weight file system library.
 */

/*!
 * \defgroup GeneralRelativityGroup General Relativity
 * \brief Contains functions used in General Relativistic simulations
 */

/*!
 * \defgroup CacheTagsGroup Global Cache Tags
 * \brief Tags for common data stored in the GlabalCache
 */

/*!
 * \defgroup HDF5Group HDF5
 * \brief Functions and classes for manipulating HDF5 files
 */

/*!
 * \defgroup OptionTagsGroup Input File Options
 * \brief Tags used for options parsed from the input file
 */

/// \defgroup MathFunctionsGroup Math Functions
/// \brief Useful analytic functions

/*!
 * \defgroup NumericalAlgorithmsGroup Numerical Algorithms
 * \brief Generic numerical algorithms
 */

/*!
 * \defgroup OptionParsingGroup Option Parsing
 * Things related to parsing YAML input files.
 */

/*!
 * \defgroup ParallelGroup Parallelization
 * \brief Functions, classes and documentation related to parallelization and
 * Charm++

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

- Metavariables: Provides high-level configuration to the compiler, e.g. the
  physical system to be simulated.
- Phase: Defines distinct simulation phases separated by a global
  synchronization point, e.g. `Initialize`, `Evolve` and `Exit`.
- Algorithm: In each phase, iterates over a list of actions until the current
  phase ends.
- %Parallel component: Maintains and executes its algorithm.
- Action: Performs a computational task, e.g. evaluating the right hand side of
  the time evolution equations. May require data to be received from another
  action potentially being executed on a different core or node.

### The Metavariables Class

SpECTRE takes a different approach to input options passed to an executable than
is common. SpECTRE not only reads an input file at runtime but also has many
choices made at compile time. The compile time options are specified by what is
referred to as the metavariables. What exactly the metavariables struct
specifies depends somewhat on the executable, but all metavariables structs must
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
- `Phase`: an `enum class` that must contain at least `Initialization` and
  `Exit`. Phases are described in the next section.
- `determine_next_phase`: a static function with the signature
  \code
    static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<EvolutionMetavars>& cache_proxy)
      noexcept;
  \endcode
  What this function does is described below in the discussion of phases.

There are also several optional members:

- `input_file`: a `static constexpr OptionString` that is the default name of
  the input file that is to be read. This can be overridden at runtime by
  passing the `--input-file` argument to the executable.
- `ignore_unrecognized_command_line_options`: a `static constexpr bool` that
  defaults to `false`. If set to `true` then unrecognized command line options
  are ignored. Ignoring unrecognized options is generally only necessary for
  tests where arguments for the testing framework, Catch, are passed to the
  executable.

### Phases of an Execution

Global synchronization points, where all cores wait for each other, are
undesirable for scalability reasons. However, they are sometimes inevitable for
algorithmic reasons. That is, in order to actually get a correct solution you
need to have a global synchronization. SpECTRE executables can have multiple
phases, where after each phase a global synchronization occurs. By global
synchronization we mean that no parallel components are executing or have more
tasks to execute: everything is waiting on a task to perform.

Every executable must have at least two phases, `Initialization` and `Exit`. The
next phase is decided by the static member function `determine_next_phase` in
the metavariables. Currently this function has access to the phase that is
ending, and also the global cache. In the future we will add support for
receiving data from various components to allow for more complex decision
making. Here is an example of a `determine_next_phase` function and the `Phase`
enum class:
\snippet Test_AlgorithmCore.cpp determine_next_phase_example

In contrast, an evolution executable might have phases `Initialization`,
`SetInitialData`, `Evolve`, and `Exit`, but have a similar `switch` or `if-else`
logic in the `determine_next_phase` function. The first phase that is entered is
always `Initialization`. During the `Initialization` phase the `initialize`
function is called on all parallel components. Once all parallel components'
`initialize` function is complete, the next phase is determined and the
`execute_next_phase` function is called after on all the parallel components.

At the end of an execution the `Exit` phase has the executable wait to make sure
no parallel components are performing or need to perform any more tasks, and
then exits. An example where this approach is important is if we are done
evolving a system but still need to write data to disk. We do not want to exit
the simulation until all data has been written to disk, even though we've
reached the final time of the evolution.

### The Algorithm

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

\note
Currently all Algorithms must execute the same actions (described below) in all
phases. This restriction is also planned on being relaxed if the need arises.

### %Parallel Components

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
3. `using action_list` is set to a `tmpl::list` of the %Actions (described
   below) that the Algorithm running on the %Parallel Component executes. The
   %Actions are executed in the order that they are given in the `tmpl::list`.
4. `using initial_databox` is set to the type of the DataBox that will be passed
   to the first Action of the `action_list`. Typically it is the output of some
   simple action called during the `Initialization` Phase.
5. `using options` is set to a (possibly empty) `tmpl::list` of the option
   structs. The options are read in from the input file specified in the main
   `Metavariables` struct. After being read in they are passed to the
   `initialize` function of the parallel component, which is described below.
6. `using const_global_cache_tag_list` is set to a `tmpl::list` of CacheTags
   that are required by the parallel component. Some examples are
   `CacheTags::TimeStepper` for evolutions, or `CacheTags::FinalTime`.

\note Array parallel components must also specify the type alias `using
array_index`, which is set to the type that indexes the %Parallel Component
Array. Charm++ allows arrays to be 1 through 6 dimensional or be indexed by a
custom type. The Charm++ provided indexes are wrapped as
`Parallel::ArrayIndex1D` through `Parallel::ArrayIndex6D`. When writing custom
array indices, the [Charm++ manual](http://charm.cs.illinois.edu/help) tells you
to write your own `CkArrayIndex`, but we have written a general implementation
that provides this functionality; all that you need to provide is a
plain-old-data
([POD](http://en.cppreference.com/w/cpp/concept/PODType)) struct of the size of
at most 3 integers.

%Parallel Components have a static `initialize` function that is used
effectively as the constructor of the components. The signature of the
initialize functions must be:
\code
static void initialize(
   Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache, opts...);
\endcode
The `initialize` function is called by the Main %Parallel Component when
the execution starts and will typically call a simple %Action
to set up the initial state of the Algorithm, similar to what a constructor
does for classes. The `initialize` function also receives arguments that
are read from the input file which were specified in the `options` typelist
described above. The options are usually used to initialize the %Parallel
Component's DataBox, or even the component itself. An example of initializing
the component itself would be using the value of an option to control the size
of the %Parallel Component Array.  The `initialize` functions of different
%Parallel Components are called in random order and so it is not safe to have
them depend on each other.

Each parallel component must also decide what to do in the different phases of
the execution. This is controlled by an `execute_next_phase` function with
signature:
\code
static void execute_next_phase(
    const typename metavariables::Phase next_phase,
    const Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache);
\endcode
The `determine_next_phase` function in the Metavariables determines the next
phase, after which the `execute_next_phase` function gets called. The
`execute_next_phase` function determines what the %Parallel Component should do
during the next phase. For example, it may simply call `perform_algorithm`, call
a series of simple actions, perform a reduction over an Array, or not do
anything at all. Note that `perform_algorithm` performs the same actions (the
ones in `action_list`) no matter what Phase it is called in.

An example of a singleton %Parallel Component is:
\snippet Test_AlgorithmParallel.cpp singleton_parallel_component

An example of an array %Parallel Component is:
\snippet Test_AlgorithmParallel.cpp array_parallel_component
Elements are inserted into the Array by using the Charm++ `insert` member
function of the CProxy for the array. The `insert` function is documented in
the Charm++ manual. In the above Array example `array_proxy` is a `CProxy` and
so all the documentation for Charm++ array proxies applies. SpECTRE always
creates empty Arrays with the constructor and requires users to insert however
many elements they want and on which cores they want them to be placed. Note
that load balancing calls may result in Array elements being moved.

### %Actions

For those familiar with Charm++, actions should be thought of as effectively
being entry methods. They are functions that can be invoked on a remote object
(chare/parallel component) using a `CProxy` (see the [Charm++
manual](http://charm.cs.illinois.edu/help)), which is retrieved from the
ConstGlobalCache using the parallel component struct and the
`Parallel::get_parallel_component()` function. %Actions are structs with a
static `apply` method and come in three variants: simple actions, iterable
actions, and reduction actions. One important thing to note
is that actions cannot return any data to the caller of the remote method.
Instead, "returning" data must be done via callbacks or a callback-like
mechanism.

The simplest signature of an `apply` method is for iterable actions:
\snippet Test_AlgorithmCore.cpp apply_iterative
The return type is discussed at the end of each section describing a particular
type of action. Simple actions can have additional arguments but must have at
least the arguments shown above. Reduction actions must have the above arguments
and an argument taken by value that is of the type the reduction was made over.
The `db::DataBox` should be thought of as the member data of the parallel
component while the actions are the member functions. The combination of a
`db::DataBox` and actions allows building up classes with arbitrary member data
and methods using template parameters and invocation of actions. This approach
allows us to eliminate the need for users to work with Charm++'s interface
files, which can be error prone and difficult to use.

The ConstGlobalCache is passed to each action so that the action has access
to global data and is able to invoke actions on other parallel components. The
`ParallelComponent` template parameter is the tag of the parallel component that
invoked the action. A proxy to the calling parallel component can then be
retrieved from the ConstGlobalCache. The remote entry method invocations are
slightly different for different types of actions, so they will be discussed
below. However, one thing that is disallowed for all actions is calling an
action locally from within an action on the same parallel component.
Specifically,

\snippet Test_AlgorithmNestedApply1.cpp bad_recursive_call

Here `ckLocal()`  is a Charm++ provided method that returns a pointer to the
local (currently executing) parallel component. See the [Charm++
manual](http://charm.cs.illinois.edu/help) for more information.
However, you are able to queue a new action to be executed later on the same
parallel component by getting your own parallel component from the
ConstGlobalCache (`Parallel::get_parallel_component<ParallelComponent>(cache)`).
The difference between the two calls is that by calling an action through the
parallel component you will first finish the series of actions you are in, then
when they are complete Charm++ will call the next queued action.

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
`ActionList` type is the `tmpl::list` of iterable actions run on the algorithm.
That is, it is equal to the `action_list` type alias in the parallel component.

#### 1. Simple %Actions

Simple actions are designed to be called in a similar fashion to member
functions of classes. They are the direct analog of entry methods in Charm++
except that the member data is stored in the `db::DataBox` that is passed in as
the first argument. There are a couple of important things to note with simple
actions:

1. A simple action must return void but can use `db::mutate` to change values
   of items in the DataBox if the DataBox is taken as a non-const reference.
   There is one exception: if the input DataBox is empty, then the
   simple action can return a DataBox of type `initial_databox`. That is, an
   action taking an empty DataBox and returning the `initial_databox` is
   effectively constructing the DataBox in its initial state.
2. A simple action is instantiated once for an empty
   `db::DataBox<tmpl::list<>>`, once for a DataBox of type
   `initial_databox` (listed in the parallel component), and once for each
   returned DataBox from the iterable actions in the `action_list` in the
   parallel component. In some cases you will need specific items to be in the
   DataBox otherwise the action won't compile. To restrict which DataBoxes can
   be passed you should use `Requires` in the action's `apply` function
   template parameter list. For example,
   \snippet Test_AlgorithmCore.cpp requires_action
   where the conditional checks if any element in the parameter pack `DbTags` is
   `CountActionsCalled`.


A simple action that does not take any arguments can be called using a `CProxy`
from the ConstGlobalCache as follows:

\snippet Test_AlgorithmCore.cpp simple_action_call

If the simple action takes arguments then the arguments must be passed to the
`simple_action` method as a `std::tuple` (because Charm++ doesn't yet support
variadic entry method templates). For example,

\snippet Test_AlgorithmNodelock.cpp simple_action_with_args

Multiple arguments can be passed to the `std::make_tuple` call.

\note
You must be careful about type deduction when using `std::make_tuple` because
`std::make_tuple(0)` will be of type `std::tuple<int>`, which will not work if
the action is expecting to receive a `size_t` as its extra argument. Instead,
you can get a `std::tuple<size_t>` in one of two ways. First, you can pass in
`std::tuple<size_t>(0)`, second you can include the header
`Utilities/Literals.hpp` and then pass in `std::make_tuple(0_st)`.

#### 2. Iterable %Actions

%Actions in the algorithm that are part of the `action_list` are
executed one after the other until one of them cannot be evaluated. Iterable
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
a `std::unordered_map<TheElementIndex, DataSent>`. Having `DATA` be a
`std::unordered_multiset` is currently also supported. Here is an example of a
receive tag:

\snippet Test_AlgorithmParallel.cpp int_receive_tag

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

Iterable actions can change the type of the DataBox by adding or removing
elements/tags from the DataBox. The only requirement is that the last action in
the `action_list` returns a DataBox that is the same type as the
`initial_databox`. Iterable actions can also request that the algorithm no
longer be executed, and choose which action in the `ActionList`/`action_list` to
execute next. This is all done via the return value from the `apply` function.
The `apply` function for iterable actions must return a `std::tuple` of one,
two, or three elements. The first element of the tuple is the new DataBox,
which can be the same as the type passed in or a DataBox with different tags.
Most iterable actions will simply return:

\snippet Test_AlgorithmParallel.cpp return_forward_as_tuple

By returning the DataBox as a reference in a `std::tuple` we avoid any
unnecessary copying of the DataBox. The second argument is an optional bool, and
controls whether or not the algorithm is terminated. If the bool is `true` then
the algorithm is terminated, by default it is `false`. Here is an example of how
to return a DataBox with the same type that is passed in and also terminate
the algorithm:

\snippet Test_AlgorithmParallel.cpp return_with_termination

Notice that we again return a reference to the DataBox, which is done to avoid
any copying. After an algorithm has been terminated it can be restarted by
passing `false` to the `set_terminate` method followed by calling the
`perform_algorithm` or `receive_data` methods.

The third optional element in the returned `std::tuple` is a `size_t` whose
value corresponds to the index of the action to be called next in the
`action_list`. The metafunction `tmpl::index_of<list, element>` can be used to
get an `tmpl::integral_constant` with the value of the index of the element
`element` in the typelist `list`. For example,

\snippet Test_AlgorithmCore.cpp out_of_order_action

Again a reference to the DataBox is returned, while the termination `bool` and
next action `size_t` are returned by value. The metafunction call
`tmpl::index_of<ActionList, iterate_increment_int0>::%value` returns a `size_t`
whose value is that of the action `iterate_increment_int0` in the `action_list`.
The indexing of actions in the `action_list` starts at `0`.

Iterable actions are invoked as part of the algorithm and so the only way
to request they be invoked is by having the algorithm run on the parallel
component. The algorithm can be explicitly evaluated by call the
`perform_algorithm` method:

\snippet Test_AlgorithmCore.cpp perform_algorithm

The algorithm is also evaluated by calling the `receive_data` function, either
on an entire array or singleton (this does a broadcast), or an on individual
element of the array. Here is an example of a broadcast call:

\snippet Test_AlgorithmParallel.cpp broadcast_to_group

and of calling individual elements:

\snippet Test_AlgorithmParallel.cpp call_on_indexed_array

The `receive_data` function always takes a `ReceiveTag`, which is set in the
actions `inbox_tags` type alias as described above.  The first argument is the
temporal identifier, and the second is the data to be sent.

#### 3. Reduction %Actions

Finally, there are reduction actions which are used when reducing data over an
array. For example, you may want to know the sum of a `int` from every
element in the array. You can do this by using Charm++'s built-in reduction
function for summing `int`s as follows:

\snippet Test_AlgorithmReduction.cpp contribute_to_reduction_example

This reduces over the parallel component
`ArrayParallelComponent<Metavariables>`, reduces to the parallel component
`SingletonParallelComponent<Metavariables>`, and calls the action
`ProcessReducedSumOfInts` after the reduction has been performed. The
`CkReduction::sum_int` argument tells Charm++ that you want to sum integers
instead of some other type or operation. The reduction action is:

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
`Parallel::ReductionData`, which takes as template parameters the types it
holds, similar to `std::tuple`. An example of a custom reduction function is:

\snippet Test_AlgorithmReduction.cpp custom_reduce_function

The action that is invoked with the result of the reduction is:

\snippet Test_AlgorithmReduction.cpp custom_reduction_action

Note that it takes a `Parallel::ReductionData` object as its last argument.

\warning
All elements of the array must call the same reductions in the same order. It is
defined behavior to do multiple reductions at once as long as all contribute
calls on all array elements occurred in the same order. It is undefined behavior
if the contribute calls are made in different orders on different array
elements.

### Charm++ Node and Processor Level Initialization Functions

Charm++ allows running functions once per core and once per node before the
construction of any parallel components. This is commonly used for setting up
error handling and enabling floating point exceptions. Other functions could
also be run. Which functions are run on each node and core is set by specifying
a `std::vector<void (*)()>` called `charm_init_node_funcs` and
`charm_init_proc_funcs` with function pointers to the functions to be called.
For example,
\snippet Test_AlgorithmCore.cpp charm_init_funcs_example

Finally, the user must include the `Parallel/CharmMain.cpp` file at the end of
the main executable cpp file. So, the end of an executables main cpp file will
then typically look as follows:
\snippet Test_AlgorithmParallel.cpp charm_include_example
 */

/*!
 * \defgroup PrettyTypeGroup Pretty Type
 * \brief Pretty printing of types
 */

/*!
 * \defgroup SpectralGroup Spectral
 * Things related to spectral transformations.
 */

/*!
 * \defgroup SurfacesGroup Surfaces
 * Things related to surfaces.
 */

/*!
 * \defgroup TensorGroup Tensor
 * Tensor use documentation.
 */

/*!
 * \defgroup TensorExpressionsGroup Tensor Expressions
 * Tensor Expressions allow writing expressions of
 * tensors in a way similar to what is used with pen and paper.
 *
 * Tensor expressions are implemented using (smart) expression templates. This
 * allows a domain specific language making expressions such as
 * \code
 * auto T = evaluate<Indices::_a_t, Indices::_b_t>(F(Indices::_b,
 * Indices::_a));
 * \endcode
 * possible.
 */

/*!
 * \defgroup TestingFrameworkGroup Testing Framework
 * \brief Classes, functions, macros, and instructions for developing tests
 *
 * \details
 *
 * SpECTRE uses the testing framework
 * [Catch](https://github.com/philsquared/Catch). Catch supports a variety of
 * different styles of tests including BDD and fixture tests. The file
 * `cmake/SpectreAddCatchTests.cmake` parses the source files and adds the found
 * tests to ctest with the correct properties specified by tags and attributes.
 *
 * ### Usage
 *
 * To run the tests, type `ctest` in the build directory. You can specify
 * a regex to match the test name using `ctest -R Unit.Blah`, or run all
 * tests with a certain tag using `ctest -L tag`.
 *
 * ### Comparing double-precision results
 *
 * To compare two floating-point numbers that may differ by round-off, use the
 * helper object `approx`. This is an instance of Catch's comparison class
 * `Approx` in which the relative tolerance for comparisons is set to roughly
 * \f$10^{-14}\f$ (i.e. `std::numeric_limits<double>::%epsilon()*100`).
 * When possible, we recommend using `approx` for fuzzy comparisons as follows:
 * \example
 * \snippet TestFramework.cpp approx_default
 *
 * For checks that need more control over the precision (e.g. an algorithm in
 * which round-off errors accumulate to a higher level), we recommend using
 * the `approx` helper with a one-time tolerance adjustment. A comment
 * should explain the reason for the adjustment:
 * \example
 * \snippet TestFramework.cpp approx_single_custom
 *
 * For tests in which the same precision adjustment is re-used many times, a new
 * helper object can be created from Catch's `Approx` with a custom precision:
 * \example
 * \snippet TestFramework.cpp approx_new_custom
 *
 * Note: We provide the `approx` object because Catch's `Approx` defaults to a
 * very loose tolerance (`std::numeric_limits<float>::%epsilon()*100`, or
 * roughly \f$10^{-5}\f$ relative error), and so is poorly-suited to checking
 * many numerical algorithms that rely on double-precision accuracy. By
 * providing a tighter tolerance with `approx`, we avoid having to redefine the
 * tolerance in every test.
 *
 * ### Attributes
 *
 * Attributes allow you to modify properties of the test. Attributes are
 * specified as follows:
 * \code
 * // [[TimeOut, 10]]
 * // [[OutputRegex, The error message expected from the test]]
 * SPECTRE_TEST_CASE("Unit.Blah", "[Unit]") {
 * \endcode
 *
 * Available attributes are:
 *
 * <table class="doxtable">
 * <tr>
 * <th>Attribute </th><th>Description  </th>
 * </tr>
 * <tr>
 * <td>TimeOut </td>
 * <td>override the default timeout and set the timeout to N seconds. This
 * should be set very sparingly since unit tests are designed to be
 * short. If your test is too long you should consider testing smaller
 * portions of the code if possible, or writing an integration test instead.
 * </td>
 * </tr>
 * <tr>
 * <td>OutputRegex </td>
 * <td>
 * When testing failure modes the exact error message must be tested, not
 * just that the test failed. Since the string passed is a regular
 * expression you must escape any regex tokens. For example, to match
 * `some (word) and` you must specify the string `some \(word\) and`.
 * </td>
 * </tr>
 * </table>
 *
 * \example
 * \snippet Test_H5.cpp willfail_example_for_dev_doc
 *
 * ### Testing static assert
 *
 * You are able to test that a `static_assert` is being triggered using
 * the compilation failure test framework. When creating a new `static_assert`
 * test you must be sure to not have it in the same file as the runtime tests
 * since the file will not compile. The new file, say
 * `Test_StaticAssertDataBox.cpp` must be added to the
 * `SPECTRE_COMPILATION_TESTS` CMake variable, not `SPECTRE_TESTS`. Here is
 * an example of how to write a compilation failure test:
 *
 * \snippet TestCompilationFramework.cpp compilation_test_example
 *
 * Each individual test must be inside an `#%ifdef COMPILATION_TEST_.*` block
 * and each compilation test `cpp` file must contain
 * `FILE_IS_COMPILATION_TEST` outside of any `#%ifdef`s and at the end of
 * the file.
 *
 * Specific compiler versions can be specified for which the regex changes.
 * That is, the compiler version specified and all versions newer than that
 * will use the regex, until a newer compiler version is specified. For
 * example, see the below code prints a different static_assert for pre-GCC 6
 * and GCC 6 and newer.
 *
 * \snippet TestCompilationFramework.cpp gnu_versions_example
 *
 * ### Debugging Tests in GDB or LLDB
 *
 * Several tests fail intentionally at the executable level to test error
 * handling like ASSERT statements in the code. CTest is aware of which
 * should fail and passes them. If you want to debug an individual test
 * in a debugger you need to run a single test
 * using the %RunTests executable (in dg-charm-build/bin/RunTests) you
 * must specify the name of the test as the first argument. For example, if you
 * want to run just the "Unit.Gradient" test you can run
 * `./bin/RunTests Unit.Gradient`. If you are using a debugger launch the
 * debugger, for example if you're using LLDB then run `lldb ./bin/RunTests`
 * and then to run the executable inside the debugger use `run Unit.Gradient`
 * inside the debugger.
 */

/*!
 * \defgroup TimeGroup Time
 * \brief Code related to the representation of time during simulations.
 *
 * The time covered by a simulation is divided up into a sequence of
 * adjacent, non-overlapping (except at endpoints) intervals referred
 * to as "slabs".  The boundaries between slabs can be placed at
 * arbitrary times.  Slabs, as represented in the code as the Slab
 * class, provide comparison operators comparing slabs agreeing with
 * the definition as a sequence of intervals.  Slabs that do not
 * jointly belong to any such sequence should not be compared.
 *
 * The specific time is represented by the Time class, which encodes
 * the slab containing the time and the fraction of the slab that has
 * elapsed as an exact rational.  Times are comparable according to
 * their natural time ordering, except for times belonging to
 * incomparable slabs.
 *
 * Differences in time within a slab are represented as exact
 * fractions of that slab by the TimeDelta class.  TimeDeltas are only
 * meaningful within a single slab, with the exception that the ratio
 * of objects with different slabs may be taken, resulting in an
 * inexact floating-point result.  Longer intervals of time are
 * represented using floating-point values.
 */

/*!
 * \defgroup TimeSteppersGroup Time Steppers
 * A collection of ODE integrators primarily used for time stepping.
 */

/*!
 * \defgroup TypeTraitsGroup Type Traits
 * A collection of useful type traits, including C++14 and C++17 additions to
 * the standard library.
 */

/*!
 * \defgroup UtilitiesGroup Utilities
 * \brief A collection of useful classes, functions and metafunctions.
 */
