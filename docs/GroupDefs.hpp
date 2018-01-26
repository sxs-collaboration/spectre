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
 * \defgroup ErrorHandlingGroup Error Handling
 * Macros and functions used for handling errors
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
initialization for the user that can otherwise lead to difficult to debug
undefined behavior. The central concept is what is called a %Parallel
Component. A %Parallel Component is a struct with several type aliases that
is used by SpECTRE to set up the Charm++ chares and allowed communication
patterns. It might be most natural to think of %Parallel Components
as input arguments to a metaprogram that writes the parallelization
infrastructure that you requested for the executable. There is no restriction
on the number of %Parallel Components, though practically it is best to have
around 10 at most.

Each %Parallel Component must have the following type aliases:
1. `using chare_type` is set to one of `Parallel::Algorithms::Singleton`,
   `Parallel::Algorithms::Array`, `Parallel::Algorithms::Group`, or
   `Parallel::Algorithms::Nodegroup`. What these mean is explained below.
2. `using metavariables` is set to the Metavariables struct that stores the
   global metavariables. It is often easiest to have the %Parallel
   Component struct have a template parameter `Metavariables` that is the
   global metavariables struct. Examples of this are given below.
3. `using action_list` is set to a `typelist` of the Actions that the Algorithm
   running on the %Parallel Component executes. The Actions are executed in
   the order that they are listed in in the typelist.
4. `using initial_databox` is set to the type of the DataBox that will be passed
   to the first Action of the `action_list`. Typically it is the output of some
   `explicit_single_action` call made during the `initialize` function. More
   on this below.
5. `using options` is set to a (possibly empty) typelist of the option structs
   which are read in from the input file specified in the main `Metavariables`
   struct and passed to the `initialize` function described below.

The following type aliases are not always required, but will be in some
circumstances:
1. `using array_index` is set to the type that indexes the %Parallel Component
   Array and is only required if `chare_type = Parallel::Algorithms::Array`.
   Charm++ allows arrays to be 1 through 6 dimensional or be indexed by a custom
   type. The Charm++ provided indexes are wrapped as `Parallel::%ArrayIndex1D'
   through `Parallel::ArrayIndex6D`. When writing custom array indices, the
   Charm++ manual tells you to write your own `CkArrayIndex`, but we have
   written a general implementation that provides this functionality; all that
   you need to provide is a plain-old-data struct of the size of at most 3
   integers.
2. `using explicit_single_actions` is set to a typelist of Actions
   that are not part of the algorithm but can be called remotely at any time,
   similar in spirit to a member function.  An Action used as an explicit
   single action has the following restrictions:
   1. It must specify a member type alias called `apply_args` whose
      elements are the types of additional arguments passed into the
      Action's `apply` function (these additional arguments are passed
      after the ParallelComponent argument).  If there are no additional
      arguments, use an empty typelist.
   2. Its returned DataBox must have the same
      type as its input DataBox (use `db::mutate` to change values
      of things in the DataBox).  There is one exception:
      if the input DataBox is empty, then the explicit single action
      can return a DataBox of type `initial_databox` (this is how one
      initializes the sequence of Actions).
   3. It is instantiated multiple times, once for
      an empty DataBox, once for a DataBox of type
      `initial_databox`, and once for the returned DataBox type of each
      Action in `action_list`.  If you want the action to be instantiated
      only for a subset of these, use `Requires` in the Action's template
      parameter list.
3. `using reduction_actions_list` is set to a typelist of the Actions that may
   be called for reductions. Each Action that is to be used in a reduction must
   contain a member type alias named `reduction_type` whose value is the type
   being reduced over. For example, the reduced type could be an `int`, a
   `double`, or a specialization of `Parallel::ReductionData`.

%Parallel Components must also have two static member functions with the
following signatures:

\code
static void initialize(
   Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache, opts...);
\endcode

and

\code
static void execute_next_global_actions(
    const typename metavariables::Phase next_phase,
    const Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache);
\endcode

The `initialize` function is called by the Main %Parallel Component when
the execution starts and will typically call an explicit single Action
to set up the initial state of the Algorithm, similar to what a constructor
does for classes. The `initialize` function also receives arguments that
are read from the input file and can the be used to initialize the %Parallel
Component. For example, the value of an option could be distributed to all
members of a %Parallel Component Array, or could be used to control the size
of the %Parallel Component Array.  The `initialize` functions of different
%Parallel Components are called in random order.

The `execute_next_global_actions` function gets run at the end of each
%Parallel Phase and determines what the %Parallel Component should do
during the next phase. For example, it may simply call `perform_algorithm`,
call a series of single Actions, perform a reduction over an Array, or not
do anything at all.  Note that `perform_algorithm` performs the same
Actions (the ones in `action_list`) no matter what Phase it is called
in.

An example of a singleton %Parallel Component is:

\snippet Test_AlgorithmParallel.cpp singleton_parallel_component

and similarly of a %Parallel Component Array:

\snippet Test_AlgorithmParallel.cpp array_parallel_component

Elements are inserted into the Array by using the Charm++ `insert` member
function of the CProxy for the array. The `insert` function is is documented in
the Charm++ manual. In the above Array example `array_proxy` is a `CProxy` to
the `AlgorithmImpl<Parallel::Algorithms::Array, ...>` and so all the
documentation for Charm++ array proxies applies. We always create empty
Arrays with the constructor and require users to insert however many elements
they want and on which cores they want them to be placed. Note that load
balancing calls may result in Array elements being moved.

There are four types of Algorithms with one Algorithm object per Charm++
chare object. The four types of Algorithms are:
1. A Parallel::Algorithms::Singleton where there is only one object
   in the entire execution of the program.
2. A Parallel::Algorithms::Array which holds zero or more
   elements each of which is an object distributed to some core. An array can
   grow and shrink in size dynamically if need be and can also be bound to
   another array. A bound array has the same number of elements as
   the array it is bound to, and elements with the same ID are on the same
   core. See Charm++'s chare arrays for details.
3. A Parallel::Algorithms::Group is an array with
   one element per core which are not able to be moved around between
   cores. These are typically useful for gathering data from array elements
   on their core, and then processing or reducing it. See Charm++'s group
   chares for details
4. A Parallel::Algorithms::Nodegroup, which is similar to a
   group except that there is one element per node. For Charm++ SMP (shared
   memory parallelism) builds a node corresponds to the usual definition of a
   node on a supercomputer. However, for non-SMP builds nodes and cores are
   equivalent. We ensure that all entry method calls done through the
   Algorithm's `explicit_single_action` and `receive_data` functions are
   threadsafe. User controlled threading is possible by calling the non-entry
   method member function `threaded_single_action`.

### Entry Methods and Remote Function Invocation

Charm++ refers to functions that can be called remotely as entry methods.
The %Parallel Components provide several generic entry methods. These fall
into two classes:
1. Entry methods that perform a single Action once
2. Entry methods that iterate the Actions in the Algorithm's `action_list`.
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
 * \f$10^{-14}\f$ (i.e. `std\:\:numeric_limits<double>\:\:epsilon()*100`).
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
 * very loose tolerance (`std\:\:numeric_limits<float>\:\:epsilon()*100`, or
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
 * <td>ErrorRegex </td>
 * <td>
 * When testing failure modes the exact error message must be tested, not
 * just that the test failed. Since the string passed is a regular
 * expression you must escape any regex tokens. For example, to match
 * "some (word) and" you must specify the string "some \(word\) and".
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
