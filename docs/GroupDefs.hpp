// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines all group definitions

#pragma once

/*!
 * \defgroup BoundaryConditions Boundary Conditions
 * A collection of boundary conditions used for evolutions.
 */

/*!
 * \defgroup ComputationalDomain  Computational Domain
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
 * \defgroup ConstantExpressions Constant Expressions
 * \brief Contains an assortment of constexpr functions
 *
 * ### Description
 * Contains an assortment of constexpr functions that are useful for
 * metaprogramming, or efficient mathematical computations, such as
 * exponentiating to an integer power, where the power is known at compile
 * time.
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
 * \defgroup DataBoxTags DataBox Tags
 * \brief Structures and metafunctions for labeling the contents of DataBoxes
 */

/*!
 * \defgroup DataStructures Data Structures
 * \brief Various useful data structures used in SpECTRE
 */

/*!
 * \defgroup DomainCreators Domain Creators
 * A collection of different computational domains.
 */

/*!
 * \defgroup ErrorHandling Error Handling
 * Macros and functions used for handling errors
 */

/*!
 * \defgroup EvolveSystem Evolve System
 * \brief Classes and functions used for starting a simulation
 *
 * The EvolveSystem class looks for a defined typelist called
 * `the_tentacle_list`
 * which is then parsed and used to create the Tentacles specified. First single
 * Tentacles (or chares) are created, that is, ones that are not groups or
 * arrays. Next the groups are created by calling the constructor of the group
 * that takes a `CProxy_GlobalCache<the_tentacle_list>&` from which only the
 * input options can be retrieved. During the constructor call no
 * parallelization calls can be made, this will be possible at a second phase
 * of initialization.
 *
 * Next an empty array is created for each of the array Tentacles, followed by
 * an update of the GlobalCache with the new Tentacles set. The second phase
 * of group construction now takes place by calling the
 * `initialize(TaggedTupleTypelist<the_tentacle_list>&, CkCallback)` member
 * function
 * of each group Tentacle distributed object class (i.e. the one that derives
 * off CBase...). The `initialize` function is allowed to communicate between
 * other group chares, however all arrays are still empty at this point. Each
 * `initialize` function must at least call `this->contribute(cb);`.
 *
 * The second last phase of initialization is inserting elements into the
 * arrays. This is done by an `initialize` function inside the *Tentacle*
 * struct. The function signature is:
 * \code
 * template <typename the_tentacle_list>
 * static void initialize(
 *     TaggedTupleTypelist<the_tentacle_list>& tentacles,
 *     CProxy_GlobalCache<the_tentacle_list>& global_cache_proxy);
 * \endcode
 *
 * Finally, if you want a Tentacle to start the simulation you must derive from
 * `Tentacles::is_startup` and define a static member function
 * \code
 * template <typename the_tentacle_list>
 * static void start_execution(
 *     TaggedTupleTypelist<the_tentacle_list>& tentacles,
 *     CProxy_GlobalCache<the_tentacle_list>& global_cache_proxy);
 * \endcode
 * To be concrete, for a dG evolution of a scalar wave only the Element
 * tentacle has a `start_execution` function which just calls the
 * `evolve_from_initial_data()` function on all Element Tentacles.
 *
 * \see
 * EvolveSystem::created_group EvolveSystem::updated_global_cache
 * EvolveSystem::call_group_constructor
 * EvolveSystem::initialize_group_chares
 * EvolveSystem::create_chare_array
 * EvolveSystem::initialize_next_chare_array
 * EvolveSystem::call_start_execution
 */

/*!
 * \defgroup Executables Executables
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
 * \defgroup FileSystem File System
 * \brief A light-weight file system library.
 */

/*!
 * \defgroup GlobalCache Global Cache
 * How to access and utilize the global cache
 */

/*!
 * \defgroup CacheTags Global Cache Tags
 * \brief Tags for common data stored in the GlabalCache
 */

/*!
 * \defgroup HDF5 HDF5
 * \brief Functions and classes for manipulating HDF5 files
 */

/*!
 * \defgroup InputOptions Input File Options
 * \brief Options for input files for different systems
 */

/*!
 * \defgroup NumericalAlgorithms Numerical Algorithms
 * \brief Generic numerical algorithms
 */

/*!
 * \defgroup NumericalFluxes Numerical Fluxes
 * A collection of numerical fluxes used by discontinuous Galerkin methods.
 */

/*!
 * \defgroup OptionParsing Option Parsing
 * Things related to parsing YAML input files.
 */

/*!
 * \defgroup Parallel Parallel Info
 * \brief Functions to get info about parallelization and printing in parallel
 */

/*!
 * \defgroup PrettyType Pretty Type
 * \brief Pretty printing of types
 */

/*!
 * \defgroup Profiling Profiling
 * \brief Functions and variables useful for profiling SpECTRE
 *
 * See the \ref profiling_with_projections "Profiling With Charm++ Projections"
 * section of the dev guide for more details.
 */

/*!
 * \defgroup SlopeLimiters Slope Limiters
 * A collection of slope limiters used for handling shocks.
 */

/*!
 * \defgroup Spectral Spectral
 * Things related to spectral transformations.
 */

/*!
 * \defgroup Tensor Tensor
 * Tensor use documentation.
 */

/*!
 * \defgroup TensorExpressions Tensor Expressions
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
 * \defgroup Tentacles Tentacles
 * Information about which Tentacles are available and how to use them
 */

/*!
 * \defgroup TestingFramework Testing Framework
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
 * // [[ErrorRegex, The error message expected from the test]]
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
 * snippet Test_H5.cpp willfail_example_for_dev_doc
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
 * \defgroup TypeTraits Type Traits
 * A collection of useful type traits, including C++14 and C++17 additions to
 * the standard library.
 */

/*!
 * \defgroup Utilities Utilities
 * \brief A collection of useful classes, functions and metafunctions.
 */

/*!
 * \defgroup VariableFixers Variable Fixers
 * A collection of methods for correcting conservative variables that have
 * unphysical primitive variables.
 */
