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
 * \defgroup AnalyticDataGroup Analytic Data
 * \brief Analytic data used to specify (for example) initial data to the
 * equations implemented in \ref EvolutionSystemsGroup.
 */

/*!
 * \defgroup AnalyticSolutionsGroup Analytic Solutions
 * \brief Analytic solutions to the equations implemented in \ref
 * EvolutionSystemsGroup and \ref EllipticSystemsGroup.
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
 * non-overlapping Block%s.  Each Block is a distorted VolumeDim-dimensional
 * hypercube.  Each codimension-1 boundary of a Block is either part of the
 * external boundary of the computational domain, or is identical to a boundary
 * of one other Block.  Each Block is subdivided into one or more Element%s
 * that may be changed dynamically if AMR is enabled.
 */

/*!
 * \defgroup ConservativeGroup Conservative System Evolution
 * \brief Contains generic functions used for evolving conservative
 * systems.
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
 *
 * The control system manages the time-dependent mapping between frames, such as
 * the fixed computational frame (grid frame) and the inertial frame. The
 * time-dependent parameters of the mapping are adjusted by a feedback control
 * system in order to follow the dynamical evolution of objects such as horizons
 * of black holes or surfaces of neutron stars. For example, in binary black
 * hole simulations the map is typically a composition of maps that include
 * translation, rotation, scaling, shape, etc.
 * Each map under the governance of the control system has an associated
 * time-dependent map parameter \f$\lambda(t)\f$ that is a piecewise Nth order
 * polynomial. At discrete times (called reset times), the control system resets
 * the Nth time derivative of \f$\lambda(t)\f$ to a new constant value, in order
 * to minimize an error function \f$Q(t)\f$ that is specific to each map. At
 * each reset time, the Nth derivative of \f$\lambda(t)\f$ is set to a function
 * \f$U(t)\f$, called the control signal, that is determined by \f$Q(t)\f$ and
 * its time derivatives and time integral. Note that \f$\lambda(t)\f$,
 * \f$U(t)\f$, and \f$Q(t)\f$ can be vectors.
 *
 * The key components of the control system are:
 * - FunctionsOfTime: each map has an associated FunctionOfTime that represents
 *   the map parameter \f$\lambda(t)\f$ and relevant time derivatives.
 * - ControlError: each map has an associated ControlError that computes
 *   the error, \f$Q(t)\f$. Note that for each map, \f$Q(t)\f$ is defined to
 *   follow the convention that \f$dQ = -d \lambda\f$ as \f$Q \rightarrow 0\f$.
 * - Averager: an averager can be used to average out the noise in the 'raw'
 *   \f$Q(t)\f$ returned by the ControlError.
 * - Controller: the map controller computes the control signal \f$U(t)\f$ from
 *   \f$Q(t)\f$ and its time integral and time derivatives.
 *   The control is accomplished by setting the Nth derivative of
 *   \f$\lambda(t)\f$ to \f$U(t)\f$. Two common controllers are PID
 *   (proportional/integral/derivative)
 *   \f[U(t) = a_{0}\int_{t_{0}}^{t} Q(t') dt'+a_{1}Q(t)+a_{2}\frac{dQ}{dt}\f]
 *   or
 *   PND (proportional/N derivatives)
 *   \f[ U(t) = \sum_{k=0}^{N} a_{k} \frac{d^kQ}{dt^k} \f]
 *   The coefficients \f$ a_{k} \f$ in the computation of \f$U(t)\f$ are chosen
 *   at each time such that the error \f$Q(t)\f$ will be critically damped
 *   on a timescale of \f$\tau\f$ (the damping time),
 *   i.e. \f$Q(t) \propto e^{-t/\tau}\f$.
 * - TimescaleTuner: each map has a TimescaleTuner that dynamically adjusts
 *   the damping timescale \f$\tau\f$ appropriately to keep the error \f$Q(t)\f$
 *   within some specified error bounds. Note that the reset time interval,
 *   \f$\Delta t\f$, is a constant fraction of this damping timescale,
 *   i.e. \f$\Delta t = \alpha \tau\f$ (empirically, we have found
 *   \f$\alpha=0.3\f$ to be a good choice).
 *
 *
 * For additional details describing our control system approach, see
 * \cite Hemberger2012jz.
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
 * \brief Documentation, functions, metafunctions, and classes necessary for
 * using DataBox
 *
 * DataBox is a heterogeneous compile-time associative container with lazy
 * evaluation of functions. DataBox can not only store data, but can also store
 * functions that depend on other data inside the DataBox. The functions will be
 * evaluated when the data they return is requested. The result is cached, and
 * if a dependency of the function is modified the cache is invalidated.
 *
 * #### Simple and Compute Tags and Their Items
 *
 * The compile-time keys are `struct`s called tags, while the values are called
 * items. Tags are quite minimal, containing only the information necessary to
 * store the data and evaluate functions. There are two different types of tags
 * that a DataBox can hold: simple tags and compute tags. Simple tags are for
 * data that is inserted into the DataBox at the time of creation, while compute
 * tags are for data that will be computed from a function when the compute item
 * is retrieved. If a compute item is never retrieved from the DataBox then it
 * is never evaluated.
 *
 * Simple tags must have a member type alias `type` that is the type of the data
 * to be stored and a `static std::string name()` method that returns the name
 * of the tag. Simple tags must inherit from `db::SimpleTag`.
 *
 * Compute tags must also have a `static std::string name()` method that returns
 * the name of the tag, but they cannot have a `type` type alias. Instead,
 * compute tags must have a static member function or static member function
 * pointer named `function`. `function` can be a function template if necessary.
 * The `function` must take all its arguments by `const` reference. The
 * arguments to the function are retrieved using tags from the DataBox that the
 * compute tag is in. The tags for the arguments are set in the member type
 * alias `argument_tags`, which must be a `tmpl::list` of the tags corresponding
 * to each argument. Note that the order of the tags in the `argument_list` is
 * the order that they will be passed to the function. Compute tags must inherit
 * from `db::ComputeTag`.
 *
 * Here is an example of a simple tag:
 *
 * \snippet Test_DataBox.cpp databox_tag_example
 *
 * and an example of a compute tag with a function pointer:
 *
 * \snippet Test_DataBox.cpp databox_compute_item_tag_example
 *
 * If the compute item's tag is inline then the compute item is of the form:
 *
 * \snippet Test_DataBox.cpp compute_item_tag_function
 *
 * Compute tags can also have their functions be overloaded on the type of its
 * arguments:
 *
 * \snippet Test_DataBox.cpp overload_compute_tag_type
 *
 * or be overloaded on the number of arguments:
 *
 * \snippet Test_DataBox.cpp overload_compute_tag_number_of_args
 *
 * Compute tag function templates are implemented as follows:
 *
 * \snippet Test_DataBox.cpp overload_compute_tag_template
 *
 * Finally, overloading, function templates, and variadic functions can be
 * combined to produce extremely generic compute tags. The below compute tag
 * takes as template parameters a parameter pack of integers, which is used to
 * specify several of the arguments. The function is overloaded for the single
 * argument case, and a variadic function template is provided for the multiple
 * arguments case. Note that in practice few compute tags will be this complex.
 *
 * \snippet Test_BaseTags.cpp compute_template_base_tags
 *
 * #### Subitems and Prefix Tags
 *
 * A simple or compute tag might also hold a collection of data, such as a
 * container of `Tensor`s. In many cases you will want to be able to retrieve
 * individual elements of the collection from the DataBox without having to
 * first retrieve the collection. The infrastructure that allows for this is
 * called *Subitems*. The subitems of the parent tag must refer to a subset of
 * the data inside the parent tag, e.g. one `Tensor` in the collection. If the
 * parent tag is `Parent` and the subitems tags are `Sub<0>, Sub<1>`, then when
 * `Parent` is added to the DataBox, so are `Sub<0>` and `Sub<1>`. This means
 * the retrieval mechanisms described below will work on `Parent`, `Sub<0>`, and
 * `Sub<1>`.
 *
 * Subitems specify requirements on the tags they act on. For example, there
 * could be a requirement that all tags with a certain type are to be treated as
 * a Subitms. Let's say that the `Parent` tag holds a `Variables`, and
 * `Variables` can be used with the Subitems infrastructure to add the nested
 * `Tensor`s. Then all tags that hold a `Variables` will have their subitems
 * added into the DataBox. To add a new type as a subitem the `db::Subitems`
 * struct must be specialized. See the documentation of `db::Subitems` for more
 * details.
 *
 * The DataBox also supports *prefix tags*, which are commonly used for items
 * that are related to a different item by some operation. Specifically, say
 * you have a tag `MyTensor` and you want to also have the time derivative of
 * `MyTensor`, then you can use the prefix tag `dt` to get `dt<MyTensor>`. The
 * benefit of a prefix tag over, say, a separate tag `dtMyTensor` is that prefix
 * tags can be added and removed by the compute tags acting on the original tag.
 * Prefix tags can also be composed, so a second time derivative would be
 * `dt<dt<MyTensor>>`. The net result of the prefix tags infrastructure is that
 * the compute tag that returns `dt<MyTensor>` only needs to know its input
 * tags, it knows how to name its output based off that. In addition to the
 * normal things a simple or a compute tag must hold, prefix tags must have a
 * nested type alias `tag`, which is the tag being prefixed. Prefix tags must
 * also inherit from `db::PrefixTag` in addition to inheriting from
 * `db::SimpleTag` or `db::ComputeTag`.
 *
 * #### Creating a DataBox
 *
 * You should never call the constructor of a DataBox directly. DataBox
 * construction is quite complicated and the helper functions `db::create` and
 * `db::create_from` should be used instead. `db::create` is used to construct a
 * new DataBox. It takes two typelists as explicit template parameters, the
 * first being a list of the simple tags to add and the second being a list of
 * compute tags to add. If no compute tags are being added then only the simple
 * tags list must be specified. The tags lists should be passed as
 * `db::create<db::AddSimpleTags<simple_tags...>,
 * db::AddComputeTags<compute_tags...>>`. The arguments to `db::create` are the
 * initial values of the simple tags and must be passed in the same order as the
 * tags in the `db::AddSimpleTags` list. If the type of an argument passed to
 * `db::create` does not match the type of the corresponding simple tag a static
 * assertion will trigger. Here is an example of how to use `db::create`:
 *
 * \snippet Test_DataBox.cpp create_databox
 *
 * To create a new DataBox from an existing one use the `db::create_from`
 * function. The only time a new DataBox needs to be created is when tags need
 * to be removed or added. Like `db::create`, `db::create_from` also takes
 * typelists as explicit template parameter. The first template parameter is the
 * list of tags to be removed, which is passed using `db::RemoveTags`, second is
 * the list of simple tags to add, and the third is the list of compute tags to
 * add. If tags are only removed then only the first template parameter needs to
 * be specified. If tags are being removed and only simple tags are being added
 * then only the first two template parameters need to be specified. Here is an
 * example of removing a tag or compute tag:
 *
 * \snippet Test_DataBox.cpp create_from_remove
 *
 * Adding a simple tag is done using:
 *
 * \snippet Test_DataBox.cpp create_from_add_item
 *
 * Adding a compute tag is done using:
 *
 * \snippet Test_DataBox.cpp create_from_add_compute_item
 *
 * #### Accessing and Mutating Items
 *
 * To retrieve an item from a DataBox use the `db::get` function. `db::get`
 * will always return a `const` reference to the object stored in the DataBox
 * and will also have full type information available. This means you are able
 * to use `const auto&` when retrieving tags from the DataBox. For example,
 * \snippet Test_DataBox.cpp using_db_get
 *
 * If you want to mutate the value of a simple item in the DataBox use
 * `db::mutate`. Any compute item that depends on the mutated item will have its
 * cached value invalidated and be recomputed the next time it is retrieved from
 * the DataBox. `db::mutate` takes a parameter pack of tags to mutate as
 * explicit template parameters, a `gsl::not_null` of the DataBox whose items
 * will be mutated, an invokable, and extra arguments to forward to the
 * invokable. The invokable takes the arguments passed from the DataBox by
 * `const gsl::not_null` while the extra arguments are forwarded to the
 * invokable. The invokable is not allowed to retrieve anything from the
 * DataBox, so any items must be passed as extra arguments using `db::get` to
 * retrieve them. For example,
 *
 * \snippet Test_DataBox.cpp databox_mutate_example
 *
 * In addition to retrieving items using `db::get` and mutating them using
 * `db::mutate`, there is a facility to invoke an invokable with tags from the
 * DataBox. `db::apply` takes a `tmpl::list` of tags as an explicit template
 * parameter, will retrieve all the tags from the DataBox passed in and then
 * invoke the  invokable with the items in the tag list. Similarly,
 * `db::mutate_apply` invokes the invokable but allows for mutating some of
 * the tags. See the documentation of `db::apply` and `db::mutate_apply` for
 * examples of how to use them.
 *
 * #### The Base Tags Mechanism
 *
 * Retrieving items by tags should not require knowing whether the item being
 * retrieved was computed using a compute tag or simply added using a simple
 * tag. The framework that handles this falls under the umbrella term
 * *base tags*. The reason is that a compute tag can inherit from a simple tag
 * with the same item type, and then calls to `db::get` with the simple tag can
 * be used to retrieve the compute item as well. That is, say you have a compute
 * tag `ArrayCompute` that derives off of the simple tag `Array`, then you can
 * retrieve the compute tag `ArrayCompute` and `Array` by calling
 * `db::get<Array>(box)`. The base tags mechanism requires that only one `Array`
 * tag be present in the DataBox, otherwise a static assertion is triggered.
 *
 * The inheritance idea can be generalized further with what are called base
 * tags. A base tag is an empty `struct` that inherits from `db::BaseTag`. Any
 * simple or compute item that derives off of the base tag can be retrieved
 * using `db::get`. Consider the following `VectorBase` and `Vector` tag:
 *
 * \snippet Test_BaseTags.cpp vector_base_definitions
 *
 * It is possible to retrieve `Vector<1>` from the DataBox using
 * `VectorBase<1>`. Most importantly, base tags can also be used in compute tag
 * arguments, as follows:
 *
 * \snippet Test_BaseTags.cpp compute_template_base_tags
 *
 * As shown in the code example, the base tag mechanism works with function
 * template compute tags, enabling generic programming to be combined with the
 * lazy evaluation and automatic dependency analysis offered by the DataBox. To
 * really demonstrate the power of base tags, let's also have `ArrayComputeBase`
 * inherit from a simple tag `Array`, which inherits from a base tag `ArrayBase`
 * as follows:
 *
 * \snippet Test_BaseTags.cpp array_base_definitions
 *
 * To start, let's create a DataBox that holds a `Vector<0>` and an
 * `ArrayComputeBase<0>` (the concrete tag must be used when creating the
 * DataBox, not the base tags), retrieve the tags using the base tag mechanism,
 * including mutating `Vector<0>`, and then verifying that the dependencies are
 * handled correctly.
 *
 * \snippet Test_BaseTags.cpp base_simple_and_compute_mutate
 *
 * Notice that we are able to retrieve `ArrayComputeBase<0>` with `ArrayBase<0>`
 * and `Array<0>`. We were also able to mutate `Vector<0>` using
 * `VectorBase<0>`.
 *
 * We can even remove tags using their base tags with `db::create_from`:
 *
 * \snippet Test_BaseTags.cpp remove_using_base
 *
 * The base tags infrastructure even works with Subitems. Even if you mutate the
 * subitem of a parent using a base tag, the appropriate compute item caches
 * will be invalidated.
 *
 * \note All of the base tags infrastructure works for `db::get`, `db::mutate`,
 * `db::apply` and `db::mutate_apply`.
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
 * \defgroup EllipticSystemsGroup Elliptic Systems
 * \brief All available elliptic systems and information on how to implement
 * elliptic systems
 *
 * \details Actions and parallel components may require an elliptic system to
 * expose the following types:
 *
 * - `volume_dim`: The number of spatial dimensions
 * - `fields_tag`: A \ref DataBoxGroup tag that represents the fields being
 * solved for.
 * - `variables_tag`: The variables to compute DG volume contributions and
 * fluxes for. Use `db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>`
 * unless you have a reason not to.
 * - `compute_operator_action`: A struct that computes the bulk contribution to
 * the DG operator. Must expose a `tmpl::list` of `argument_tags` and a static
 * `apply` function that takes the following arguments in this order:
 *   - First, the types of the tensors in
 * `db::add_tag_prefix<Metavariables::temporal_id::step_prefix, variables_tag>`
 * (which represent the linear operator applied to the variables) as not-null
 * pointers.
 *   - Followed by the types of the `argument_tags` as constant references.
 *
 * Actions and parallel components may also require the Metavariables to expose
 * the following types:
 *
 * - `system`: See above.
 * - `temporal_id`: A DataBox tag that identifies steps in the algorithm.
 * Generally use `LinearSolver::Tags::IterationId`.
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
 * \brief All available evolution systems and information on how to implement
 * evolution systems
 *
 * \details Actions and parallel components may require an evolution system to
 * expose the following types:
 *
 * - `volume_dim`: The number of spatial dimensions
 * - `variables_tag`: The evolved variables to compute DG volume contributions
 * and fluxes for.
 * - `compute_time_derivative`: A struct that computes the bulk contribution to
 * the DG discretization of the time derivative. Must expose a `tmpl::list` of
 * `argument_tags` and a static `apply` function that takes the following
 * arguments in this order:
 *   - First, the types of the tensors in
 * `db::add_tag_prefix<Metavariables::temporal_id::step_prefix, variables_tag>`
 * (which represent the time derivatives of the variables) as not-null pointers.
 *   - The types of the `argument_tags` as constant references.
 *
 * Actions and parallel components may also require the Metavariables to expose
 * the following types:
 *
 * - `system`: See above.
 * - `temporal_id`: A DataBox tag that identifies steps in the algorithm.
 * Generally use `Tags::TimeStepId`.
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
 * \defgroup HDF5Group HDF5
 * \brief Functions and classes for manipulating HDF5 files
 */

/*!
 * \defgroup InitializationGroup Initialization
 * \brief Actions and metafunctions used for initialization of parallel
 * components.
 */

/*!
 * \defgroup LimitersGroup Limiters
 * \brief Limiters to control shocks and surfaces in the solution.
 */

/*!
 * \defgroup LinearSolverGroup  Linear Solver
 * \brief Algorithms to solve linear systems of equations
 *
 * \details In a way, the linear solver is for elliptic systems what time
 * stepping is for the evolution code. This is because the DG scheme for an
 * elliptic system reduces to a linear system of equations of the type
 * \f$Ax=b\f$, where \f$A\f$ is a global matrix representing the DG
 * discretization of the problem. Since this is one equation for each node in
 * the computational domain it becomes unfeasible to numerically invert the
 * global matrix \f$A\f$. Instead, we solve the problem iteratively so that we
 * never need to construct \f$A\f$ globally but only need \f$Ax\f$ that can be
 * evaluated locally by virtue of the DG formulation. This action of the
 * operator is what we have to supply in each step of the iterative algorithms
 * implemented here. It is where most of the computational cost goes and usually
 * involves computing a volume contribution for each element and communicating
 * fluxes with neighboring elements. Since the iterative algorithms typically
 * scale badly with increasing grid size, a preconditioner \f$P\f$ is needed
 * in order to make \f$P^{-1}A\f$ easier to invert.
 *
 * In the iterative algorithms we usually don't work with the physical field
 * \f$x\f$ directly. Instead we need to apply the operator to an internal
 * variable defined by the respective algorithm. This variable is exposed as the
 * `LinearSolver::Tags::Operand` prefix, and the algorithm expects that the
 * computed operator action is written into
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * LinearSolver::Tags::Operand<...>>` in each step.
 *
 * Each linear solver is expected to expose the following compile-time
 * interface:
 * - `component_list`: A `tmpl::list` that collects the additional parallel
 * components this linear solver uses. The executables will append these to
 * their own `component_list`.
 * - `initialize_element`: An action that initializes the DataBox items
 * required by the linear solver.
 * - `reinitialize_element`: An action that resets the linear solver to its
 * initial state.
 * - `perform_step`: The action to be executed after the linear operator has
 * been applied to the operand and written to the DataBox (see above). It will
 * converge the fields towards their solution and update the operand before
 * handing responsibility back to the algorithm for the next application of the
 * linear operator:
 * \snippet LinearSolverAlgorithmTestHelpers.hpp action_list
 */

/// \defgroup LoggingGroup Logging
/// \brief Functions for logging progress of running code

/// \defgroup MathFunctionsGroup Math Functions
/// \brief Useful analytic functions

/*!
 * \defgroup NumericalAlgorithmsGroup Numerical Algorithms
 * \brief Generic numerical algorithms
 */

/*!
 * \defgroup NumericalFluxesGroup Numerical Fluxes
 * \brief The set of available numerical fluxes
 */

/*!
 * \defgroup ObserversGroup Observers
 * \brief Observing/writing data to disk.
 */

/*!
 * \defgroup OptionGroupsGroup Option Groups
 * \brief Tags used for grouping input file options.
 *
 * An \ref OptionTagsGroup "option tag" can be placed in a group with other
 * option tags to give the input file more structure. To assign a group to an
 * option tag, set its `group` type alias to a struct that provides a help
 * string and may override a static `name()` function:
 *
 * \snippet Test_Options.cpp options_example_group
 *
 * A number of commonly used groups are listed here.
 *
 * See also the \ref dev_guide_option_parsing "option parsing guide".
 */

/*!
 * \defgroup OptionParsingGroup Option Parsing
 * Things related to parsing YAML input files.
 */

/*!
 * \defgroup OptionTagsGroup Option Tags
 * \brief Tags used for options parsed from the input file.
 *
 * These can be stored in the GlobalCache or passed to the `initialize`
 * function of a parallel component.
 */

/*!
 * \defgroup ParallelGroup Parallelization
 * \brief Functions, classes and documentation related to parallelization and
 * Charm++
 *
 * See
 * \ref dev_guide_parallelization_foundations "Parallelization infrastructure"
 * for details.
 */

/*!
 * \defgroup PeoGroup Performance, Efficiency, and Optimizations
 * \brief Classes and functions useful for performance optimizations.
 */

/*!
 * \defgroup PrettyTypeGroup Pretty Type
 * \brief Pretty printing of types
 */

/*!
 * \defgroup ProtocolsGroup Protocols
 * \brief Classes that define metaprogramming interfaces
 *
 * See the \ref protocols section of the dev guide for details.
 */

/*!
 * \defgroup PythonBindingsGroup Python Bindings
 * \brief Classes and functions useful when writing python bindings.
 *
 * See the \ref spectre_writing_python_bindings "Writing Python Bindings"
 * section of the dev guide for details on how to write python bindings.
 */

/*!
 * \defgroup SpecialRelativityGroup Special Relativity
 * \brief Contains functions used in special relativity calculations
 */

/*!
 * \defgroup SpectralGroup Spectral
 * Things related to spectral transformations.
 */

// Note: this group is ordered by how it appears in the rendered Doxygen pages
// (i.e., "Spin-weighted..."), rather than the group's name (i.e., "Swsh...").
/*!
 * \defgroup SwshGroup Spin-weighted spherical harmonics
 * Utilities, tags, and metafunctions for using and manipulating spin-weighted
 * spherical harmonics
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
 * \snippet Test_TestingFramework.cpp approx_default
 *
 * For checks that need more control over the precision (e.g. an algorithm in
 * which round-off errors accumulate to a higher level), we recommend using
 * the `approx` helper with a one-time tolerance adjustment. A comment
 * should explain the reason for the adjustment:
 * \example
 * \snippet Test_TestingFramework.cpp approx_single_custom
 *
 * For tests in which the same precision adjustment is re-used many times, a new
 * helper object can be created from Catch's `Approx` with a custom precision:
 * \example
 * \snippet Test_TestingFramework.cpp approx_new_custom
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
 * If your error message contains a newline, you can match it using the
 * dot operator `.`, which matches any character.
 * </td>
 * </tr>
 * </table>
 *
 * \example
 * \snippet Test_H5.cpp willfail_example_for_dev_doc
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

/*!
 * \defgroup VariableFixingGroup Variable Fixing
 * \brief A collection of different variable fixers ranging in sophistication.
 *
 * Build-up of numerical error can cause physical quantities to evolve
 * toward non-physical values. For example, pressure and density may become
 * negative. This will subsequently lead to failures in numerical inversion
 * schemes to recover the corresponding convervative values. A rough fix that
 * enforces physical quantities stay physical is to simply change them by hand
 * when needed. This can be done at various degrees of sophistication, but in
 * general the fixed quantities make up a negligible amount of the physics of
 * the simulation; a rough fix is vastly preferred to a simulation that fails
 * to complete due to nonphysical quantities.
 */
