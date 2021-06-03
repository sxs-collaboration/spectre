// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/BoundaryConditionsImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/InternalMortarDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Tags.hpp"
#include "Time/TakeStep.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace evolution::dg::subcell {
// We use a forward declaration instead of including a header file to avoid
// coupling to the DG-subcell libraries for executables that don't use subcell.
template <typename Metavariables, typename DbTagsList>
DirectionMap<Metavariables::volume_dim, std::vector<double>>
prepare_neighbor_data(gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept;
}  // namespace evolution::dg::subcell
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Actions {
/*!
 * \brief Computes the time derivative for a DG time step.
 *
 * Computes the volume fluxes, the divergence of the fluxes and all additional
 * interior contributions to the time derivatives (both nonconservative products
 * and source terms). The internal mortar data is also computed.
 *
 * The general first-order hyperbolic evolution equation solved for conservative
 * systems is:
 *
 * \f{align*}{
 * \frac{\partial u_\alpha}{\partial \hat{t}}
 *  + \partial_{i}
 *   \left(F^i_\alpha - v^i_g u_\alpha\right)
 *   = S_\alpha-u_\alpha\partial_i v^i_g,
 * \f}
 *
 * where \f$F^i_{\alpha}\f$ are the fluxes when the mesh isn't moving,
 * \f$v^i_g\f$ is the velocity of the mesh, \f$u_{\alpha}\f$ are the evolved
 * variables, \f$S_{\alpha}\f$ are the source terms, and \f$\hat{t}\f$ is the
 * time in the logical frame. For evolution equations that do not have any
 * fluxes and only nonconservative products we evolve:
 *
 * \f{align*}{
 * \frac{\partial u_\alpha}{\partial \hat{t}}
 *   +\left(B^i_{\alpha\beta}-v^i_g \delta_{\alpha\beta}
 *   \right)\partial_{i}u_\beta = S_\alpha.
 * \f}
 *
 * Finally, for equations with both conservative terms and nonconservative
 * products we use:
 *
 * \f{align*}{
 * \frac{\partial u_\alpha}{\partial \hat{t}}
 *   + \partial_{i}
 *   \left(F^i_\alpha - v^i_g u_\alpha\right)
 *   +B^i_{\alpha\beta}\partial_{i}u_\beta
 *   = S_\alpha-u_\alpha\partial_i v^i_g,
 * \f}
 *
 * where \f$B^i_{\alpha\beta}\f$ is the matrix for the nonconservative products.
 *
 * ### Volume Terms
 *
 * The mesh velocity is added to the flux automatically if the mesh is moving.
 * That is,
 *
 * \f{align*}{
 *  F^i_{\alpha}\to F^i_{\alpha}-v^i_{g} u_{\alpha}
 * \f}
 *
 * The source terms are also altered automatically by adding:
 *
 * \f{align*}{
 *  -u_\alpha \partial_i v^i_g,
 * \f}
 *
 * For systems with equations that only contain nonconservative products, the
 * following mesh velocity is automatically added to the time derivative:
 *
 * \f{align*}{
 *  v^i_g \partial_i u_\alpha,
 * \f}
 *
 * \note The term is always added in the `Frame::Inertial` frame, and the plus
 * sign arises because we add it to the time derivative.
 *
 * Here are examples of the `TimeDerivative` struct used to compute the volume
 * time derivative. This struct is what the type alias
 * `System::compute_volume_time_derivative` points to. The time derivatives are
 * as `gsl::not_null` first, then the temporary tags as `gsl::not_null`,
 * followed by the `argument_tags`. These type aliases are given by
 *
 * \snippet ComputeTimeDerivativeImpl.tpp dt_ta
 *
 * for the examples. For a conservative system without primitives the `apply`
 * function would look like
 *
 * \snippet ComputeTimeDerivativeImpl.tpp dt_con
 *
 * For a nonconservative system it would be
 *
 * \snippet ComputeTimeDerivativeImpl.tpp dt_nc
 *
 * And finally, for a mixed conservative-nonconservative system with primitive
 * variables
 *
 * \snippet ComputeTimeDerivativeImpl.tpp dt_mp
 *
 * Uses:
 * - System:
 *   - `variables_tag`
 *   - `flux_variables`
 *   - `gradient_variables`
 *   - `compute_volume_time_derivative_terms`
 *
 * - DataBox:
 *   - Items in `system::compute_volume_time_derivative_terms::argument_tags`
 *   - `domain::Tags::MeshVelocity<Metavariables::volume_dim>`
 *   - `Metavariables::system::variables_tag`
 *   - `Metavariables::system::flux_variables`
 *   - `Metavariables::system::gradient_variables`
 *   - `domain::Tags::DivMeshVelocity`
 *   - `DirectionsTag`,
 *   - Required interface items for `Metavariables::system::normal_dot_fluxes`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - db::add_tag_prefix<Tags::Flux, variables_tag,
 *                        tmpl::size_t<system::volume_dim>, Frame::Inertial>
 *   - `Tags::dt<system::variable_tags>`
 *   - Tags::Interface<
 *     DirectionsTag, db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>
 *   - `Tags::Mortars<typename BoundaryScheme::mortar_data_tag, VolumeDim>`
 *
 * ### Internal Boundary Terms
 *
 * Internal boundary terms are computed from the
 * `System::boundary_correction_base` type alias. This type alias must point to
 * a base class with `creatable_classes`. Each concrete boundary correction must
 * specify:
 *
 * - type alias template `dg_package_field_tags`. These are what will be
 *   returned by `gsl::not_null` from the `dg_package_data` member function.
 *
 * - type alias template `dg_package_temporary_tags`. These are temporary tags
 *   that are projected to the face and then passed to the `dg_package_data`
 *   function.
 *
 * - type alias template `dg_package_primitive_tags`. These are the primitive
 *   variables (if any) that are projected to the face and then passed to
 *   `dg_package_data`.
 *
 * - type alias template `dg_package_volume_tags`. These are tags that are not
 *   projected to the interface and are retrieved directly from the `DataBox`.
 *   The equation of state for hydrodynamics systems is an example of what would
 *   be a "volume tag".
 *
 * A `static constexpr bool need_normal_vector` must be specified. If `true`
 * then the normal vector is computed from the normal covector. This is
 * currently not implemented.
 *
 * The `dg_package_data` function takes as arguments `gsl::not_null` of the
 * `dg_package_data_field_tags`, then the projected evolved variables, the
 * projected fluxes, the projected temporaries, the projected primitives, the
 * unit normal covector, mesh velocity, normal dotted into the mesh velocity,
 * the `volume_tags`, and finally the `dg::Formulation`. The `dg_package_data`
 * function must compute all ingredients for the boundary correction, including
 * mesh-velocity-corrected characteristic speeds. However, the projected fluxes
 * passed in are \f$F^i - u v^i_g\f$ (the mesh velocity term is already
 * included). The `dg_package_data` function must also return a `double` that is
 * the maximum absolute characteristic speed over the entire face. This will be
 * used for checking that the time step doesn't violate the CFL condition.
 *
 * Here is an example of the type aliases and `bool`:
 *
 * \snippet ComputeTimeDerivativeImpl.tpp bt_ta
 *
 * The normal vector requirement is:
 *
 * \snippet ComputeTimeDerivativeImpl.tpp bt_nnv
 *
 * For a conservative system with primitive variables and using the `TimeStepId`
 * as a volume tag the `dg_package_data` function looks like:
 *
 * \snippet ComputeTimeDerivativeImpl.tpp bt_cp
 *
 * For a mixed conservative-nonconservative system with primitive variables and
 * using the `TimeStepId` as a volume tag the `dg_package_data` function looks
 * like:
 *
 * \snippet ComputeTimeDerivativeImpl.tpp bt_mp
 *
 * Uses:
 * - System:
 *   - `boundary_correction`
 *   - `variables_tag`
 *   - `flux_variables`
 *   - `gradients_tags`
 *   - `compute_volume_time_derivative`
 *   - `has_primitive_and_conservative_vars`
 *   - `primitive_variables_tag` if system has primitive variables
 *
 * - DataBox:
 *   - `domain::Tags::Element<Dim>`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `evolution::dg::Tags::MortarMesh<Dim>`
 *   - `evolution::dg::Tags::MortarSize<Dim>`
 *   - `evolution::dg::Tags::MortarData<Dim>`
 *   - `Tags::TimeStepId`
 *   - \code{.cpp}
 *      domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
 *                                       domain::Tags::Mesh<Dim - 1>>
 *     \endcode
 *   - \code{.cpp}
 *     domain::Tags::Interface<
 *         domain::Tags::InternalDirections<Dim>,
 *         ::Tags::Normalized<
 *             domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>
 *     \endcode
 *   - \code{.cpp}
 *     domain::Tags::Interface<
 *              domain::Tags::InternalDirections<Dim>,
 *              domain::Tags::MeshVelocity<Dim, Frame::Inertial>>
 *     \endcode
 *   - `Metavariables::system::variables_tag`
 *   - `Metavariables::system::flux_variables`
 *   - `Metavariables::system::primitive_tags` if exists
 *   - `system::boundary_correction_base::dg_package_data_volume_tags`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `evolution::dg::Tags::MortarData<Dim>`
 */
template <typename Metavariables>
struct ComputeTimeDerivative {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Metavariables::volume_dim>>;
  using const_global_cache_tags = tmpl::append<
      tmpl::list<::dg::Tags::Formulation, evolution::Tags::BoundaryCorrection<
                                              typename Metavariables::system>>,
      tmpl::conditional_t<
          Metavariables::local_time_stepping,
          tmpl::list<::Tags::StepChoosers, ::Tags::StepController,
                     ::Tags::TimeStepper<LtsTimeStepper>>,
          tmpl::list<>>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* /*meta*/) noexcept;  // NOLINT const

 private:
  // The below functions will be removed once we've finished writing the new
  // method of handling boundaries.
  template <typename... NormalDotFluxTags, typename... Args>
  static void apply_flux(
      gsl::not_null<Variables<tmpl::list<NormalDotFluxTags...>>*> boundary_flux,
      const Args&... boundary_variables) noexcept {
    Metavariables::system::normal_dot_fluxes::apply(
        make_not_null(&get<NormalDotFluxTags>(*boundary_flux))...,
        boundary_variables...);
  }

  template <typename DbTagsList>
  static void boundary_terms_nonconservative_products(
      gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept;

  template <size_t VolumeDim, typename BoundaryScheme, typename DbTagsList>
  static void fill_mortar_data_for_internal_boundaries(
      gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept;

  template <typename ParallelComponent, typename DbTagsList>
  static void send_data_for_fluxes(
      gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept;
};

template <typename Metavariables>
template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
          typename ActionList, typename ParallelComponent>
std::tuple<db::DataBox<DbTagsList>&&>
ComputeTimeDerivative<Metavariables>::apply(
    db::DataBox<DbTagsList>& box,
    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
    Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
    const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using system = typename Metavariables::system;
  using variables_tag = typename system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using partial_derivative_tags = typename system::gradient_variables;
  using flux_variables = typename system::flux_variables;
  using compute_volume_time_derivative_terms =
      typename system::compute_volume_time_derivative_terms;

  const Mesh<volume_dim>& mesh = db::get<::domain::Tags::Mesh<volume_dim>>(box);
  const ::dg::Formulation dg_formulation =
      db::get<::dg::Tags::Formulation>(box);
  ASSERT(alg::all_of(mesh.basis(),
                     [&mesh](const Spectral::Basis current_basis) noexcept {
                       return current_basis == mesh.basis(0);
                     }),
         "An isotropic basis must be used in the evolution code. While "
         "theoretically this restriction could be lifted, the simplification "
         "it offers are quite substantial. Relaxing this assumption is likely "
         "to require quite a bit of careful code refactoring and debugging.");
  ASSERT(alg::all_of(
             mesh.quadrature(),
             [&mesh](const Spectral::Quadrature current_quadrature) noexcept {
               return current_quadrature == mesh.quadrature(0);
             }),
         "An isotropic quadrature must be used in the evolution code. While "
         "theoretically this restriction could be lifted, the simplification "
         "it offers are quite substantial. Relaxing this assumption is likely "
         "to require quite a bit of careful code refactoring and debugging.");

  // Allocate the Variables classes needed for the time derivative
  // computation.
  //
  // This is factored out so that we will be able to do ADER-DG/CG where a
  // spacetime polynomial is constructed by solving implicit equations in time
  // using a Picard iteration. A high-order initial guess is needed to
  // efficiently construct the ADER spacetime solution. This initial guess is
  // obtained using continuous RK methods, and so we will want to reuse
  // buffers. Thus, the volume_terms function returns by reference rather than
  // by value.
  Variables<typename compute_volume_time_derivative_terms::temporary_tags>
      temporaries{mesh.number_of_grid_points()};
  Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                             tmpl::size_t<volume_dim>, Frame::Inertial>>
      volume_fluxes{mesh.number_of_grid_points()};
  Variables<db::wrap_tags_in<::Tags::deriv, partial_derivative_tags,
                             tmpl::size_t<volume_dim>, Frame::Inertial>>
      partial_derivs{mesh.number_of_grid_points()};

  const Scalar<DataVector>* det_inverse_jacobian = nullptr;
  if constexpr (tmpl::size<flux_variables>::value != 0) {
    if (dg_formulation == ::dg::Formulation::WeakInertial) {
      det_inverse_jacobian = &db::get<
          domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(box);
    }
  }
  db::mutate_apply<
      tmpl::list<dt_variables_tag>,
      typename compute_volume_time_derivative_terms::argument_tags>(
      [&dg_formulation, &det_inverse_jacobian,
       &div_mesh_velocity = db::get<::domain::Tags::DivMeshVelocity>(box),
       &evolved_variables = db::get<variables_tag>(box),
       &inertial_coordinates =
           db::get<domain::Tags::Coordinates<volume_dim, Frame::Inertial>>(box),
       &logical_to_inertial_inv_jacobian =
           db::get<::domain::Tags::InverseJacobian<volume_dim, Frame::Logical,
                                                   Frame::Inertial>>(box),
       &mesh,
       &mesh_velocity = db::get<::domain::Tags::MeshVelocity<volume_dim>>(box),
       &partial_derivs, &temporaries, &volume_fluxes](
          const gsl::not_null<Variables<
              db::wrap_tags_in<::Tags::dt, typename variables_tag::tags_list>>*>
              dt_vars_ptr,
          const auto&... time_derivative_args) noexcept {
        detail::volume_terms<compute_volume_time_derivative_terms>(
            dt_vars_ptr, make_not_null(&volume_fluxes),
            make_not_null(&partial_derivs), make_not_null(&temporaries),
            evolved_variables, dg_formulation, mesh, inertial_coordinates,
            logical_to_inertial_inv_jacobian, det_inverse_jacobian,
            mesh_velocity, div_mesh_velocity, time_derivative_args...);
      },
      make_not_null(&box));

  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection<system>>(box);
  using derived_boundary_corrections =
      typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

  const Variables<detail::get_primitive_vars_tags_from_system<system>>*
      primitive_vars{nullptr};
  if constexpr (system::has_primitive_and_conservative_vars) {
    primitive_vars = &db::get<typename system::primitive_variables_tag>(box);
  }

  static_assert(
      tmpl::all<derived_boundary_corrections, std::is_final<tmpl::_1>>::value,
      "All createable classes for boundary corrections must be marked "
      "final.");
  tmpl::for_each<derived_boundary_corrections>(
      [&boundary_correction, &box, &partial_derivs, &primitive_vars,
       &temporaries, &volume_fluxes](auto derived_correction_v) noexcept {
        using DerivedCorrection =
            tmpl::type_from<decltype(derived_correction_v)>;
        if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
          // Compute internal boundary quantities on the mortar for sides
          // of the element that have neighbors, i.e. they are not an
          // external side.
          // Note: this call mutates:
          //  - evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
          //  - evolution::dg::Tags::MortarData<Dim>
          detail::internal_mortar_data<system, volume_dim>(
              make_not_null(&box),
              dynamic_cast<const DerivedCorrection&>(boundary_correction),
              db::get<variables_tag>(box), volume_fluxes, temporaries,
              primitive_vars,
              typename DerivedCorrection::dg_package_data_volume_tags{});

          detail::apply_boundary_conditions_on_all_external_faces<system,
                                                                  volume_dim>(
              make_not_null(&box),
              dynamic_cast<const DerivedCorrection&>(boundary_correction),
              temporaries, volume_fluxes, partial_derivs, primitive_vars);
        }
      });

  if constexpr (Metavariables::local_time_stepping) {
    take_step(make_not_null(&box), cache);
  }

  send_data_for_fluxes<ParallelComponent>(make_not_null(&cache),
                                          make_not_null(&box));
  return {std::move(box)};
}

template <typename Metavariables>
template <typename ParallelComponent, typename DbTagsList>
void ComputeTimeDerivative<Metavariables>::send_data_for_fluxes(
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
  constexpr size_t volume_dim = Metavariables::volume_dim;
  auto& receiver_proxy =
      Parallel::get_parallel_component<ParallelComponent>(*cache);
  const auto& element = db::get<domain::Tags::Element<volume_dim>>(*box);

    const auto& time_step_id = db::get<::Tags::TimeStepId>(*box);
    const auto& all_mortar_data =
        db::get<evolution::dg::Tags::MortarData<volume_dim>>(*box);
    const auto& mortar_meshes =
        get<evolution::dg::Tags::MortarMesh<volume_dim>>(*box);

    std::optional<DirectionMap<volume_dim, std::vector<double>>>
        all_neighbor_data_for_reconstruction = std::nullopt;
    if constexpr (using_subcell_v<Metavariables>) {
      all_neighbor_data_for_reconstruction =
          evolution::dg::subcell::prepare_neighbor_data<Metavariables>(box);
    }

    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      std::vector<double> ghost_and_subcell_data{};
      if constexpr (using_subcell_v<Metavariables>) {
        ASSERT(all_neighbor_data_for_reconstruction.has_value(),
               "Trying to do DG-subcell but the ghost and subcell data for the "
               "neighbor has not been set.");
        ghost_and_subcell_data =
            std::move(all_neighbor_data_for_reconstruction.value()[direction]);
      }

      for (const auto& neighbor : neighbors) {
        const std::pair mortar_id{direction, neighbor};

        std::pair<Mesh<volume_dim - 1>, std::vector<double>>
            neighbor_boundary_data_on_mortar{};
        ASSERT(time_step_id == all_mortar_data.at(mortar_id).time_step_id(),
               "The current time step id of the volume is "
                   << time_step_id
                   << "but the time step id on the mortar with mortar id "
                   << mortar_id << " is "
                   << all_mortar_data.at(mortar_id).time_step_id());

        // Reorient the data to the neighbor orientation if necessary
        if (LIKELY(orientation.is_aligned())) {
          neighbor_boundary_data_on_mortar =
              *all_mortar_data.at(mortar_id).local_mortar_data();
        } else {
          const auto& slice_extents = mortar_meshes.at(mortar_id).extents();
          neighbor_boundary_data_on_mortar.first =
              all_mortar_data.at(mortar_id).local_mortar_data()->first;
          neighbor_boundary_data_on_mortar.second = orient_variables_on_slice(
              all_mortar_data.at(mortar_id).local_mortar_data()->second,
              slice_extents, direction.dimension(), orientation);
        }

        const TimeStepId& next_time_step_id = [&box]() noexcept {
          if (Metavariables::local_time_stepping) {
            return db::get<::Tags::Next<::Tags::TimeStepId>>(*box);
          } else {
            return db::get<::Tags::TimeStepId>(*box);
          }
        }();

        std::tuple<Mesh<volume_dim - 1>, std::optional<std::vector<double>>,
                   std::optional<std::vector<double>>, ::TimeStepId>
            data{neighbor_boundary_data_on_mortar.first,
                 ghost_and_subcell_data,
                 {std::move(neighbor_boundary_data_on_mortar.second)},
                 next_time_step_id};

        // Send mortar data (the `std::tuple` named `data`) to neighbor
        Parallel::receive_data<
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                volume_dim>>(
            receiver_proxy[neighbor], time_step_id,
            std::make_pair(std::pair{direction_from_neighbor, element.id()},
                           data));
      }
    }

    if constexpr (Metavariables::local_time_stepping) {
      using variables_tag = typename Metavariables::system::variables_tag;
      using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
      // We assume isotropic quadrature, i.e. the quadrature is the same in
      // all directions.
      const bool using_gauss_points =
          db::get<domain::Tags::Mesh<volume_dim>>(*box).quadrature() ==
          make_array<volume_dim>(Spectral::Quadrature::Gauss);

      const Scalar<DataVector> volume_det_inv_jacobian{};
      if (using_gauss_points) {
        // NOLINTNEXTLINE
        const_cast<DataVector&>(get(volume_det_inv_jacobian))
            .set_data_ref(make_not_null(&const_cast<DataVector&>(  // NOLINT
                get(db::get<domain::Tags::DetInvJacobian<
                        Frame::Logical, Frame::Inertial>>(*box)))));
      }

      // Add face normal and Jacobian determinants to the local mortar data. We
      // only need the Jacobians if we are using Gauss points. Then copy over
      // into the boundary history, since that's what the LTS steppers use.
      //
      // The boundary history coupling computation (which computes the _lifted_
      // boundary correction) returns a Variables<dt<EvolvedVars>> instead of
      // using the `NormalDotNumericalFlux` prefix tag. This is because the
      // returned quantity is more a `dt` quantity than a
      // `NormalDotNormalDotFlux` since it's been lifted to the volume.
      using Key = std::pair<Direction<volume_dim>, ElementId<volume_dim>>;
      const auto integration_order =
          db::get<::Tags::HistoryEvolvedVariables<>>(*box).integration_order();
      db::mutate<evolution::dg::Tags::MortarData<volume_dim>,
                 evolution::dg::Tags::MortarDataHistory<
                     volume_dim, typename dt_variables_tag::type>>(
          box,
          [&element, integration_order, &time_step_id, using_gauss_points,
           &volume_det_inv_jacobian](
              const gsl::not_null<
                  std::unordered_map<Key, evolution::dg::MortarData<volume_dim>,
                                     boost::hash<Key>>*>
                  mortar_data,
              const gsl::not_null<
                  std::unordered_map<Key,
                                     TimeSteppers::BoundaryHistory<
                                         evolution::dg::MortarData<volume_dim>,
                                         evolution::dg::MortarData<volume_dim>,
                                         typename dt_variables_tag::type>,
                                     boost::hash<Key>>*>
                  boundary_data_history,
              const Mesh<volume_dim>& volume_mesh,
              const DirectionMap<
                  volume_dim,
                  std::optional<Variables<tmpl::list<
                      evolution::dg::Tags::MagnitudeOfNormal,
                      evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
                  normal_covector_and_magnitude) noexcept {
            Scalar<DataVector> volume_det_jacobian{};
            Scalar<DataVector> face_det_jacobian{};
            if (using_gauss_points) {
              get(volume_det_jacobian) = 1.0 / get(volume_det_inv_jacobian);
            }
            for (const auto& [direction, neighbors_in_direction] :
                 element.neighbors()) {
              // We can perform projections once for all neighbors in the
              // direction because we care about the _face_ mesh, not the mortar
              // mesh.
              ASSERT(
                  normal_covector_and_magnitude.at(direction).has_value(),
                  "The normal covector and magnitude have not been computed.");
              const Scalar<DataVector>& face_normal_magnitude =
                  get<evolution::dg::Tags::MagnitudeOfNormal>(
                      *normal_covector_and_magnitude.at(direction));
              if (using_gauss_points) {
                const Matrix identity{};
                auto interpolation_matrices =
                    make_array<Metavariables::volume_dim>(std::cref(identity));
                const std::pair<Matrix, Matrix>& matrices =
                    Spectral::boundary_interpolation_matrices(
                        volume_mesh.slice_through(direction.dimension()));
                gsl::at(interpolation_matrices, direction.dimension()) =
                    direction.side() == Side::Upper ? matrices.second
                                                    : matrices.first;
                if (get(face_det_jacobian).size() !=
                    get(face_normal_magnitude).size()) {
                  get(face_det_jacobian) =
                      DataVector{get(face_normal_magnitude).size()};
                }
                apply_matrices(make_not_null(&get(face_det_jacobian)),
                               interpolation_matrices, get(volume_det_jacobian),
                               volume_mesh.extents());
              }

              for (const auto& neighbor : neighbors_in_direction) {
                const std::pair mortar_id{direction, neighbor};
                if (using_gauss_points) {
                  mortar_data->at(mortar_id).insert_local_geometric_quantities(
                      volume_det_inv_jacobian, face_det_jacobian,
                      face_normal_magnitude);
                } else {
                  mortar_data->at(mortar_id).insert_local_face_normal_magnitude(
                      face_normal_magnitude);
                }
                ASSERT(boundary_data_history->count(mortar_id) != 0,
                       "Could not insert the mortar data for "
                           << mortar_id
                           << " because the unordered map has not been "
                              "initialized "
                              "to have the mortar id.");
                boundary_data_history->at(mortar_id).local_insert(
                    time_step_id, std::move(mortar_data->at(mortar_id)));
                boundary_data_history->at(mortar_id).integration_order(
                    integration_order);
                mortar_data->at(mortar_id) =
                    MortarData<Metavariables::volume_dim>{};
              }
            }
          },
          db::get<domain::Tags::Mesh<volume_dim>>(*box),
          db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<volume_dim>>(
              *box));
  }
}
}  // namespace evolution::dg::Actions
