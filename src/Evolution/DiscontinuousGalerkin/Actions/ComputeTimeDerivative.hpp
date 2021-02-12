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
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/InternalMortarDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
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
 * Internal boundary terms are computed from the `System::boundary_correction`
 * type alias. This type alias must point to a base class with
 * `creatable_classes`. Each concrete boundary correction must specify:
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
 *   - `Metavariables::system::boundary_correction::dg_package_data_volume_tags`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `evolution::dg::Tags::MortarData<Dim>`
 */
template <typename Metavariables>
struct ComputeTimeDerivative {
 private:
  template <typename LocalMetaVars, typename = std::void_t<>>
  struct GetFluxInboxTag {
    using type = tmpl::list<>;
  };

  template <typename LocalMetaVars>
  struct GetFluxInboxTag<LocalMetaVars,
                         std::void_t<typename LocalMetaVars::boundary_scheme>> {
    using type = tmpl::list<
        ::dg::FluxesInboxTag<typename Metavariables::boundary_scheme>>;
  };

 public:
  using inbox_tags = tmpl::append<
      typename GetFluxInboxTag<Metavariables>::type,
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Metavariables::volume_dim>>>;
  using const_global_cache_tags = tmpl::conditional_t<
      detail::has_boundary_correction_v<typename Metavariables::system>,
      tmpl::list<::dg::Tags::Formulation, evolution::Tags::BoundaryCorrection<
                                              typename Metavariables::system>>,
      tmpl::list<::dg::Tags::Formulation>>;

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
      const db::DataBox<DbTagsList>& box) noexcept;
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

  if constexpr (detail::has_boundary_correction_v<system>) {
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
        [&boundary_correction, &box, &primitive_vars, &temporaries,
         &volume_fluxes](auto derived_correction_v) noexcept {
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
          }
        });
  } else {
    // The below if-else and fill_mortar_data_for_internal_boundaries are for
    // compatibility with the current boundary schemes. Once the current
    // boundary schemes are removed the code will be refactored to reflect that
    // change.
    if constexpr (not std::is_same_v<tmpl::list<>, flux_variables>) {
      using flux_variables_tag = ::Tags::Variables<flux_variables>;
      using fluxes_tag =
          db::add_tag_prefix<::Tags::Flux, flux_variables_tag,
                             tmpl::size_t<Metavariables::volume_dim>,
                             Frame::Inertial>;

      // We currently set the fluxes in the DataBox to interface with the
      // boundary correction communication code.
      db::mutate<fluxes_tag>(make_not_null(&box),
                             [&volume_fluxes](const auto fluxes_ptr) noexcept {
                               *fluxes_ptr = volume_fluxes;
                             });
    } else {
      boundary_terms_nonconservative_products(make_not_null(&box));
    }

    // Compute internal boundary quantities
    fill_mortar_data_for_internal_boundaries<
        volume_dim, typename Metavariables::boundary_scheme>(
        make_not_null(&box));
  }

  send_data_for_fluxes<ParallelComponent>(make_not_null(&cache), box);
  return {std::move(box)};
}

template <typename Metavariables>
template <typename DbTagsList>
void ComputeTimeDerivative<Metavariables>::
    boundary_terms_nonconservative_products(
        const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
  using system = typename Metavariables::system;
  using variables_tag = typename system::variables_tag;
  using DirectionsTag =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;

  using interface_normal_dot_fluxes_tag = domain::Tags::Interface<
      DirectionsTag, db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>;

  using interface_compute_item_argument_tags = tmpl::transform<
      typename Metavariables::system::normal_dot_fluxes::argument_tags,
      tmpl::bind<domain::Tags::Interface, DirectionsTag, tmpl::_1>>;

  db::mutate_apply<
      tmpl::list<interface_normal_dot_fluxes_tag>,
      tmpl::push_front<interface_compute_item_argument_tags, DirectionsTag>>(
      [](const auto boundary_fluxes_ptr,
         const std::unordered_set<Direction<Metavariables::volume_dim>>&
             internal_directions,
         const auto&... tensors) noexcept {
        for (const auto& direction : internal_directions) {
          apply_flux(make_not_null(&boundary_fluxes_ptr->at(direction)),
                     tensors.at(direction)...);
        }
      },
      box);
}

template <typename Metavariables>
template <size_t VolumeDim, typename BoundaryScheme, typename DbTagsList>
void ComputeTimeDerivative<Metavariables>::
    fill_mortar_data_for_internal_boundaries(
        const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, VolumeDim>;
  using boundary_data_computer =
      typename BoundaryScheme::boundary_data_computer;

  // Collect data on element interfaces
  auto boundary_data_on_interfaces =
      interface_apply<domain::Tags::InternalDirections<VolumeDim>,
                      boundary_data_computer>(*box);

  // Project collected data to all internal mortars and store in DataBox
  const auto& element = db::get<domain::Tags::Element<VolumeDim>>(*box);
  const auto& face_meshes =
      get<domain::Tags::Interface<domain::Tags::InternalDirections<VolumeDim>,
                                  domain::Tags::Mesh<VolumeDim - 1>>>(*box);
  const auto& mortar_meshes =
      get<::Tags::Mortars<domain::Tags::Mesh<VolumeDim - 1>, VolumeDim>>(*box);
  const auto& mortar_sizes =
      get<::Tags::Mortars<::Tags::MortarSize<VolumeDim - 1>, VolumeDim>>(*box);
  const auto& temporal_id = get<temporal_id_tag>(*box);
  const auto integration_order =
      db::get<::Tags::HistoryEvolvedVariables<>>(*box).integration_order();
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const auto& face_mesh = face_meshes.at(direction);
    for (const auto& neighbor : neighbors) {
      const auto mortar_id = std::make_pair(direction, neighbor);
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);

      // Project the data from the face to the mortar.
      // Where no projection is necessary we `std::move` the data directly to
      // avoid a copy. We can't move the data or modify it in-place when
      // projecting, because in that case the face may touch more than one
      // mortar so we need to keep the data around.
      auto boundary_data_on_mortar =
          ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? boundary_data_on_interfaces.at(direction).project_to_mortar(
                    face_mesh, mortar_mesh, mortar_size)
              : std::move(boundary_data_on_interfaces.at(direction));

      // Store the boundary data on this side of the mortar
      db::mutate<all_mortar_data_tag>(
          box, [&mortar_id, &temporal_id, &boundary_data_on_mortar,
                &integration_order](
                   const gsl::not_null<typename all_mortar_data_tag::type*>
                       all_mortar_data) noexcept {
              auto& mortar_data = all_mortar_data->at(mortar_id);
              mortar_data.integration_order(integration_order);
              mortar_data.local_insert(temporal_id,
                                       std::move(boundary_data_on_mortar));
          });
    }
  }
}

template <typename Metavariables>
template <typename ParallelComponent, typename DbTagsList>
void ComputeTimeDerivative<Metavariables>::send_data_for_fluxes(
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const db::DataBox<DbTagsList>& box) noexcept {
  using system = typename Metavariables::system;
  constexpr size_t volume_dim = Metavariables::volume_dim;
  auto& receiver_proxy =
      Parallel::get_parallel_component<ParallelComponent>(*cache);
  const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);

  if constexpr (detail::has_boundary_correction_v<system>) {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    const auto& all_mortar_data =
        db::get<evolution::dg::Tags::MortarData<volume_dim>>(box);
    const auto& mortar_meshes =
        get<evolution::dg::Tags::MortarMesh<volume_dim>>(box);

    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

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

        std::tuple<Mesh<volume_dim - 1>, std::optional<std::vector<double>>,
                   std::optional<std::vector<double>>, ::TimeStepId>
            data{neighbor_boundary_data_on_mortar.first,
                 {},
                 {std::move(neighbor_boundary_data_on_mortar.second)},
                 db::get<tmpl::conditional_t<Metavariables::local_time_stepping,
                                             ::Tags::Next<::Tags::TimeStepId>,
                                             ::Tags::TimeStepId>>(box)};

        // Send mortar data (the `std::tuple` named `data`) to neighbor
        Parallel::receive_data<
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                volume_dim>>(
            receiver_proxy[neighbor], time_step_id,
            std::make_pair(std::pair{direction_from_neighbor, element.id()},
                           data));
      }
    }
  } else {
    using BoundaryScheme = typename Metavariables::boundary_scheme;
    using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
    using receive_temporal_id_tag =
        typename BoundaryScheme::receive_temporal_id_tag;
    using fluxes_inbox_tag = ::dg::FluxesInboxTag<BoundaryScheme>;
    using all_mortar_data_tag =
        ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;

    const auto& all_mortar_data = get<all_mortar_data_tag>(box);
    const auto& temporal_id = db::get<temporal_id_tag>(box);
    const auto& receive_temporal_id = db::get<receive_temporal_id_tag>(box);
    const auto& mortar_meshes = db::get<
        ::Tags::Mortars<domain::Tags::Mesh<volume_dim - 1>, volume_dim>>(box);

    // Iterate over neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_and_neighbors.second;
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      for (const auto& neighbor : neighbors_in_direction) {
        const ::dg::MortarId<volume_dim> mortar_id{direction, neighbor};

        // Make a copy of the local boundary data on the mortar to send to the
        // neighbor
        ASSERT(all_mortar_data.find(mortar_id) != all_mortar_data.end(),
               "Mortar data on mortar "
                   << mortar_id
                   << " not available for sending. Did you forget to collect "
                      "the data on mortars?");
        auto neighbor_boundary_data_on_mortar =
            all_mortar_data.at(mortar_id).local_data(temporal_id);

        // Reorient the data to the neighbor orientation if necessary
        if (not orientation.is_aligned()) {
          neighbor_boundary_data_on_mortar.orient_on_slice(
              mortar_meshes.at(mortar_id).extents(), dimension, orientation);
        }

        // Send mortar data (named 'neighbor_boundary_data_on_mortar') to
        // neighbor
        Parallel::receive_data<fluxes_inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                ::dg::MortarId<volume_dim>{direction_from_neighbor,
                                           element.id()},
                std::make_pair(receive_temporal_id,
                               std::move(neighbor_boundary_data_on_mortar))));
      }
    }
  }
}
}  // namespace evolution::dg::Actions
