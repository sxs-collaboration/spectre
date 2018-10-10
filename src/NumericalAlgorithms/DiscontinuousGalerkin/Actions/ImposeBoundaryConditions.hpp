// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Packages data on external boundaries for calculating numerical flux.
/// Computes contributions on the interior side from the volume, and imposes
/// Dirichlet boundary conditions on the exterior side.
///
/// With:
/// - Boundary<Tag> =
///   Tags::Interface<Tags::BoundaryDirections<volume_dim>, Tag>
/// - External<Tag> =
///   Tags::Interface<Tags::ExternalBoundaryDirections<volume_dim>, Tag>
///
/// Uses:
/// - ConstGlobalCache:
///   - Metavariables::normal_dot_numerical_flux
///   - Metavariables::boundary_condition
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Boundary<Tags listed in
///               Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - External<Tags listed in
///               Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - Boundary<Tags::Mesh<volume_dim - 1>>
///   - External<Tags::Mesh<volume_dim - 1>>
///   - Boundary<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - External<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - Boundary<Tags::BoundaryCoordinates<volume_dim>>,
///   - Metavariables::temporal_id
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///      - Tags::VariablesBoundaryData
///      - External<typename system::variables_tag>
///
/// \see ReceiveDataForFluxes
template <typename Metavariables>
struct ImposeDirichletBoundaryConditions {
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux,
                 typename Metavariables::boundary_condition_tag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;

    static_assert(
        system::is_conservative or
            cpp17::is_same_v<typename Metavariables::analytic_solution_tag,
                             typename Metavariables::boundary_condition_tag>,
        "Only analytic boundary conditions, or dirichlet boundary conditions "
        "for conservative systems are implemented");

    constexpr size_t volume_dim = system::volume_dim;

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);

    const double time = db::get<Tags::Time>(box).value();

    const auto& boundary_coords = db::get<
        Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
                        Tags::Coordinates<volume_dim, Frame::Inertial>>>(box);

    const auto& boundary_condition =
        get<typename Metavariables::boundary_condition_tag>(cache);

    // Apply the boundary condition
    db::mutate_apply<
        tmpl::list<Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
                                   typename system::variables_tag>>,
        tmpl::list<>>(
        [&boundary_condition, &time, &boundary_coords](
            const gsl::not_null<db::item_type<
                Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
                                typename system::variables_tag>>*>
                external_bdry_vars) noexcept {
          for (auto& external_direction_and_vars : *external_bdry_vars) {
            auto& direction = external_direction_and_vars.first;
            auto& vars = external_direction_and_vars.second;
            vars.assign_subset(boundary_condition.variables(
                boundary_coords.at(direction), time,
                typename system::variables_tag::type::tags_list{}));
          }
        },
        make_not_null(&box));

    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());

      auto interior_data = DgActions_detail::compute_local_mortar_data(
          box, direction, normal_dot_numerical_flux_computer,
          Tags::BoundaryDirectionsInterior<volume_dim>{}, Metavariables{});

      auto exterior_data = DgActions_detail::compute_packaged_data(
          box, direction, normal_dot_numerical_flux_computer,
          Tags::BoundaryDirectionsExterior<volume_dim>{}, Metavariables{});

      db::mutate<Tags::VariablesBoundaryData>(
          make_not_null(&box),
          [&mortar_id, &temporal_id, &interior_data, &exterior_data ](
              const gsl::not_null<
                  db::item_type<Tags::VariablesBoundaryData, DbTags>*>
                  mortar_data) noexcept {
            mortar_data->at(mortar_id).local_insert(temporal_id,
                                                    std::move(interior_data));
            mortar_data->at(mortar_id).remote_insert(
                temporal_id, std::move(exterior_data));
          });
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
