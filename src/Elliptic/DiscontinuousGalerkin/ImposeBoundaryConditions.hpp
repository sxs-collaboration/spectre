// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
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

namespace Elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Packages data on external boundaries so that they represent
 * homogeneous (zero) Dirichlet boundary conditions.
 *
 * This action imposes homogeneous boundary conditions on all fields in
 * `system::impose_boundary_conditions_on_fields`. The fields are wrapped in
 * `LinearSolver::Tags::Operand`. The result should be a subset of the
 * `system::variables`. Because we are working with the linear solver operand,
 * we cannot impose non-zero boundary conditions here. Instead, non-zero
 * boundary conditions are handled as contributions to the linear solver source
 * during initialization.
 *
 * \warning This actions works only for scalar fields right now. It should be
 * considered a temporary solution and will have to be reworked for more
 * involved boundary conditions.
 *
 * With:
 * - `interior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `exterior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>, Tag>`
 *
 * Uses:
 * - Metavariables:
 *   - `normal_dot_numerical_flux`
 *   - `temporal_id`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `impose_boundary_conditions_on_fields`
 * - ConstGlobalCache:
 *   - `normal_dot_numerical_flux`
 * - DataBox:
 *   - `Tags::Element<volume_dim>`
 *   - `temporal_id`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `Tags::BoundaryDirectionsExterior<volume_dim>`
 *   - `interior<variables_tag>`
 *   - `exterior<variables_tag>`
 *   - `interior<normal_dot_numerical_flux::type::argument_tags>`
 *   - `exterior<normal_dot_numerical_flux::type::argument_tags>`
 *
 * DataBox changes:
 * - Modifies:
 *   - `exterior<variables_tag>`
 *   - `Tags::VariablesBoundaryData`
 */
template <typename Metavariables>
struct ImposeHomogeneousDirichletBoundaryConditions {
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using dirichlet_tags =
        typename system::impose_boundary_conditions_on_fields;
    constexpr size_t volume_dim = system::volume_dim;

    // Set the data on exterior (ghost) faces to impose the boundary conditions
    db::mutate<Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
                               typename system::variables_tag>>(
        make_not_null(&box),
        // Need to use system::volume_dim below instead of just
        // volume_dim to avoid an ICE on gcc 7.
        [](const gsl::not_null<db::item_type<Tags::Interface<
               Tags::BoundaryDirectionsExterior<system::volume_dim>,
               typename system::variables_tag>>*>
               exterior_boundary_vars,
           const db::item_type<Tags::Interface<
               Tags::BoundaryDirectionsInterior<system::volume_dim>,
               typename system::variables_tag>>& interior_vars) noexcept {
          for (auto& exterior_direction_and_vars : *exterior_boundary_vars) {
            auto& direction = exterior_direction_and_vars.first;
            auto& exterior_vars = exterior_direction_and_vars.second;

            // By default, use the variables on the external boundary for the
            // exterior
            exterior_vars = interior_vars.at(direction);

            // For those variables where we have boundary conditions, impose
            // zero Dirichlet b.c. here. The non-zero boundary conditions are
            // handled as contributions to the source in InitializeElement.
            // Imposing them here would not work because we are working with the
            // linear solver operand.
            tmpl::for_each<dirichlet_tags>([
              &interior_vars, &exterior_vars, &direction
            ](auto dirichlet_tag_val) noexcept {
              using dirichlet_tag =
                  tmpl::type_from<decltype(dirichlet_tag_val)>;
              using dirichlet_operand_tag =
                  LinearSolver::Tags::Operand<dirichlet_tag>;
              // Use mirror principle. This only works for scalars right now.
              get(get<dirichlet_operand_tag>(exterior_vars)) =
                  -1. *
                  get<dirichlet_operand_tag>(interior_vars.at(direction)).get();
            });
          }
        },
        get<Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
                            typename system::variables_tag>>(box));

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    // Store local and packaged data on the mortars
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
          [&mortar_id, &temporal_id, &interior_data,
           &exterior_data ](const gsl::not_null<
                            db::item_type<Tags::VariablesBoundaryData, DbTags>*>
                                mortar_data) noexcept {
            mortar_data->at(mortar_id).local_insert(temporal_id,
                                                    std::move(interior_data));
            mortar_data->at(mortar_id).remote_insert(temporal_id,
                                                     std::move(exterior_data));
          });
    }
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace Elliptic
