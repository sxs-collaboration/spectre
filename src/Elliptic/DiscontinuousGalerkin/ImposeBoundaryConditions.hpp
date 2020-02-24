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

namespace elliptic {
namespace dg {

/*!
 * \brief Set the `exterior_vars` so that they represent homogeneous (zero)
 * Dirichlet boundary conditions.
 *
 * To impose homogeneous Dirichlet boundary conditions we mirror the
 * `interior_vars` and invert their sign. Variables that are not in the
 * `DirichletTags` list are mirrored without changing their sign, so no boundary
 * conditions are imposed on them.
 */
template <typename DirichletTags, typename TagsList>
void homogeneous_dirichlet_boundary_conditions(
    const gsl::not_null<Variables<TagsList>*> exterior_vars,
    const Variables<TagsList>& interior_vars) noexcept {
  // By default, use the variables on the external boundary for the
  // exterior
  *exterior_vars = interior_vars;
  // For those variables where we have boundary conditions, impose
  // zero Dirichlet b.c. here. The non-zero boundary conditions are
  // handled as contributions to the source in InitializeElement.
  // Imposing them here would not work because we are working with the
  // linear solver operand.
  tmpl::for_each<DirichletTags>(
      [&exterior_vars](auto dirichlet_tag_val) noexcept {
        using dirichlet_tag = tmpl::type_from<decltype(dirichlet_tag_val)>;
        // Use mirror principle
        auto& exterior_dirichlet_field = get<dirichlet_tag>(*exterior_vars);
        for (size_t i = 0; i < exterior_dirichlet_field.size(); i++) {
          exterior_dirichlet_field[i] *= -1.;
        }
      });
}

namespace Actions {

/*!
 * \brief Set field data on external boundaries so that they represent
 * homogeneous (zero) Dirichlet boundary conditions.
 *
 * This action imposes homogeneous boundary conditions on all fields in
 * `system::primal_variables`.
 *
 * \see `elliptic::dg::homogeneous_dirichlet_boundary_conditions`
 *
 * \note We cannot impose inhomogeneous boundary conditions here because it
 * would break linearity of the DG operator: If in the system of equations
 * \f$A(x)=b\f$ the DG operator \f$A\f$ had non-zero boundary contributions then
 * \f$A(x=0)\neq 0\f$, which breaks linearity. Instead, inhomogeneous
 * boundary conditions are handled as contributions to the source \f$b\f$
 * during initialization.
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
 *   - `primal_variables`
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
    using dirichlet_tags = typename system::primal_variables;
    constexpr size_t volume_dim = system::volume_dim;

    // Set the data on exterior (ghost) faces to impose the boundary conditions
    db::mutate<domain::Tags::Interface<
        domain::Tags::BoundaryDirectionsExterior<volume_dim>,
        typename system::variables_tag>>(
        make_not_null(&box),
        // Need to use system::volume_dim below instead of just
        // volume_dim to avoid an ICE on gcc 7.
        [](const gsl::not_null<db::item_type<domain::Tags::Interface<
               domain::Tags::BoundaryDirectionsExterior<system::volume_dim>,
               typename system::variables_tag>>*>
               exterior_boundary_vars,
           const db::const_item_type<domain::Tags::Interface<
               domain::Tags::BoundaryDirectionsInterior<system::volume_dim>,
               typename system::variables_tag>>& interior_vars) noexcept {
          for (auto& exterior_direction_and_vars : *exterior_boundary_vars) {
            auto& direction = exterior_direction_and_vars.first;
            homogeneous_dirichlet_boundary_conditions<dirichlet_tags>(
                make_not_null(&exterior_direction_and_vars.second),
                interior_vars.at(direction));
          }
        },
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<volume_dim>,
            typename system::variables_tag>>(box));

    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    auto interior_data = DgActions_detail::compute_local_mortar_data(
        box, normal_dot_numerical_flux_computer,
        domain::Tags::BoundaryDirectionsInterior<volume_dim>{},
        Metavariables{});

    auto exterior_data = DgActions_detail::compute_packaged_data(
        box, normal_dot_numerical_flux_computer,
        domain::Tags::BoundaryDirectionsExterior<volume_dim>{},
        Metavariables{});

    // Store local and packaged data on the mortars
    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());

      db::mutate<domain::Tags::VariablesBoundaryData>(
          make_not_null(&box),
          [&mortar_id, &temporal_id, &direction, &interior_data,
           &exterior_data](
              const gsl::not_null<
                  db::item_type<domain::Tags::VariablesBoundaryData, DbTags>*>
                  mortar_data) noexcept {
            mortar_data->at(mortar_id).local_insert(
                temporal_id, std::move(interior_data.at(direction)));
            mortar_data->at(mortar_id).remote_insert(
                temporal_id, std::move(exterior_data.at(direction)));
          });
    }
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
