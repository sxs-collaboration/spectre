// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"  // IWYU pragma: keep // for db::item_type<Tags::Mortars<...>>
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace Tags {
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Compute element boundary contributions to the temporal step of the
 * variables
 *
 * This action invokes the numerical flux operator for each mortar of the
 * element, lifts the data to the volume and adds it to the temporal step
 * variables. These would be the time derivatives `Tags::dt` of the system
 * variables for an evolution system, for instance.
 *
 * With:
 * - `flux_comm_types = dg::FluxCommunicationTypes<Metavariables>`
 *
 * Uses:
 * - All items used by `flux_comm_types`
 * - Metavariables:
 *   - `temporal_id::step_prefix`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 * - ConstGlobalCache:
 *   - `Metavariables::normal_dot_numerical_flux`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>`
 *   - `Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>`
 *
 * DataBox changes:
 * - Modifies:
 *   - `db::add_tag_prefix<step_prefix, variables_tag>`
 *   - `flux_comm_types::mortar_data_tag`
 */
struct ApplyFluxes {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using dt_variables_tag =
        db::add_tag_prefix<Metavariables::temporal_id::template step_prefix,
                           variables_tag>;

    using flux_comm_types = FluxCommunicationTypes<Metavariables>;
    using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
    db::mutate<dt_variables_tag, mortar_data_tag>(
        make_not_null(&box),
        [&cache](
            const gsl::not_null<db::item_type<dt_variables_tag>*> dt_vars,
            const gsl::not_null<db::item_type<mortar_data_tag>*> mortar_data,
            const db::item_type<Tags::Mesh<volume_dim>>& mesh,
            const db::item_type<Tags::Mortars<Tags::Mesh<volume_dim - 1>,
                                              volume_dim>>& mortar_meshes,
            const db::item_type<
                Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>>&
                mortar_sizes) noexcept {
          const auto& normal_dot_numerical_flux_computer =
              get<typename Metavariables::normal_dot_numerical_flux>(cache);

          for (auto& this_mortar_data : *mortar_data) {
            const auto& mortar_id = this_mortar_data.first;
            const auto& direction = mortar_id.first;
            const size_t dimension = direction.dimension();

            auto data = this_mortar_data.second.extract();
            auto& local_mortar_data = data.first;
            const auto& remote_mortar_data = data.second;

            db::item_type<dt_variables_tag> lifted_data(
                compute_boundary_flux_contribution<flux_comm_types>(
                    normal_dot_numerical_flux_computer,
                    std::move(local_mortar_data), remote_mortar_data,
                    mesh.slice_away(dimension), mortar_meshes.at(mortar_id),
                    mesh.extents(dimension), mortar_sizes.at(mortar_id)));

            add_slice_to_data(dt_vars, lifted_data, mesh.extents(), dimension,
                              index_to_slice_at(mesh.extents(), direction));
          }
        },
        db::get<Tags::Mesh<volume_dim>>(box),
        db::get<Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>>(box),
        db::get<Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>>(
            box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
