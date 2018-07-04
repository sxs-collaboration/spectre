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
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <typename TagsList>
class Variables;
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
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Compute boundary contribution to the time derivative for
/// use in global time stepping.
///
/// Uses:
/// - ConstGlobalCache: Metavariables::normal_dot_numerical_flux
/// - DataBox: Tags::Mesh<volume_dim>,
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   db::add_tag_prefix<Tags::dt, variables_tag>,
///   FluxCommunicationTypes<Metavariables>::mortar_data_tag
struct ApplyBoundaryFluxesGlobalTimeStepping {
 private:
  template <typename Flux, typename... NumericalFluxTags, typename... SelfTags,
            typename... PackagedTags, typename... ArgumentTags,
            typename... Args>
  static void apply_normal_dot_numerical_flux(
      const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
          numerical_fluxes,
      const Flux& flux,
      const Variables<tmpl::list<SelfTags...>>& self_packaged_data,
      const Variables<tmpl::list<PackagedTags...>>&
          neighbor_packaged_data) noexcept {
    flux(make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
         get<PackagedTags>(self_packaged_data)...,
         get<PackagedTags>(neighbor_packaged_data)...);
  }

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;

    using flux_comm_types = FluxCommunicationTypes<Metavariables>;

    using normal_dot_numerical_flux_tag =
        db::add_tag_prefix<Tags::NormalDotNumericalFlux, variables_tag>;

    using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
    db::mutate<dt_variables_tag, mortar_data_tag>(
        make_not_null(&box),
        [&cache](
            const gsl::not_null<db::item_type<dt_variables_tag>*> dt_vars,
            const gsl::not_null<db::item_type<mortar_data_tag>*> mortar_data,
            const db::item_type<Tags::Mesh<volume_dim>>& mesh) noexcept {
          const auto& normal_dot_numerical_flux_computer =
              get<typename Metavariables::normal_dot_numerical_flux>(cache);

          for (auto& this_mortar_data : *mortar_data) {
            const auto& mortar_id = this_mortar_data.first;
            const auto& direction = mortar_id.first;
            const size_t dimension = direction.dimension();

            const auto data = this_mortar_data.second.extract();
            const auto& local_mortar_data = data.first;
            const auto& remote_mortar_data = data.second;

            // Compute numerical flux
            db::item_type<normal_dot_numerical_flux_tag>
                normal_dot_numerical_fluxes(
                    mesh.slice_away(dimension).number_of_grid_points(), 0.0);
            apply_normal_dot_numerical_flux(
                make_not_null(&normal_dot_numerical_fluxes),
                normal_dot_numerical_flux_computer, local_mortar_data,
                remote_mortar_data);

            // Projections need to happen here.

            db::item_type<dt_variables_tag> lifted_data(dg::lift_flux(
                local_mortar_data, std::move(normal_dot_numerical_fluxes),
                mesh.extents(dimension),
                get<typename flux_comm_types::MagnitudeOfFaceNormal>(
                    local_mortar_data)));

            add_slice_to_data(dt_vars, lifted_data, mesh.extents(), dimension,
                              index_to_slice_at(mesh.extents(), direction));
          }
        },
        db::get<Tags::Mesh<volume_dim>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
