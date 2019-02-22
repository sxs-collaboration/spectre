// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep // for db::item_type<Tags::Mesh<...>>
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"  // IWYU pragma: keep // for db::item_type<Tags::Mortars<...>>
#include "Time/Tags.hpp"  // IWYU pragma: keep // for db::item_type<Tags::TimeStep>
#include "Time/TimeSteppers/TimeStepper.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "Time/Time.hpp" // for TimeDelta

/// \cond
// IWYU pragma: no_forward_declare TimeDelta
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Perform the boundary part of the update of the variables
/// for local time stepping.
///
/// Uses:
/// - ConstGlobalCache:
///   - OptionTags::TimeStepper
///   - Metavariables::normal_dot_numerical_flux
/// - DataBox:
///   - Tags::Mesh<volume_dim>
///   - Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>
///   - Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>
///   - Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - variables_tag
///   - FluxCommunicationTypes<Metavariables>::mortar_data_tag
struct ApplyBoundaryFluxesLocalTimeStepping {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    static_assert(Metavariables::local_time_stepping,
                  "ApplyBoundaryFluxesLocalTimeStepping can only be used with "
                  "local time-stepping.");

    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;

    using flux_comm_types = FluxCommunicationTypes<Metavariables>;

    using mortar_data_tag =
        typename flux_comm_types::local_time_stepping_mortar_data_tag;
    db::mutate<variables_tag, mortar_data_tag>(
        make_not_null(&box),
        [&cache](
            const gsl::not_null<db::item_type<variables_tag>*> vars,
            const gsl::not_null<db::item_type<mortar_data_tag>*> mortar_data,
            const db::item_type<Tags::Mesh<volume_dim>>& mesh,
            const db::item_type<Tags::Mortars<Tags::Mesh<volume_dim - 1>,
                                              volume_dim>>& mortar_meshes,
            const db::item_type<Tags::Mortars<Tags::MortarSize<volume_dim - 1>,
                                              volume_dim>>& mortar_sizes,
            const db::item_type<Tags::TimeStep>& time_step) noexcept {
          // Having the lambda just wrap another lambda works around a
          // gcc 6.4.0 segfault.
          [
            &cache, &vars, &mortar_data, &mesh, &mortar_meshes, &mortar_sizes,
            &time_step
          ]() noexcept {
            const auto& normal_dot_numerical_flux_computer =
                get<typename Metavariables::normal_dot_numerical_flux>(cache);
            const LtsTimeStepper& time_stepper =
                get<OptionTags::TimeStepper>(cache);

            for (auto& mortar_id_and_data : *mortar_data) {
              const auto& mortar_id = mortar_id_and_data.first;
              auto& data = mortar_id_and_data.second;
              const auto& direction = mortar_id.first;
              const size_t dimension = direction.dimension();

              const auto face_mesh = mesh.slice_away(dimension);
              const auto perpendicular_extent = mesh.extents(dimension);
              const auto& mortar_mesh = mortar_meshes.at(mortar_id);
              const auto& mortar_size = mortar_sizes.at(mortar_id);

              // This lambda must only capture quantities that are
              // independent of the simulation state.
              const auto coupling =
                  [
                    &face_mesh, &mortar_mesh, &mortar_size,
                    &normal_dot_numerical_flux_computer, &perpendicular_extent
                  ](const typename flux_comm_types::LocalData& local_data,
                    const typename flux_comm_types::PackagedData&
                        remote_data) noexcept {
                return compute_boundary_flux_contribution<flux_comm_types>(
                    normal_dot_numerical_flux_computer, local_data, remote_data,
                    face_mesh, mortar_mesh, perpendicular_extent, mortar_size);
              };

              const auto lifted_data = time_stepper.compute_boundary_delta(
                  coupling, make_not_null(&data), time_step);

              add_slice_to_data(vars, lifted_data, mesh.extents(), dimension,
                                index_to_slice_at(mesh.extents(), direction));
            }
          }();
        },
        db::get<Tags::Mesh<volume_dim>>(box),
        db::get<Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>>(box),
        db::get<Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>>(
            box),
        db::get<Tags::TimeStep>(box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
