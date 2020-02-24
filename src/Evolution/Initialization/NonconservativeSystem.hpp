// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep  // for move

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Initialization {
namespace Tags {
struct InitialTime;
}  // namespace Tags
}  // namespace Initialization
namespace Tags {
struct AnalyticSolutionOrData;
}  // namespace Tags
namespace domain {
namespace Tags {
template <size_t VolumeDim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate and set variables needed for evolution of nonconservative
/// systems
///
/// Uses:
/// - DataBox:
///   * `Tags::Mesh<Dim>`
///
/// DataBox changes:
/// - Adds:
///   * System::variables_tag
///
/// - Removes: nothing
/// - Modifies: nothing
struct NonconservativeSystem {
  using initialization_tags = tmpl::list<Initialization::Tags::InitialTime>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    static_assert(not system::is_in_flux_conservative_form,
                  "System is in flux conservative form");
    static constexpr size_t dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using simple_tags = db::AddSimpleTags<variables_tag>;
    using compute_tags = db::AddComputeTags<>;
    using Vars = typename variables_tag::type;

    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    const auto& inertial_coords =
        db::get<domain::Tags::Coordinates<dim, Frame::Inertial>>(box);

    // Set initial data from analytic solution
    Vars vars{num_grid_points};
    vars.assign_subset(evolution::initial_data(
        Parallel::get<::Tags::AnalyticSolutionOrData>(cache), inertial_coords,
        initial_time, typename Vars::tags_list{}));

    return std::make_tuple(
        merge_into_databox<NonconservativeSystem, simple_tags, compute_tags>(
            std::move(box), std::move(vars)));
  }
};
}  // namespace Actions
}  // namespace Initialization
