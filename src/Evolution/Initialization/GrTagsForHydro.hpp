// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep  // for move

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Grid;
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
template <size_t Dim, typename Frame>
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
/// \brief Allocate and set general relativity quantities needed for evolution
/// of some hydro systems
///
/// Uses:
/// - DataBox:
///   * `Tags::Mesh<Dim>`
///
/// DataBox changes:
/// - Adds:
///   * system::spacetime_variables_tag
///
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename System>
struct GrTagsForHydro {
  using initialization_tags = tmpl::list<Initialization::Tags::InitialTime>;

  static constexpr size_t dim = System::volume_dim;
  using gr_tag = typename System::spacetime_variables_tag;

  using simple_tags = tmpl::list<gr_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    using GrVars = typename gr_tag::type;

    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    const auto inertial_coords =
        db::get<domain::CoordinateMaps::Tags::CoordinateMap<dim, Frame::Grid,
                                                            Frame::Inertial>>(
            box)(
            db::get<domain::Tags::ElementMap<dim, Frame::Grid>>(box)(
                db::get<domain::Tags::Coordinates<dim, Frame::ElementLogical>>(
                    box)),
            initial_time, db::get<domain::Tags::FunctionsOfTime>(box));

    // Set initial data from analytic solution
    GrVars gr_vars{num_grid_points};
    gr_vars.assign_subset(evolution::initial_data(
        Parallel::get<::Tags::AnalyticSolutionOrData>(cache), inertial_coords,
        initial_time, typename GrVars::tags_list{}));
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(gr_vars));

    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Initialization
