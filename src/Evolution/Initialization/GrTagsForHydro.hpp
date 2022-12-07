// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>  // IWYU pragma: keep  // for move

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
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
template <typename System>
struct GrTagsForHydro {
  using simple_tags_from_options = tmpl::list<::Tags::Time>;

  static constexpr size_t dim = System::volume_dim;
  using gr_tag = typename System::spacetime_variables_tag;

  using simple_tags = tmpl::list<gr_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const double initial_time = db::get<::Tags::Time>(box);
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

    constexpr bool initial_data_in_box =
        db::tag_is_retrievable_v<evolution::initial_data::Tags::InitialData,
                                 db::DataBox<DbTagsList>>;
    constexpr bool analytic_soln_or_data_in_box =
        db::tag_is_retrievable_v<::Tags::AnalyticSolutionOrData,
                                 db::DataBox<DbTagsList>>;

    static_assert(initial_data_in_box or analytic_soln_or_data_in_box,
                  "Either ::Tags::AnalyticSolutionOrData or "
                  "evolution::initial_data::Tags::InitialData must be in the"
                  "DataBox.");

    // Set initial data from analytic solution
    GrVars gr_vars{num_grid_points};

    if constexpr (initial_data_in_box) {
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      call_with_dynamic_type<void, derived_classes>(
          &db::get<evolution::initial_data::Tags::InitialData>(box),
          [&gr_vars, &initial_time,
           &inertial_coords](const auto* const data_or_solution) {
            gr_vars.assign_subset(evolution::Initialization::initial_data(
                *data_or_solution, inertial_coords, initial_time,
                typename GrVars::tags_list{}));
          });
    } else if constexpr (analytic_soln_or_data_in_box) {
      gr_vars.assign_subset(evolution::Initialization::initial_data(
          db::get<::Tags::AnalyticSolutionOrData>(box), inertial_coords,
          initial_time, typename GrVars::tags_list{}));
    }

    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(gr_vars));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Initialization
