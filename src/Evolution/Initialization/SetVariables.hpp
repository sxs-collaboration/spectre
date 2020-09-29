// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace evolution {
namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Sets variables needed for evolution of hyperbolic systems
///
/// Uses:
/// - DataBox:
///   * `CoordinatesTag`
/// - GlobalCache:
///   * `OptionTags::AnalyticSolutionBase` or `OptionTags::AnalyticDataBase`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   * System::variables_tag (if system has no primitive variables)
///   * System::primitive_variables_tag (if system has primitive variables)
template <typename LogicalCoordinatesTag>
struct SetVariables {
  using initialization_option_tags =
      tmpl::list<::Initialization::Tags::InitialTime>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (tmpl::list_contains_v<
                      typename db::DataBox<DbTagsList>::simple_item_tags,
                      ::Initialization::Tags::InitialTime>) {
      impl<Metavariables>(make_not_null(&box));
    } else {
      ERROR(
          "Could not find dependency 'Initialization::Tags::InitialTime' in "
          "DataBox.");
    }
    return {std::move(box)};
  }

 private:
  template <typename Metavariables, typename DbTagsList>
  static void impl(const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
    const double initial_time =
        db::get<::Initialization::Tags::InitialTime>(*box);
    const auto inertial_coords =
        db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
            Metavariables::volume_dim, Frame::Grid, Frame::Inertial>>(*box)(
            db::get<::domain::Tags::ElementMap<Metavariables::volume_dim,
                                               Frame::Grid>>(*box)(
                db::get<LogicalCoordinatesTag>(*box)),
            initial_time, db::get<::domain::Tags::FunctionsOfTime>(*box));

    using system = typename Metavariables::system;

    const auto& solution_or_data =
        db::get<::Tags::AnalyticSolutionOrData>(*box);

    if constexpr (Metavariables::system::has_primitive_and_conservative_vars) {
      using primitives_tag = typename system::primitive_variables_tag;
      // Set initial data from analytic solution
      db::mutate<primitives_tag>(
          box, [&initial_time, &inertial_coords, &solution_or_data ](
                   const gsl::not_null<typename primitives_tag::type*>
                       primitive_vars) noexcept {
            primitive_vars->assign_subset(evolution::initial_data(
                solution_or_data, inertial_coords, initial_time,
                typename Metavariables::analytic_variables_tags{}));
          });
    } else {
      using variables_tag = typename system::variables_tag;

      // Set initial data from analytic solution
      using Vars = typename variables_tag::type;
      db::mutate<variables_tag>(
          box, [&initial_time, &inertial_coords,
                &solution_or_data](const gsl::not_null<Vars*> vars) noexcept {
            vars->assign_subset(evolution::initial_data(
                solution_or_data, inertial_coords, initial_time,
                typename Vars::tags_list{}));
          });
    }
  }
};
}  // namespace Actions
}  // namespace Initialization
}  // namespace evolution
