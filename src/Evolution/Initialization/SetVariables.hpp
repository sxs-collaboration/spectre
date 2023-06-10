// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
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
  using simple_tags_from_options = tmpl::list<::Tags::Time>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if constexpr (db::tag_is_retrievable_v<
                      evolution::initial_data::Tags::InitialData,
                      db::DataBox<DbTagsList>>) {
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      call_with_dynamic_type<void, derived_classes>(
          &db::get<evolution::initial_data::Tags::InitialData>(box),
          [&box](const auto* const data_or_solution) {
            using initial_data_subclass =
                std::decay_t<decltype(*data_or_solution)>;
            if constexpr (is_analytic_data_v<initial_data_subclass> or
                          is_analytic_solution_v<initial_data_subclass>) {
              impl<Metavariables>(make_not_null(&box), *data_or_solution);
            } else {
              ERROR(
                  "Trying to use "
                  "'evolution::Initialization::Actions::SetVariables' with a "
                  "class that's not marked as analytic solution or analytic "
                  "data. To support numeric initial data, add a "
                  "system-specific initialization routine to your executable.");
            }
          });
    } else if constexpr (db::tag_is_retrievable_v<
                             ::Tags::AnalyticSolutionOrData,
                             db::DataBox<DbTagsList>>) {
      impl<Metavariables>(make_not_null(&box),
                          db::get<::Tags::AnalyticSolutionOrData>(box));
    } else {
      ERROR(
          "Either ::Tags::AnalyticSolutionOrData or "
          "evolution::initial_data::Tags::InitialData must be in the "
          "DataBox.");
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 private:
  template <typename Metavariables, typename DbTagsList, typename T>
  static void impl(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                   const T& solution_or_data) {
    const double initial_time = db::get<::Tags::Time>(*box);
    const auto inertial_coords =
        db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
            Metavariables::volume_dim, Frame::Grid, Frame::Inertial>>(*box)(
            db::get<::domain::Tags::ElementMap<Metavariables::volume_dim,
                                               Frame::Grid>>(*box)(
                db::get<LogicalCoordinatesTag>(*box)),
            initial_time, db::get<::domain::Tags::FunctionsOfTime>(*box));

    using system = typename Metavariables::system;

    if constexpr (Metavariables::system::has_primitive_and_conservative_vars) {
      using primitives_tag = typename system::primitive_variables_tag;
      // Set initial data from analytic solution
      db::mutate<primitives_tag>(
          [&initial_time, &inertial_coords, &solution_or_data](
              const gsl::not_null<typename primitives_tag::type*>
                  primitive_vars) {
            primitive_vars->assign_subset(
                evolution::Initialization::initial_data(
                    solution_or_data, inertial_coords, initial_time,
                    typename Metavariables::analytic_variables_tags{}));
          },
          box);
      using non_conservative_variables =
          typename system::non_conservative_variables;
      using variables_tag = typename system::variables_tag;
      if constexpr (not std::is_same_v<non_conservative_variables,
                                       tmpl::list<>>) {
        db::mutate<variables_tag>(
            [&initial_time, &inertial_coords, &solution_or_data](
                const gsl::not_null<typename variables_tag::type*>
                    evolved_vars) {
              evolved_vars->assign_subset(
                  evolution::Initialization::initial_data(
                      solution_or_data, inertial_coords, initial_time,
                      non_conservative_variables{}));
            },
            box);
      }
    } else {
      using variables_tag = typename system::variables_tag;

      // Set initial data from analytic solution
      using Vars = typename variables_tag::type;
      db::mutate<variables_tag>(
          [&initial_time, &inertial_coords,
           &solution_or_data](const gsl::not_null<Vars*> vars) {
            vars->assign_subset(evolution::Initialization::initial_data(
                solution_or_data, inertial_coords, initial_time,
                typename Vars::tags_list{}));
          },
          box);
    }
  }
};
}  // namespace Actions
}  // namespace Initialization
}  // namespace evolution
