// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution {
namespace Initialization {
namespace Actions {
template <typename LogicalCoordinatesTag>
struct SetInitialData {
  using initialization_tags = tmpl::list<::Initialization::Tags::InitialTime>;

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
                      typename db::DataBox<DbTagsList>::mutable_item_tags,
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

    // retrieve initial_data type object from box
    const auto& initial_data =
        db::get<InitialDataUtilities::Tags::InitialDataBase>(*box);
    using initial_data_type = std::decay_t<decltype(initial_data)>;

    if constexpr (Metavariables::system::has_primitive_and_conservative_vars) {
      // Aug24 2021 Yoonsoo : no changes made yet for systems with primitive and
      // conservative vars
      using primitives_tag = typename system::primitive_variables_tag;

      // Set initial data from analytic solution
      db::mutate<primitives_tag>(
          box, [&initial_time, &inertial_coords, &initial_data](
                   const gsl::not_null<typename primitives_tag::type*>
                       primitive_vars) noexcept {
            primitive_vars->assign_subset(evolution::initial_data(
                initial_data, inertial_coords, initial_time,
                typename Metavariables::analytic_variables_tags{}));
          });
      using non_conservative_variables =
          typename system::non_conservative_variables;
      using variables_tag = typename system::variables_tag;
      if constexpr (not std::is_same_v<non_conservative_variables,
                                       tmpl::list<>>) {
        db::mutate<variables_tag>(
            box, [&initial_time, &inertial_coords, &initial_data](
                     const gsl::not_null<typename variables_tag::type*>
                         evolved_vars) noexcept {
              evolved_vars->assign_subset(evolution::initial_data(
                  initial_data, inertial_coords, initial_time,
                  non_conservative_variables{}));
            });
      }
    } else {
      using variables_tag = typename system::variables_tag;

      // Set initial data from analytic solution
      using Vars = typename variables_tag::type;
      db::mutate<variables_tag>(
          box, [&initial_time, &inertial_coords,
                &initial_data](const gsl::not_null<Vars*> vars) noexcept {
            using factory_classes =
                typename Metavariables::factory_creation::factory_classes;

            vars->assign_subset(evolution::initial_data<
                                tmpl::at<factory_classes, initial_data_type>>(
                initial_data, std::move(inertial_coords), initial_time,
                typename Vars::tags_list{}));
          });
    }
  }
};
}  // namespace Actions
}  // namespace Initialization
}  // namespace evolution
