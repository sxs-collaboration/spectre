// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"

namespace CurvedScalarWave::Actions {

/// \ingroup ActionsGroup
/// \brief Action that initializes or updates items related to the
/// spacetime background of the CurvedScalarWave system
///
/// DataBox changes:
/// - Adds:
///   * `CurvedScalarWave::System::spacetime_tag_list`
/// - Removes: nothing
/// - Modifies: nothing

template <typename System>
struct CalculateGrVars {
  static constexpr size_t Dim = System::volume_dim;
  using simple_tags = db::AddSimpleTags<typename System::spacetime_tag_list>;
  using compute_tags = db::AddComputeTags<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto initial_data = evolution::Initialization::initial_data(
        db::get<CurvedScalarWave::Tags::BackgroundSpacetime<
            typename Metavariables::background_spacetime>>(box),
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box),
        db::get<::Tags::Time>(box), typename System::spacetime_tag_list{});
    tmpl::for_each<typename System::spacetime_tag_list>(
        [&box, &initial_data](auto spacetime_tag_v) {
          using spacetime_tag = tmpl::type_from<decltype(spacetime_tag_v)>;
          db::mutate<spacetime_tag>(
              [&initial_data](const auto spacetime_tag_ptr) {
                *spacetime_tag_ptr =
                    std::move(get<spacetime_tag>(initial_data));
              },
              make_not_null(&box));
        });

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Actions
