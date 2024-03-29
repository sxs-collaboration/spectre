// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace CurvedScalarWave::Actions {

/// \ingroup ActionsGroup
/// \brief Action that initializes or updates items related to the
/// spacetime background of the CurvedScalarWave system
///
/// If `SkipForStaticBlocks` is `true`, then this action does nothing if the
/// block of the domain is time independent. This is a performance optimization
/// to avoid updating the background spacetime in blocks that are time
/// independent. Note that this assumes that the background spacetime is also
/// time independent.
///
/// DataBox changes:
/// - Adds:
///   * `CurvedScalarWave::System::spacetime_tag_list`
/// - Removes: nothing
/// - Modifies: nothing
template <typename System, bool SkipForStaticDomains>
struct CalculateGrVars {
  static constexpr size_t Dim = System::volume_dim;
  using simple_tags = db::AddSimpleTags<typename System::spacetime_tag_list>;
  using compute_tags = db::AddComputeTags<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if constexpr (SkipForStaticDomains) {
      const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
      const auto& block = domain.blocks()[element_id.block_id()];
      if (not block.is_time_dependent()) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }
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
