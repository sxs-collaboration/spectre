// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"

/// \cond
namespace logging::Tags {
template <typename OptionsGroup>
struct Verbosity;
}  // namespace logging::Tags
/// \endcond

namespace intrp::callbacks {
/*!
 * \brief Post horizon find callback that will send a message to the
 * ObserverWriter if we have a dependency telling it to write the volume data to
 * disk or not depending on the \p WriteVolumeData template bool.
 *
 * \details Can be used whether the horizon finder fails or not. The
 * `failure_reason` is not used when the horizon find fails.
 */
template <bool WriteVolumeData, typename InterpolationTargetTag>
struct SendDependencyToObserverWriter {
  // Callback for when horizon find succeeds
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    static_assert(WriteVolumeData);
    apply_impl<InterpolationTargetTag>(box, cache, temporal_id);
  }

  // Callback for when horizon find fails
  template <typename LocalInterpolationTargetTag, typename DbTags,
            typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id,
                    const FastFlow::Status /*failure_reason*/) {
    static_assert(not WriteVolumeData);
    apply_impl<LocalInterpolationTargetTag>(box, cache, temporal_id);
  }

 private:
  template <typename LocalInterpolationTargetTag, typename DbTags,
            typename Metavariables, typename TemporalId>
  static void apply_impl(const db::DataBox<DbTags>& box,
                         Parallel::GlobalCache<Metavariables>& cache,
                         const TemporalId& temporal_id) {
    static_assert(
        std::is_same_v<InterpolationTargetTag, LocalInterpolationTargetTag>);

    const auto& dependencies =
        db::get<intrp::Tags::Dependencies<TemporalId>>(box);

    ASSERT(dependencies.contains(temporal_id),
           "TemporalId " << temporal_id
                         << " not found in dependencies: " << dependencies);

    const auto& dependency = dependencies.at(temporal_id);

    if (dependency.has_value()) {
      const auto& verbosity =
          db::get<logging::Tags::Verbosity<InterpolationTargetTag>>(box);

      auto& observer_writer_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      Parallel::threaded_action<
          observers::ThreadedActions::ContributeDependency>(
          observer_writer_proxy,
          InterpolationTarget_detail::get_temporal_id_value(temporal_id),
          pretty_type::name<InterpolationTargetTag>(), dependency.value(),
          WriteVolumeData);

      if (verbosity >= ::Verbosity::Verbose) {
        Parallel::printf(
            "Remark: We are%s writing volume data for horizon finder %s",
            WriteVolumeData ? "" : " not",
            pretty_type::name<InterpolationTargetTag>());
      }
    }
  }
};
}  // namespace intrp::callbacks

namespace ah::callbacks {
/*!
 * \brief A `ah::protocols::Callback` that will send a message to the
 * ObserverWriter if we have a dependency using the
 * `observers::ThreadedActions::ContributeDependency` action.
 *
 * \details If we have a dependency, the template bool \p WriteVolumeData
 * determines if the message sent to the ObserverWriter says to write the volume
 * data to disk or not.
 */
template <typename HorizonMetavars, bool WriteVolumeData>
struct SendDependencyToObserverWriter
    : tt::ConformsTo<ah::protocols::Callback> {
  // Callback for when horizon find succeeds
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status /*status*/) {
    const auto& time = db::get<ah::Tags::CurrentTime>(box).value();
    const auto& dependency = db::get<ah::Tags::Dependency>(box);

    if (dependency.has_value()) {
      const auto& verbosity = db::get<ah::Tags::Verbosity>(box);

      auto& observer_writer_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      Parallel::threaded_action<
          observers::ThreadedActions::ContributeDependency>(
          observer_writer_proxy, time.id, pretty_type::name<HorizonMetavars>(),
          dependency.value(), WriteVolumeData);

      if (verbosity >= ::Verbosity::Verbose) {
        Parallel::printf(
            "Remark: We are%s writing volume data for horizon finder %s\n",
            WriteVolumeData ? "" : " not",
            pretty_type::name<HorizonMetavars>());
      }
    }
  }
};
}  // namespace ah::callbacks
