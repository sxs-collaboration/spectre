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
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"

/// \cond
namespace logging::Tags {
template <typename OptionsGroup>
struct Verbosity;
}  // namespace logging::Tags
/// \endcond

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
