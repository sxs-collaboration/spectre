// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace control_system {
/// \ingroup ControlSystemGroup
/// Apply the `process_measurement` struct of each of the \p
/// ControlSystems to the result of the \p Submeasurement.
///
/// The submeasurement results are supplied as a `db::DataBox` in
/// order to allow individual control systems to select the portions
/// of the submeasurement that they are interested in.
///
/// In addition to being manually called, this struct is designed to
/// be usable as a `post_horizon_find_callback` or a
/// `post_interpolation_callback`.
template <typename Submeasurement, typename ControlSystems>
struct RunCallbacks
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
 private:
  static_assert(
      tt::assert_conforms_to_v<Submeasurement, protocols::Submeasurement>);
  static_assert(
      tmpl::all<
          ControlSystems,
          tt::assert_conforms_to<tmpl::_1, protocols::ControlSystem>>::value);

 public:
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& measurement_id) {
    static_assert(
        std::is_same_v<TemporalId, LinkedMessageId<double>>,
        "RunCallbacks expects a LinkedMessageId<double> as its temporal id");
    tmpl::for_each<ControlSystems>(
        [&box, &cache, &measurement_id](auto control_system_v) {
          using ControlSystem = tmpl::type_from<decltype(control_system_v)>;
          db::apply<typename ControlSystem::process_measurement::
                        template argument_tags<Submeasurement>>(
              [&cache, &measurement_id](const auto&... args) {
                ControlSystem::process_measurement::apply(
                    Submeasurement{}, args..., cache, measurement_id);
              },
              box);
        });
  }
};
}  // namespace control_system
