// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Skew.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Distorted;
}  // namespace Frame
/// \endcond

namespace control_system::Systems {
/*!
 * \brief Controls the 3D \link
 * domain::CoordinateMaps::TimeDependent::Skew Skew \endlink map
 *
 * \details Controls the map parameters $F_y(t)$ and $F_z(t)$ from the \link
 * domain::CoordinateMaps::TimeDependent::Skew Skew \endlink map.
 *
 * Requirements:
 * - This control system requires that there be exactly two objects in the
 *   simulation
 * - Currently both these objects must be black holes
 * - Currently this control system can only be used with the \link
 *   control_system::measurements::BothHorizons BothHorizons \endlink
 *   measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Skew Skew \endlink control error
 */
template <size_t DerivOrder, typename Measurement>
struct Skew : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() {
    return pretty_type::short_name<Skew<DerivOrder, Measurement>>();
  }

  // Skew has two components, Y and Z
  static std::optional<std::string> component_name(
      const size_t i, const size_t num_components) {
    ASSERT(num_components == 2,
           "Skew control expects 2 component but there are " << num_components
                                                             << " instead.");
    return i == 0 ? "Y" : "Z";
  }

  using measurement = Measurement;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Skew;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type = LinkedMessageQueue<
        double,
        tmpl::list<
            QueueTags::Horizon<Frame::Distorted, ::domain::ObjectLabel::A>,
            QueueTags::Horizon<Frame::Distorted, ::domain::ObjectLabel::B>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<ylm::Tags::Strahlkorper<Frame::Distorted>>;

    template <::domain::ObjectLabel Horizon, typename Metavariables>
    static void apply(
        measurements::BothHorizons::FindHorizon<Horizon> submeasurement,
        const ylm::Strahlkorper<Frame::Distorted>& horizon_strahlkorper,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Skew<DerivOrder, Measurement>>>(
          cache);

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          MeasurementQueue, UpdateControlSystem<Skew>,
          QueueTags::Horizon<Frame::Distorted, Horizon>>>(
          control_sys_proxy, measurement_id, horizon_strahlkorper);

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %.16f: Received measurement '%s'.\n",
                         name(), measurement_id.id,
                         pretty_type::name(submeasurement));
      }
    }
  };
};
}  // namespace control_system::Systems
