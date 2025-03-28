// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/GridCenters.hpp"
#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
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
 * \brief Tracks the centers of two neutron stars.
 *
 * Controls the function of time with the same name which can be used to
 * determine (until merger) the centers of the two stars.
 *
 * Requirements:
 * - This control system requires that there be two objects in the simulation
 * - Can only be used with the control_system::ControlErrors::GridCenters
 *   control error.
 */
template <size_t DerivOrder, typename Measurement>
struct GridCenters : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() { return "GridCenters"; }

  static std::optional<std::string> component_name(
      const size_t component, const size_t num_components) {
    ASSERT(num_components == 6,
           "GridCenters control expects 6 components but there are "
               << num_components << " instead.");

    const bool a_or_b = component < 3;
    const size_t index = component % 3;
    const std::string component_name =
        (a_or_b ? "A" : "B") +
        (index == 0 ? "_x"s : (index == 1 ? "_y" : "_z"));

    return {component_name};
  }

  using measurement = Measurement;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);
  static_assert(
      std::is_same_v<measurement, measurements::BothNSCenters>,
      "GridCenters only accepts BothNSCenters measurement currently.");

  using control_error = ControlErrors::GridCenters;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type = LinkedMessageQueue<
        double,
        tmpl::list<
            QueueTags::Center<::domain::ObjectLabel::A, Frame::Grid>,
            QueueTags::Center<::domain::ObjectLabel::B, Frame::Grid>,
            QueueTags::Center<::domain::ObjectLabel::A, Frame::Inertial>,
            QueueTags::Center<::domain::ObjectLabel::B, Frame::Inertial>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<
        measurements::Tags::NeutronStarCenter<::domain::ObjectLabel::A,
                                              Frame::Grid>,
        measurements::Tags::NeutronStarCenter<::domain::ObjectLabel::B,
                                              Frame::Grid>,
        measurements::Tags::NeutronStarCenter<::domain::ObjectLabel::A,
                                              Frame::Inertial>,
        measurements::Tags::NeutronStarCenter<::domain::ObjectLabel::B,
                                              Frame::Inertial>>;

    template <typename Metavariables>
    static void apply(
        measurements::BothNSCenters::FindTwoCenters submeasurement,
        const std::array<double, 3> grid_center_a,
        const std::array<double, 3> grid_center_b,
        const std::array<double, 3> inertial_center_a,
        const std::array<double, 3> inertial_center_b,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, GridCenters>>(cache);

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          MeasurementQueue, UpdateControlSystem<GridCenters>,
          QueueTags::Center<::domain::ObjectLabel::A, Frame::Grid>,
          QueueTags::Center<::domain::ObjectLabel::B, Frame::Grid>,
          QueueTags::Center<::domain::ObjectLabel::A, Frame::Inertial>,
          QueueTags::Center<::domain::ObjectLabel::B, Frame::Inertial>>>(
          control_sys_proxy, measurement_id, DataVector(grid_center_a),
          DataVector(grid_center_b), DataVector(inertial_center_a),
          DataVector(inertial_center_b));

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %.16f: Received measurement '%s'.\n",
                         name(), measurement_id.id,
                         pretty_type::name(submeasurement));
      }
    }
  };
};
}  // namespace control_system::Systems
