// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Expansion.hpp"
#include "ControlSystem/DataVectorHelpers.hpp"
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
 * domain::CoordinateMaps::TimeDependent::CubicScale CubicScale \endlink map
 *
 * \details Controls the function \f$a(t)\f$ (a FunctionOfTime) in the
 * \link domain::CoordinateMaps::TimeDependent::CubicScale
 * CubicScale \endlink coordinate map, while the function \f$b(t)\f$ is
 * typically an analytic function (usually a FixedSpeedCubic). See \link
 * domain::CoordinateMaps::TimeDependent::CubicScale CubicScale \endlink for
 * definitions of both \f$a(t)\f$ and \f$b(t)\f$.
 *
 * Requirements:
 * - This control system requires that there be exactly two objects in the
 *   simulation
 * - Currently both these objects must be black holes
 * - Currently this control system can only be used with the \link
 *   control_system::measurements::BothHorizons BothHorizons \endlink
 * measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Expansion Expansion \endlink control error
 */
template <size_t DerivOrder, typename Measurement>
struct Expansion : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() {
    return pretty_type::short_name<Expansion<DerivOrder, Measurement>>();
  }

  // Expansion only has one component so just make it "Expansion"
  static std::optional<std::string> component_name(
      const size_t /*i*/, const size_t num_components) {
    ASSERT(num_components == 1,
           "Expansion control expects 1 component but there are "
               << num_components << " instead.");
    return name();
  }

  using measurement = Measurement;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Expansion;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type = LinkedMessageQueue<
        double, tmpl::list<QueueTags::Center<::domain::ObjectLabel::A>,
                           QueueTags::Center<::domain::ObjectLabel::B>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::conditional_t<
        std::is_same_v<Submeasurement,
                       measurements::BothNSCenters::FindTwoCenters>,
        tmpl::list<
            measurements::Tags::NeutronStarCenter<::domain::ObjectLabel::A>,
            measurements::Tags::NeutronStarCenter<::domain::ObjectLabel::B>>,
        tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Distorted>>>;

    template <::domain::ObjectLabel Horizon, typename Metavariables>
    static void apply(
        measurements::BothHorizons::FindHorizon<Horizon> submeasurement,
        const Strahlkorper<Frame::Distorted>& horizon_strahlkorper,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Expansion<DerivOrder, Measurement>>>(
          cache);

      const DataVector center =
          array_to_datavector(horizon_strahlkorper.physical_center());

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<Horizon>, MeasurementQueue,
          UpdateControlSystem<Expansion>>>(control_sys_proxy, measurement_id,
                                           center);

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %.16f: Received measurement '%s'.\n",
                         name(), measurement_id.id,
                         pretty_type::name(submeasurement));
      }
    }

    template <typename Metavariables>
    static void apply(
        measurements::BothNSCenters::FindTwoCenters submeasurement,
        const std::array<double, 3> center_a,
        const std::array<double, 3> center_b,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Expansion<DerivOrder, Measurement>>>(
          cache);

      const DataVector center_a_dv = array_to_datavector(center_a);
      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<::domain::ObjectLabel::A>, MeasurementQueue,
          UpdateControlSystem<Expansion>>>(control_sys_proxy, measurement_id,
                                           center_a_dv);
      const DataVector center_b_dv = array_to_datavector(center_b);
      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<::domain::ObjectLabel::B>, MeasurementQueue,
          UpdateControlSystem<Expansion>>>(control_sys_proxy, measurement_id,
                                           center_b_dv);

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %.16f: Received measurement '%s'.\n",
                         name(), measurement_id.id,
                         pretty_type::name(submeasurement));
      }
    }
  };
};
}  // namespace control_system::Systems
