// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Translation.hpp"
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
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
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
 * domain::CoordinateMaps::TimeDependent::Translation Translation \endlink map
 *
 * \details Controls the function \f$ \vec{T}(t) \f$ in the \link
 * domain::CoordinateMaps::TimeDependent::Translation Translation \endlink map.
 *
 * Requirements:
 * - This control system requires that there be exactly two objects in the
 *   simulation
 * - Currently both these objects must be black holes
 * - Currently this control system can only be used with the \link
 *   control_system::measurements::BothHorizons BothHorizons \endlink
 * measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Translation Translation \endlink control
 *   error
 */
template <size_t DerivOrder, typename Measurement>
struct Translation : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() {
    return pretty_type::short_name<Translation<DerivOrder, Measurement>>();
  }

  static std::optional<std::string> component_name(
      const size_t component, const size_t num_components) {
    ASSERT(num_components == 3,
           "Translation control expects 3 components but there are "
               << num_components << " instead.");
    return component == 0 ? "x" : component == 1 ? "y" : "z";
  }

  using measurement = Measurement;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Translation;
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
        const Strahlkorper<Frame::Distorted>& strahlkorper,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy =
          Parallel::get_parallel_component<ControlComponent<
              Metavariables, Translation<DerivOrder, Measurement>>>(cache);

      const DataVector center =
          array_to_datavector(strahlkorper.physical_center());

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<Horizon>, MeasurementQueue,
          UpdateControlSystem<Translation>>>(control_sys_proxy, measurement_id,
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
      auto& control_sys_proxy =
          Parallel::get_parallel_component<ControlComponent<
              Metavariables, Translation<DerivOrder, Measurement>>>(cache);

      const DataVector center_a_dv = array_to_datavector(center_a);
      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<::domain::ObjectLabel::A>, MeasurementQueue,
          UpdateControlSystem<Translation>>>(control_sys_proxy, measurement_id,
                                             center_a_dv);
      const DataVector center_b_dv = array_to_datavector(center_b);
      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<::domain::ObjectLabel::B>, MeasurementQueue,
          UpdateControlSystem<Translation>>>(control_sys_proxy, measurement_id,
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
