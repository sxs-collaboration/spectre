// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "ApparentHorizons/ObjectLabel.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/ApparentHorizons/Measurements.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Expansion.hpp"
#include "ControlSystem/DataVectorHelpers.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
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
 *   control_system::ah::BothHorizons BothHorizons \endlink measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Expansion Expansion \endlink control error
 */
template <size_t DerivOrder>
struct Expansion : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() {
    return pretty_type::short_name<Expansion<DerivOrder>>();
  }

  // Expansion only has one component so just make it "Expansion"
  static std::string component_name(const size_t /*i*/) { return name(); }

  using measurement = ah::BothHorizons;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Expansion;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type =
        LinkedMessageQueue<double,
                           tmpl::list<QueueTags::Center<::ah::ObjectLabel::A>,
                                      QueueTags::Center<::ah::ObjectLabel::B>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags =
        tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Grid>>;

    template <::ah::ObjectLabel Horizon, typename Metavariables>
    static void apply(ah::BothHorizons::FindHorizon<Horizon> /*meta*/,
                      const Strahlkorper<Frame::Grid>& horizon_strahlkorper,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Expansion<DerivOrder>>>(cache);

      const DataVector center =
          array_to_datavector(horizon_strahlkorper.physical_center());

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Center<Horizon>, MeasurementQueue,
          UpdateControlSystem<Expansion>>>(control_sys_proxy, measurement_id,
                                           center);
    }
  };
};
}  // namespace control_system::Systems
