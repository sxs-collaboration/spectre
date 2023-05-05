// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Shape.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Measurements/SingleHorizon.hpp"
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
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Distorted;
}  // namespace Frame
/// \endcond

namespace control_system::Systems {
/*!
 * \brief Controls the \link domain::CoordinateMaps::TimeDependent::Shape Shape
 * \endlink map
 *
 * \details Controls the functions \f$ \lambda_{lm}(t) \f$ in the \link
 * domain::CoordinateMaps::TimeDependent::Shape Shape \endlink map to match the
 * shape of the excision sphere to the shape of the horizon.
 *
 * Requirements:
 * - This control system requires that there be at least one excision surface in
 *   the simulation
 * - Currently this control system can only be used with the \link
 *   control_system::measurements::BothHorizons BothHorizons \endlink
 * measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Shape Shape \endlink control error
 */
template <::domain::ObjectLabel Horizon, size_t DerivOrder,
          typename Measurement>
struct Shape : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() { return "Shape"s + ::domain::name(Horizon); }

  static std::optional<std::string> component_name(
      const size_t i, const size_t num_components) {
    // num_components = 2 * (l_max + 1)**2 if l_max == m_max which it is for the
    // shape map. This is why we can divide by 2 and take the sqrt without
    // worrying about odd numbers or non-perfect squares
    const size_t l_max = -1 + sqrt(num_components / 2);
    SpherepackIterator iter(l_max, l_max);
    const auto compact_index = iter.compact_index(i);
    if (compact_index.has_value()) {
      iter.set(compact_index.value());
      const int m =
          iter.coefficient_array() == SpherepackIterator::CoefficientArray::a
              ? static_cast<int>(iter.m())
              : -static_cast<int>(iter.m());
      return {"l"s + get_output(iter.l()) + "m"s + get_output(m)};
    } else {
      return std::nullopt;
    }
  }

  using measurement = Measurement;
  static_assert(
      std::is_same_v<measurement, measurements::SingleHorizon<Horizon>> or
          std::is_same_v<measurement, measurements::BothHorizons>,
      "Must use either SingleHorizon or BothHorizon measurement for Shape "
      "control system.");
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Shape<Horizon>;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type =
        LinkedMessageQueue<double,
                           tmpl::list<QueueTags::Horizon<Frame::Distorted>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags =
        tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Distorted>>;

    template <typename Metavariables>
    static void apply(typename measurements::SingleHorizon<
                          Horizon>::Submeasurement submeasurement,
                      const Strahlkorper<Frame::Distorted>& strahlkorper,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Shape>>(cache);

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::Horizon<Frame::Distorted>, MeasurementQueue,
          UpdateControlSystem<Shape>>>(control_sys_proxy, measurement_id,
                                       strahlkorper);

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %.16f: Received measurement '%s'.\n",
                         name(), measurement_id.id,
                         pretty_type::name(submeasurement));
      }
    }

    template <::domain::ObjectLabel MeasureHorizon, typename Metavariables>
    static void apply(
        measurements::BothHorizons::FindHorizon<MeasureHorizon> submeasurement,
        const Strahlkorper<Frame::Distorted>& strahlkorper,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      // The measurement event will call this for both horizons, but we only
      // need one of the horizons. So if it is called for the wrong horizon,
      // just do nothing.
      if constexpr (MeasureHorizon == Horizon) {
        auto& control_sys_proxy = Parallel::get_parallel_component<
            ControlComponent<Metavariables, Shape>>(cache);

        Parallel::simple_action<::Actions::UpdateMessageQueue<
            QueueTags::Horizon<Frame::Distorted>, MeasurementQueue,
            UpdateControlSystem<Shape>>>(control_sys_proxy, measurement_id,
                                         strahlkorper);

        if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
          Parallel::printf("%s, time = %.16f: Received measurement '%s'.\n",
                           name(), measurement_id.id,
                           pretty_type::name(submeasurement));
        }
      } else {
        (void)submeasurement;
        (void)strahlkorper;
        (void)cache;
        (void)measurement_id;
      }
    }
  };
};
}  // namespace control_system::Systems
