// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

#include "ApparentHorizons/StrahlkorperInDifferentFrame.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Size.hpp"
#include "ControlSystem/Measurements/CharSpeed.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
}  // namespace Frame
namespace control_system {
template <typename ControlSystem>
struct UpdateControlSystem;
}
/// \endcond

namespace control_system::Systems {
/*!
 * \brief Controls the \f$l=0\f$ component of the \link
 * domain::CoordinateMaps::TimeDependent::Shape Shape \endlink map
 *
 * Requirements:
 * - This control system requires that there be at least one object in the
 *   simulation
 * - This object must be a black hole (have an excision)
 * - Currently this control system can only be used with the \link
 *   control_system::measurements::CharSpeed CharSpeed \endlink
 * measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Size Size \endlink control error
 */
template <::domain::ObjectLabel Horizon, size_t DerivOrder>
struct Size : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() { return "Size"s + ::domain::name(Horizon); }

  static std::optional<std::string> component_name(
      const size_t /*i*/, const size_t /*num_components*/) {
    return "Size";
  }

  using measurement = control_system::measurements::CharSpeed<Horizon>;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Size<deriv_order, Horizon>;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type = LinkedMessageQueue<
        double, tmpl::list<QueueTags::SizeExcisionQuantities<Frame::Distorted>,
                           QueueTags::SizeHorizonQuantities<Frame::Distorted>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::conditional_t<
        std::is_same_v<Submeasurement,
                       typename measurements::CharSpeed<Horizon>::Excision>,
        tmpl::list<
            StrahlkorperTags::Strahlkorper<Frame::Grid>,
            gr::Tags::Lapse<DataVector>,
            gr::Tags::ShiftyQuantity<DataVector, 3, Frame::Distorted>,
            gr::Tags::SpatialMetric<DataVector, 3, Frame::Distorted>,
            gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Distorted>>,
        tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Distorted>,
                   ::ah::Tags::TimeDerivStrahlkorper<Frame::Distorted>>>;

    template <typename Metavariables>
    static void apply(
        typename measurements::CharSpeed<Horizon>::Excision /*meta*/,
        const Strahlkorper<Frame::Grid>& grid_excision_surface,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, 3, Frame::Distorted>& shifty_quantity,
        const tnsr::ii<DataVector, 3, Frame::Distorted>&
            spatial_metric_on_excision_surface,
        const tnsr::II<DataVector, 3, Frame::Distorted>&
            inverse_spatial_metric_on_excision_surface,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Size>>(cache);

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %s: Received excision measurement\n",
                         name(), measurement_id);
      }

      Strahlkorper<Frame::Distorted> distorted_excision_surface{};
      strahlkorper_in_different_frame_aligned(
          make_not_null(&distorted_excision_surface), grid_excision_surface,
          Parallel::get<domain::Tags::Domain<3>>(cache),
          Parallel::get<domain::Tags::FunctionsOfTime>(cache),
          measurement_id.id);

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::SizeExcisionQuantities<Frame::Distorted>, MeasurementQueue,
          UpdateControlSystem<Size>>>(
          control_sys_proxy, measurement_id,
          QueueTags::SizeExcisionQuantities<Frame::Distorted>::type{
              std::move(distorted_excision_surface), lapse, shifty_quantity,
              spatial_metric_on_excision_surface,
              inverse_spatial_metric_on_excision_surface});
    }

    template <typename Metavariables>
    static void apply(
        typename measurements::CharSpeed<Horizon>::Horizon /*meta*/,
        const Strahlkorper<Frame::Distorted>& horizon,
        const Strahlkorper<Frame::Distorted>& time_deriv_horizon,
        Parallel::GlobalCache<Metavariables>& cache,
        const LinkedMessageId<double>& measurement_id) {
      auto& control_sys_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, Size>>(cache);

      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf("%s, time = %s: Received horizon measurement\n",
                         name(), measurement_id);
      }

      Parallel::simple_action<::Actions::UpdateMessageQueue<
          QueueTags::SizeHorizonQuantities<Frame::Distorted>, MeasurementQueue,
          UpdateControlSystem<Size>>>(
          control_sys_proxy, measurement_id,
          QueueTags::SizeHorizonQuantities<Frame::Distorted>::type{
              horizon, time_deriv_horizon});
    }
  };
};
}  // namespace control_system::Systems
