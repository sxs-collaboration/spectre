// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class ElementId;
template <size_t Dim>
class Mesh;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain::Tags {
template <size_t Dim>
struct Mesh;
}  // namespace domain::Tags
/// \endcond

namespace control_system::measurements {
/*!
 * \brief A `control_system::protocols::Measurement` that relies on only one
 * apparent horizon; the template parameter `Horizon`.
 */
template <::domain::ObjectLabel Horizon>
struct SingleHorizon : tt::ConformsTo<protocols::Measurement> {
  /*!
   * \brief A `control_system::protocols::Submeasurement` that starts the
   * interpolation to the interpolation target in order to find the apparent
   * horizon.
   */
  struct Submeasurement : tt::ConformsTo<protocols::Submeasurement> {
   private:
    template <typename ControlSystems>
    struct InterpolationTarget
        : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
      static std::string name() {
        return "ControlSystemSingleAh" + ::domain::name(Horizon);
      }

      using temporal_id = ::Tags::TimeAndPrevious;

      using vars_to_interpolate_to_target =
          ::ah::vars_to_interpolate_to_target<3, ::Frame::Grid>;
      using compute_vars_to_interpolate = ::ah::ComputeHorizonVolumeQuantities;
      using tags_to_observe = ::ah::tags_for_observing<Frame::Grid>;
      using compute_items_on_target =
          ::ah::compute_items_on_target<3, Frame::Grid>;
      using compute_target_points =
          intrp::TargetPoints::ApparentHorizon<InterpolationTarget,
                                               ::Frame::Grid>;
      using post_interpolation_callback =
          intrp::callbacks::FindApparentHorizon<InterpolationTarget,
                                                ::Frame::Grid>;
      using horizon_find_failure_callback =
          intrp::callbacks::ErrorOnFailedApparentHorizon;
      using post_horizon_find_callbacks = tmpl::list<
          control_system::RunCallbacks<Submeasurement, ControlSystems>>;
    };

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = InterpolationTarget<ControlSystems>;

    using argument_tags =
        tmpl::push_front<::ah::source_vars<3>, domain::Tags::Mesh<3>>;

    template <typename Metavariables, typename ParallelComponent,
              typename ControlSystems>
    static void apply(
        const Mesh<3>& mesh,
        const tnsr::aa<DataVector, 3, ::Frame::Inertial>& spacetime_metric,
        const tnsr::aa<DataVector, 3, ::Frame::Inertial>& pi,
        const tnsr::iaa<DataVector, 3, ::Frame::Inertial>& phi,
        const tnsr::ijaa<DataVector, 3, ::Frame::Inertial>& deriv_phi,
        const LinkedMessageId<double>& measurement_id,
        Parallel::GlobalCache<Metavariables>& cache,
        const ElementId<3>& array_index,
        const ParallelComponent* const /*meta*/, ControlSystems /*meta*/) {
      intrp::interpolate<interpolation_target_tag<ControlSystems>>(
          measurement_id, mesh, cache, array_index, spacetime_metric, pi, phi,
          deriv_phi);
    }
  };

  using submeasurements = tmpl::list<Submeasurement>;
};
}  // namespace control_system::measurements
