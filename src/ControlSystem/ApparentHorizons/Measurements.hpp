// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/ObjectLabel.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
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

namespace control_system::ah {
struct BothHorizons : tt::ConformsTo<protocols::Measurement> {
  template <::ah::ObjectLabel Horizon>
  struct FindHorizon : tt::ConformsTo<protocols::Submeasurement> {
   private:
    template <typename ControlSystems>
    struct InterpolationTarget
        : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
      static std::string name() {
        return "ControlSystemAh" + ::ah::name(Horizon);
      }

      struct temporal_id : db::SimpleTag {
        using type = LinkedMessageId<double>;
      };

      using vars_to_interpolate_to_target =
          ::ah::vars_to_interpolate_to_target<3, ::Frame::Grid>;
      using compute_vars_to_interpolate = ::ah::ComputeHorizonVolumeQuantities;
      using compute_items_on_target = tmpl::list<>;
      using compute_target_points =
          intrp::TargetPoints::ApparentHorizon<InterpolationTarget,
                                               ::Frame::Grid>;
      using post_interpolation_callback =
          intrp::callbacks::FindApparentHorizon<InterpolationTarget,
                                                ::Frame::Grid>;
      using horizon_find_failure_callback =
          intrp::callbacks::ErrorOnFailedApparentHorizon;
      using post_horizon_find_callbacks =
          tmpl::list<control_system::RunCallbacks<FindHorizon, ControlSystems>>;
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

  using submeasurements = tmpl::list<FindHorizon<::ah::ObjectLabel::A>,
                                     FindHorizon<::ah::ObjectLabel::B>>;
};
}  // namespace control_system::ah
