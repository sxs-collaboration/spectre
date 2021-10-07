// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ObjectLabel.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
    struct InterpolationTarget {
      static std::string name() {
        return "ControlSystem::BothHorizons::Ah" + ::ah::name(Horizon);
      }

      struct temporal_id : db::SimpleTag {
        using type = LinkedMessageId<double>;
      };

      using vars_to_interpolate_to_target =
          tmpl::list<gr::Tags::SpatialMetric<3, ::Frame::Grid, DataVector>,
                     gr::Tags::InverseSpatialMetric<3, ::Frame::Grid>,
                     gr::Tags::ExtrinsicCurvature<3, ::Frame::Grid>,
                     gr::Tags::SpatialChristoffelSecondKind<3, ::Frame::Grid>>;
      using compute_vars_to_interpolate = ::ah::ComputeHorizonVolumeQuantities;
      using compute_items_on_target = tmpl::list<
          StrahlkorperTags::ThetaPhiCompute<::Frame::Grid>,
          StrahlkorperTags::RadiusCompute<::Frame::Grid>,
          StrahlkorperTags::RhatCompute<::Frame::Grid>,
          StrahlkorperTags::InvJacobianCompute<::Frame::Grid>,
          StrahlkorperTags::DxRadiusCompute<::Frame::Grid>,
          StrahlkorperTags::OneOverOneFormMagnitudeCompute<3, ::Frame::Grid,
                                                           DataVector>,
          StrahlkorperTags::NormalOneFormCompute<::Frame::Grid>,
          StrahlkorperTags::UnitNormalOneFormCompute<::Frame::Grid>,
          StrahlkorperTags::UnitNormalVectorCompute<::Frame::Grid>,
          StrahlkorperTags::GradUnitNormalOneFormCompute<::Frame::Grid>,
          StrahlkorperTags::ExtrinsicCurvatureCompute<::Frame::Grid>>;
      using compute_target_points =
          intrp::TargetPoints::ApparentHorizon<InterpolationTarget,
                                               ::Frame::Grid>;
      using post_interpolation_callback =
          intrp::callbacks::FindApparentHorizon<InterpolationTarget,
                                                ::Frame::Grid>;
      using horizon_find_failure_callback =
          intrp::callbacks::ErrorOnFailedApparentHorizon;
      using post_horizon_find_callback =
          control_system::RunCallbacks<FindHorizon, ControlSystems>;
    };

    using source_tensors =
        tmpl::list<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>,
                   GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
                   GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>;

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = InterpolationTarget<ControlSystems>;

    using argument_tags =
        tmpl::push_front<source_tensors, domain::Tags::Mesh<3>>;

    template <typename Metavariables, typename ParallelComponent,
              typename ControlSystems>
    static void apply(
        const Mesh<3>& mesh,
        const tnsr::aa<DataVector, 3, ::Frame::Inertial>& spacetime_metric,
        const tnsr::aa<DataVector, 3, ::Frame::Inertial>& pi,
        const tnsr::iaa<DataVector, 3, ::Frame::Inertial>& phi,
        const LinkedMessageId<double>& measurement_id,
        Parallel::GlobalCache<Metavariables>& cache,
        const ElementId<3>& array_index,
        const ParallelComponent* const /*meta*/, ControlSystems /*meta*/) {
      intrp::interpolate<interpolation_target_tag<ControlSystems>,
                         source_tensors>(
          measurement_id, mesh, cache, array_index, spacetime_metric, pi, phi);
    }
  };

  using submeasurements = tmpl::list<FindHorizon<::ah::ObjectLabel::A>,
                                     FindHorizon<::ah::ObjectLabel::B>>;
};
}  // namespace control_system::ah
