// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ApparentHorizons/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ApparentHorizons/Callbacks/FindApparentHorizon.hpp"
#include "ApparentHorizons/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/InterpolationTarget.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
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
 * \brief A `control_system::protocols::Measurement` that relies on one
 * apparent horizon, the template parameter `Object`, and one excision surface.
 */
template <::domain::ObjectLabel Object>
struct CharSpeed : tt::ConformsTo<protocols::Measurement> {
  static std::string name() { return "CharSpeed" + ::domain::name(Object); }

  /*!
   * \brief A `control_system::protocols::Submeasurement` that does an
   * interpolation to the excision boundary for this `Object` from the elements.
   *
   * This does not go through the interpolation framework.
   */
  struct Excision : tt::ConformsTo<protocols::Submeasurement> {
    static std::string name() { return CharSpeed::name() + "::Excision"; }

   private:
    template <typename ControlSystems>
    struct InterpolationTarget
        : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
      static std::string name() {
        return "ControlSystemCharSpeedExcision" + ::domain::name(Object);
      }

      using temporal_id = ::Tags::TimeAndPrevious<1>;

      using vars_to_interpolate_to_target =
          tmpl::list<gr::Tags::Lapse<DataVector>,
                     gr::Tags::Shift<DataVector, 3, Frame::Distorted>,
                     gr::Tags::ShiftyQuantity<DataVector, 3, Frame::Distorted>,
                     gr::Tags::SpatialMetric<DataVector, 3, Frame::Distorted>,
                     gh::ConstraintDamping::Tags::ConstraintGamma1>;
      using compute_vars_to_interpolate =
          ah::ComputeExcisionBoundaryVolumeQuantities;
      using compute_items_on_source = tmpl::list<>;
      using compute_items_on_target =
          tmpl::list<gr::Tags::DetAndInverseSpatialMetricCompute<
              DataVector, 3, Frame::Distorted>>;
      using compute_target_points =
          intrp::TargetPoints::Sphere<InterpolationTarget, ::Frame::Grid>;
      using post_interpolation_callback =
          control_system::RunCallbacks<Excision, ControlSystems>;

      template <typename Metavariables>
      using interpolating_component =
          typename Metavariables::gh_dg_element_array;
    };

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = InterpolationTarget<ControlSystems>;

    using compute_tags_for_observation_box = tmpl::list<>;

    using argument_tags =
        tmpl::push_front<::ah::source_vars<3>, intrp::Tags::InterpPointInfoBase,
                         domain::Tags::Mesh<3>>;

    template <typename Metavariables, typename ParallelComponent,
              typename ControlSystems>
    static void apply(
        const typename intrp::Tags::InterpPointInfo<Metavariables>::type&
            point_infos,
        const Mesh<3>& mesh,
        const tnsr::aa<DataVector, 3, ::Frame::Inertial>& spacetime_metric,
        const tnsr::aa<DataVector, 3, ::Frame::Inertial>& pi,
        const tnsr::iaa<DataVector, 3, ::Frame::Inertial>& phi,
        const tnsr::ijaa<DataVector, 3, ::Frame::Inertial>& deriv_phi,
        const Scalar<DataVector>& constraint_gamma1,
        const LinkedMessageId<double>& measurement_id,
        Parallel::GlobalCache<Metavariables>& cache,
        const ElementId<3>& array_index,
        const ParallelComponent* const component, ControlSystems /*meta*/) {
      using Event = typename intrp::Events::InterpolateWithoutInterpComponent<
          3, InterpolationTarget<ControlSystems>, Metavariables,
          ::ah::source_vars<3>>;

      Event event{};

      // ObservationValue unused
      event(measurement_id, point_infos, mesh, spacetime_metric, pi, phi,
            deriv_phi, constraint_gamma1, cache, array_index, component,
            ::Event::ObservationValue{});
    }
  };

  /*!
   * \brief A `control_system::protocols::Submeasurement` that starts the
   * interpolation to the interpolation target in order to find the apparent
   * horizon.
   */
  struct Horizon : tt::ConformsTo<protocols::Submeasurement> {
    static std::string name() { return CharSpeed::name() + "::Horizon"; }

   private:
    template <typename ControlSystems>
    struct InterpolationTarget
        : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
      static std::string name() {
        return "ControlSystemCharSpeedAh" + ::domain::name(Object);
      }

      using temporal_id = ::Tags::TimeAndPrevious<1>;

      using vars_to_interpolate_to_target =
          ::ah::vars_to_interpolate_to_target<3, ::Frame::Distorted>;
      using compute_vars_to_interpolate = ::ah::ComputeHorizonVolumeQuantities;
      using compute_items_on_target = tmpl::push_back<
          ::ah::compute_items_on_target<3, Frame::Distorted>,
          StrahlkorperTags::TimeDerivStrahlkorperCompute<Frame::Distorted>>;
      using compute_target_points =
          intrp::TargetPoints::ApparentHorizon<InterpolationTarget,
                                               ::Frame::Distorted>;
      using post_interpolation_callback =
          intrp::callbacks::FindApparentHorizon<InterpolationTarget,
                                                ::Frame::Distorted>;
      using horizon_find_failure_callback =
          intrp::callbacks::ErrorOnFailedApparentHorizon;
      using post_horizon_find_callbacks =
          tmpl::list<control_system::RunCallbacks<Horizon, ControlSystems>>;
    };

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = InterpolationTarget<ControlSystems>;

    using compute_tags_for_observation_box = tmpl::list<>;

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
        const Scalar<DataVector>& constraint_gamma1,
        const LinkedMessageId<double>& measurement_id,
        Parallel::GlobalCache<Metavariables>& cache,
        const ElementId<3>& array_index,
        const ParallelComponent* const /*meta*/, ControlSystems /*meta*/) {
      intrp::interpolate<interpolation_target_tag<ControlSystems>>(
          measurement_id, mesh, cache, array_index, spacetime_metric, pi, phi,
          deriv_phi, constraint_gamma1);
    }
  };

  using submeasurements = tmpl::list<Horizon, Excision>;
};
}  // namespace control_system::measurements
