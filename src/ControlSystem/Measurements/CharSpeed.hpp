// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
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
template <size_t Dim, typename Frame>
struct Coordinates;
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
                     gr::Tags::SpatialMetric<DataVector, 3, Frame::Distorted>>;
      using compute_vars_to_interpolate =
          ah::ComputeExcisionBoundaryVolumeQuantities;
      using compute_items_on_source =
          tmpl::list<::Tags::TimeAndPreviousCompute<1>>;
      using compute_items_on_target =
          tmpl::list<gr::Tags::DetAndInverseSpatialMetricCompute<
              DataVector, 3, Frame::Distorted>>;
      using compute_target_points =
          intrp::TargetPoints::Sphere<InterpolationTarget, ::Frame::Grid>;
      using post_interpolation_callbacks =
          tmpl::list<control_system::RunCallbacks<Excision, ControlSystems>>;

      template <typename Metavariables>
      using interpolating_component =
          typename Metavariables::gh_dg_element_array;
    };

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = InterpolationTarget<ControlSystems>;

    template <typename ControlSystems>
    using event = intrp::Events::InterpolateWithoutInterpComponent<
        3, InterpolationTarget<ControlSystems>, ::ah::source_vars<3>>;
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
     private:
      static constexpr size_t index =
          Object == ::domain::ObjectLabel::A ? 1_st : 2_st;

     public:
      static std::string name() {
        return "ControlSystemCharSpeedAh" + ::domain::name(Object);
      }

      // Separate temporal IDs for each object
      using temporal_id = ::Tags::TimeAndPrevious<index>;

      using vars_to_interpolate_to_target =
          ::ah::vars_to_interpolate_to_target<3, ::Frame::Distorted>;
      using compute_vars_to_interpolate = ::ah::ComputeHorizonVolumeQuantities;
      using compute_items_on_source =
          tmpl::list<::Tags::TimeAndPreviousCompute<index>>;
      using compute_items_on_target = tmpl::push_back<
          ::ah::compute_items_on_target<3, Frame::Distorted>,
          ylm::Tags::TimeDerivStrahlkorperCompute<Frame::Distorted>>;
      using compute_target_points =
          intrp::TargetPoints::ApparentHorizon<InterpolationTarget,
                                               ::Frame::Distorted>;
      using post_interpolation_callbacks =
          tmpl::list<intrp::callbacks::FindApparentHorizon<InterpolationTarget,
                                                           ::Frame::Distorted>>;
      using horizon_find_failure_callback =
          intrp::callbacks::ErrorOnFailedApparentHorizon;
      using post_horizon_find_callbacks =
          tmpl::list<control_system::RunCallbacks<Horizon, ControlSystems>>;
    };

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = InterpolationTarget<ControlSystems>;

    template <typename ControlSystems>
    using event =
        intrp::Events::Interpolate<3, InterpolationTarget<ControlSystems>,
                                   ::ah::source_vars<3>>;
  };

  using submeasurements = tmpl::list<Horizon, Excision>;
};
}  // namespace control_system::measurements
