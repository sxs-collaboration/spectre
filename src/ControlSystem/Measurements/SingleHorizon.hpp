// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ControlSystem/Measurements/NonFactoryCreatable.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FailedHorizonFind.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
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
 * \brief A `control_system::protocols::Measurement` that relies on only one
 * apparent horizon; the template parameter `Horizon`.
 */
template <::domain::ObjectLabel Horizon>
struct SingleHorizon : tt::ConformsTo<protocols::Measurement> {
  static std::string name() {
    return "SingleHorizon" + ::domain::name(Horizon);
  }

  /*!
   * \brief A `control_system::protocols::Submeasurement` that starts the
   * interpolation to the interpolation target in order to find the apparent
   * horizon.
   */
  struct Submeasurement : tt::ConformsTo<protocols::Submeasurement> {
    static std::string name() { return SingleHorizon::name(); }

   private:
    template <typename ControlSystems>
    struct HorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
      static std::string name() {
        return "ControlSystemSingleAh" + ::domain::name(Horizon);
      }

      using time_tag = ::Tags::TimeAndPrevious<0>;

      using frame = ::Frame::Distorted;

      using horizon_find_callbacks = tmpl::list<
          control_system::RunCallbacks<Submeasurement, ControlSystems>>;
      using horizon_find_failure_callbacks =
          tmpl::list<ah::callbacks::FailedHorizonFind<HorizonMetavars, false>>;

      using compute_tags_on_element =
          tmpl::list<::Tags::TimeAndPreviousCompute<0>>;

      static constexpr ah::Destination destination =
          ah::Destination::ControlSystem;
    };

   public:
    template <typename ControlSystems>
    using interpolation_target_tag = void;
    template <typename ControlSystems>
    using horizon_metavars = HorizonMetavars<ControlSystems>;

    template <typename ControlSystems>
    using event = NonFactoryCreatableWrapper<
        ah::Events::FindApparentHorizon<HorizonMetavars<ControlSystems>>>;
  };

  using submeasurements = tmpl::list<Submeasurement>;
};
}  // namespace control_system::measurements
