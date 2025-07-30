// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "IO/H5/TensorData.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Events/ObserveConstantsPerElement.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
enum class FloatingPointType;
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
class TimeDelta;
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
namespace Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
template <size_t VolumeDim, typename Frame>
struct MinimumGridSpacing;
}  // namespace Tags
}  // namespace domain
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename Tag>
struct HistoryEvolvedVariables;
struct Time;
struct TimeStep;
}  // namespace Tags
namespace TimeSteppers {
template <typename Vars>
class History;
}  // namespace TimeSteppers
/// \endcond

namespace dg::Events {
/*!
 * \brief %Observe the time step in the volume.
 *
 * Observe the time step size in each element.  Each element is output
 * as a single cell with two points per dimension and the observation
 * constant on all those points.
 *
 * Writes volume quantities:
 * - InertialCoordinates (only element corners)
 * - Time step
 * - Slab fraction
 * - Minimum grid spacing
 * - Integration order
 */
template <typename System>
class ObserveTimeStepVolume
    : public ObserveConstantsPerElement<System::volume_dim> {
 public:
  static constexpr size_t volume_dim = System::volume_dim;
  static_assert(not tt::is_a_v<tmpl::list, typename System::variables_tag>,
                "Split variables systems not handled.");

  /// \cond
  explicit ObserveTimeStepVolume(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTimeStepVolume);  // NOLINT
  /// \endcond

  static constexpr Options::String help =
      "Observe the time step and integration order in the volume.";

  ObserveTimeStepVolume();

  ObserveTimeStepVolume(const std::string& subfile_name,
                        ::FloatingPointType coordinates_floating_point_type,
                        ::FloatingPointType floating_point_type);

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<
      ::Tags::Time, ::domain::Tags::FunctionsOfTime,
      ::domain::Tags::Domain<volume_dim>, ::Tags::TimeStep,
      domain::Tags::MinimumGridSpacing<volume_dim, Frame::Inertial>,
      ::Tags::HistoryEvolvedVariables<typename System::variables_tag>>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const Domain<volume_dim>& domain, const TimeDelta& time_step,
      const double minimum_grid_spacing,
      const TimeSteppers::History<typename System::variables_tag::type>&
          history,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<volume_dim>& element_id,
      const ParallelComponent* const component,
      const Event::ObservationValue& observation_value) const {
    std::vector<TensorComponent> components =
        assemble_data(time, functions_of_time, domain, element_id, time_step,
                      minimum_grid_spacing, history);

    this->observe(components, cache, element_id, component, observation_value);
  }

  bool needs_evolved_variables() const override;

 private:
  std::vector<TensorComponent> assemble_data(
      double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const Domain<volume_dim>& domain, const ElementId<volume_dim>& element_id,
      const TimeDelta& time_step, double minimum_grid_spacing,
      const TimeSteppers::History<typename System::variables_tag::type>&
          history) const;
};
}  // namespace dg::Events
