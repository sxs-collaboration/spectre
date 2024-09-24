// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/FloatingPointType.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Domain;
class TimeDelta;
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
}  // namespace domain::Tags
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
 */
template <size_t VolumeDim>
class ObserveTimeStepVolume : public Event {
 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveTimeStepVolume(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTimeStepVolume);  // NOLINT
  /// \endcond

  /// The floating point type/precision with which to write the data to disk.
  struct FloatingPointType {
    static constexpr Options::String help =
        "The floating point type/precision with which to write the data to "
        "disk.";
    using type = ::FloatingPointType;
  };

  /// The floating point type/precision with which to write the coordinates to
  /// disk.
  struct CoordinatesFloatingPointType {
    static constexpr Options::String help =
        "The floating point type/precision with which to write the coordinates "
        "to disk.";
    using type = ::FloatingPointType;
  };

  using options =
      tmpl::list<SubfileName, CoordinatesFloatingPointType, FloatingPointType>;

  static constexpr Options::String help =
      "Observe the time step in the volume.";

  ObserveTimeStepVolume() = default;

  ObserveTimeStepVolume(const std::string& subfile_name,
                        ::FloatingPointType coordinates_floating_point_type,
                        ::FloatingPointType floating_point_type);

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Time, ::domain::Tags::FunctionsOfTime,
                 ::domain::Tags::Domain<VolumeDim>, ::Tags::TimeStep,
                 domain::Tags::MinimumGridSpacing<VolumeDim, Frame::Inertial>>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const double time,
                  const domain::FunctionsOfTimeMap& functions_of_time,
                  const Domain<VolumeDim>& domain, const TimeDelta& time_step,
                  const double minimum_grid_spacing,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& element_id,
                  const ParallelComponent* const /*component*/,
                  const ObservationValue& observation_value) const {
    std::vector<TensorComponent> components =
        assemble_data(time, functions_of_time, domain, element_id, time_step,
                      minimum_grid_spacing);

    const Mesh<VolumeDim> single_cell_mesh(2, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto);
    const Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<VolumeDim>>{element_id}};
    ElementVolumeData element_volume_data{element_id, std::move(components),
                                          single_cell_mesh};
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_path_ + ".vol"};

    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));

    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      // Send data to reduction observer writer (nodegroup)
      std::unordered_map<Parallel::ArrayComponentId,
                         std::vector<ElementVolumeData>>
          data_to_send{};
      data_to_send[array_component_id] =
          std::vector{std::move(element_volume_data)};
      Parallel::threaded_action<
          observers::ThreadedActions::ContributeVolumeDataToWriter>(
          local_observer, std::move(observation_id), array_component_id,
          subfile_path_, std::move(data_to_send));
    } else {
      // Send data to volume observer
      Parallel::simple_action<observers::Actions::ContributeVolumeData>(
          local_observer, std::move(observation_id), subfile_path_,
          array_component_id, std::move(element_volume_data));
    }
  }

  using observation_registration_tags = tmpl::list<>;

  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration() const;

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override;

  void pup(PUP::er& p) override;

 private:
  std::vector<TensorComponent> assemble_data(
      double time, const domain::FunctionsOfTimeMap& functions_of_time,
      const Domain<VolumeDim>& domain, const ElementId<VolumeDim>& element_id,
      const TimeDelta& time_step, double minimum_grid_spacing) const;

  std::string subfile_path_;
  ::FloatingPointType coordinates_floating_point_type_ =
      ::FloatingPointType::Double;
  ::FloatingPointType floating_point_type_ = ::FloatingPointType::Double;
};
}  // namespace dg::Events
