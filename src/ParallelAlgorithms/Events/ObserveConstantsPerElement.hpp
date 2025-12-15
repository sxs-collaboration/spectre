// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/FloatingPointType.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Domain;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace observers {
class ObservationKey;
enum class TypeOfObservation;
}  // namespace observers
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace dg::Events {
/// \brief Base class for Observers that write data that is constant within an
/// Element
template <size_t VolumeDim>
class ObserveConstantsPerElement : public Event {
 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

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

  ObserveConstantsPerElement() = default;

  ObserveConstantsPerElement(
      const std::string& subfile_name,
      ::FloatingPointType coordinates_floating_point_type,
      ::FloatingPointType floating_point_type);

  /// \cond
  explicit ObserveConstantsPerElement(CkMigrateMessage* /*unused*/);
  /// \endcond

  using observation_registration_tags = tmpl::list<>;

  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration() const;

  using is_ready_argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(const double time, Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index,
                const Component* const component) const {
    return ::domain::functions_of_time_are_ready_algorithm_callback<
        ::domain::Tags::FunctionsOfTime, VolumeDim>(cache, array_index,
                                                    component, time);
  }

  void pup(PUP::er& p) override;

 protected:
  /// \brief Creates the vector of tensor components that will be observed
  /// and starts to fill it with the inertial coordinates
  ///
  /// \details number_of_constants is the number of additional tensor components
  /// that will be observed.  These are added by calling add_constant
  std::vector<TensorComponent> allocate_and_insert_coords(
      size_t number_of_constants, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const Domain<VolumeDim>& domain,
      const ElementId<VolumeDim>& element_id) const;

  /// \brief Adds a TensorComponent to components with the given name and the
  /// given value.
  ///
  /// \details Should be called after allocate_and_insert_coords
  void add_constant(gsl::not_null<std::vector<TensorComponent>*> components,
                    std::string name, double value) const;

  /// Observes the components as volume data
  template <typename Metavariables, typename ParallelComponent>
  void observe(std::vector<TensorComponent> components,
               Parallel::GlobalCache<Metavariables>& cache,
               const ElementId<VolumeDim>& element_id,
               const ParallelComponent* /*component*/,
               const ObservationValue& observation_value) const;
private:
  std::string subfile_path_;
  ::FloatingPointType coordinates_floating_point_type_ =
      ::FloatingPointType::Double;
  ::FloatingPointType floating_point_type_ = ::FloatingPointType::Double;
};

template <size_t VolumeDim>
template <typename Metavariables, typename ParallelComponent>
void ObserveConstantsPerElement<VolumeDim>::observe(
    std::vector<TensorComponent> components,
    Parallel::GlobalCache<Metavariables>& cache,
    const ElementId<VolumeDim>& element_id,
    const ParallelComponent* const /*component*/,
    const ObservationValue& observation_value) const {
  const Mesh<VolumeDim> single_cell_mesh(2, Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto);
  const Parallel::ArrayComponentId array_component_id{
      std::add_pointer_t<ParallelComponent>{nullptr},
      Parallel::ArrayIndex<ElementId<VolumeDim>>{element_id}};
  ElementVolumeData element_volume_data{element_id, std::move(components),
                                        single_cell_mesh};
  observers::ObservationId observation_id{observation_value.value,
                                          subfile_path_ + ".vol"};

  observers::contribute_volume_data<
      not Parallel::is_nodegroup_v<ParallelComponent>>(
      cache, std::move(observation_id), subfile_path_, array_component_id,
      std::move(element_volume_data));
}

}  // namespace dg::Events
