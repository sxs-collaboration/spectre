// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"

#include <cstddef>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct ElementLogical;
struct Inertial;
}  // namespace Frame

namespace dg::Events {
template <size_t VolumeDim>
ObserveTimeStepVolume<VolumeDim>::ObserveTimeStepVolume(
    const std::string& subfile_name,
    const ::FloatingPointType coordinates_floating_point_type,
    const ::FloatingPointType floating_point_type)
    : subfile_path_("/" + subfile_name),
      coordinates_floating_point_type_(coordinates_floating_point_type),
      floating_point_type_(floating_point_type) {}

template <size_t VolumeDim>
std::optional<
    std::pair<observers::TypeOfObservation, observers::ObservationKey>>
ObserveTimeStepVolume<
    VolumeDim>::get_observation_type_and_key_for_registration() const {
  return {{observers::TypeOfObservation::Volume,
           observers::ObservationKey{subfile_path_ + ".vol"}}};
}

template <size_t VolumeDim>
bool ObserveTimeStepVolume<VolumeDim>::needs_evolved_variables() const {
  return false;
}

template <size_t VolumeDim>
void ObserveTimeStepVolume<VolumeDim>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
  p | coordinates_floating_point_type_;
  p | floating_point_type_;
}

namespace {
template <typename T, size_t VolumeDim>
void map_corners(const gsl::not_null<std::vector<TensorComponent>*> components,
                 const double time,
                 const domain::FunctionsOfTimeMap& functions_of_time,
                 const Domain<VolumeDim>& domain,
                 const ElementId<VolumeDim>& element_id) {
  // Get the name from the tag even though we don't fetch it to make
  // sure it's consistent with other observers.
  using coordinates_tag =
      ::domain::Tags::Coordinates<VolumeDim, Frame::Inertial>;
  const std::string coordinates_name = db::tag_name<coordinates_tag>();

  const ElementMap<VolumeDim, Frame::Inertial> map(
      element_id, domain.blocks()[element_id.block_id()]);

  auto corners = make_array<VolumeDim, T>(two_to_the(VolumeDim));
  size_t index = 0;

  const auto add_point =
      [&](const tnsr::I<double, VolumeDim, Frame::ElementLogical>& point) {
        const tnsr::I<double, VolumeDim, Frame::Inertial> mapped =
            map(point, time, functions_of_time);
        for (size_t i = 0; i < VolumeDim; ++i) {
          gsl::at(corners, i)[index] = mapped.get(i);
        }
        ++index;
      };

  if constexpr (VolumeDim == 1) {
    for (auto xi : {-1.0, 1.0}) {
      add_point(tnsr::I<double, 1, Frame::ElementLogical>{{xi}});
    }
  } else if constexpr (VolumeDim == 2) {
    for (auto eta : {-1.0, 1.0}) {
      for (auto xi : {-1.0, 1.0}) {
        add_point(tnsr::I<double, 2, Frame::ElementLogical>{{xi, eta}});
      }
    }
  } else {
    static_assert(VolumeDim == 3);
    for (auto zeta : {-1.0, 1.0}) {
      for (auto eta : {-1.0, 1.0}) {
        for (auto xi : {-1.0, 1.0}) {
          add_point(tnsr::I<double, 3, Frame::ElementLogical>{{xi, eta, zeta}});
        }
      }
    }
  }

  for (size_t i = 0; i < VolumeDim; ++i) {
    const std::string component_name =
        coordinates_name + coordinates_tag::type::component_suffix(i);
    components->emplace_back(component_name, std::move(gsl::at(corners, i)));
  }
}
}  // namespace

template <size_t VolumeDim>
std::vector<TensorComponent> ObserveTimeStepVolume<VolumeDim>::assemble_data(
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const Domain<VolumeDim>& domain, const ElementId<VolumeDim>& element_id,
    const TimeDelta& time_step, const double minimum_grid_spacing) const {
  constexpr auto num_corners = two_to_the(VolumeDim);

  std::vector<TensorComponent> components{};
  components.reserve(VolumeDim + 2);

  if (coordinates_floating_point_type_ == ::FloatingPointType::Float) {
    map_corners<std::vector<float>>(&components, time, functions_of_time,
                                    domain, element_id);
  } else {
    map_corners<DataVector>(&components, time, functions_of_time, domain,
                            element_id);
  }

  // We could save some space by using cell-centered data instead of
  // the same value at each corner, but that would require also adding
  // support to the rest of our visualization tools, and we'd still
  // need the coordinates at the corners which would be most of the
  // data.
  const auto add_constant = [&](std::string name, const double value) {
    if (floating_point_type_ == ::FloatingPointType::Float) {
      components.emplace_back(
          std::move(name),
          std::vector<float>(num_corners, static_cast<float>(value)));
    } else {
      components.emplace_back(std::move(name), DataVector(num_corners, value));
    }
  };
  add_constant("Time step", time_step.value());
  add_constant("Slab fraction", time_step.fraction().value());
  add_constant("Minimum grid spacing", minimum_grid_spacing);

  return components;
}

template <size_t VolumeDim>
PUP::able::PUP_ID ObserveTimeStepVolume<VolumeDim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ObserveTimeStepVolume<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace dg::Events
