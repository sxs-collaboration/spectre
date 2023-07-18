// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ExcisionSphere.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>
#include <unordered_map>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
ExcisionSphere<VolumeDim>::ExcisionSphere(
    const double radius, const tnsr::I<double, VolumeDim, Frame::Grid> center,
    std::unordered_map<size_t, Direction<VolumeDim>> abutting_directions)
    : radius_(radius),
      center_(center),
      abutting_directions_(std::move(abutting_directions)) {
  ASSERT(radius_ > 0.0,
         "The ExcisionSphere must have a radius greater than zero.");
}

template <size_t VolumeDim>
ExcisionSphere<VolumeDim>::ExcisionSphere(
    const ExcisionSphere<VolumeDim>& rhs) {
  *this = rhs;
}

template <size_t VolumeDim>
ExcisionSphere<VolumeDim>& ExcisionSphere<VolumeDim>::operator=(
    const ExcisionSphere<VolumeDim>& rhs) {
  radius_ = rhs.radius_;
  center_ = rhs.center_;
  abutting_directions_ = rhs.abutting_directions_;
  if (rhs.grid_to_inertial_map_ == nullptr) {
    grid_to_inertial_map_ = nullptr;
  } else {
    grid_to_inertial_map_ = rhs.grid_to_inertial_map_->get_clone();
  }
  return *this;
}

template <size_t VolumeDim>
void ExcisionSphere<VolumeDim>::inject_time_dependent_maps(
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
        moving_mesh_grid_to_inertial_map) {
  grid_to_inertial_map_ = std::move(moving_mesh_grid_to_inertial_map);
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
ExcisionSphere<VolumeDim>::moving_mesh_grid_to_inertial_map() const {
  ASSERT(
      grid_to_inertial_map_ != nullptr,
      "Trying to access grid to inertial map from ExcisionSphere, but it has "
      "not been set.");

  return *grid_to_inertial_map_;
}

template <size_t VolumeDim>
std::optional<Direction<VolumeDim>>
ExcisionSphere<VolumeDim>::abutting_direction(
    const ElementId<VolumeDim>& element_id) const {
  const size_t block_id = element_id.block_id();
  if (abutting_directions_.count(block_id)) {
    const auto& direction = abutting_directions_.at(block_id);
    const auto& segment_id = element_id.segment_id(direction.dimension());
    const size_t abutting_index =
        direction.side() == Side::Lower
            ? 0
            : two_to_the(segment_id.refinement_level()) - 1;
    if (segment_id.index() == abutting_index) {
      return direction;
    }
  }
  return std::nullopt;
}

template <size_t VolumeDim>
void ExcisionSphere<VolumeDim>::pup(PUP::er& p) {
  size_t version = 1;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | radius_;
    p | center_;
    p | abutting_directions_;
  }
  if (version >= 1) {
    p | grid_to_inertial_map_;
  } else if (p.isUnpacking()) {
    grid_to_inertial_map_ = nullptr;
  }
}

template <size_t VolumeDim>
bool operator==(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) {
  // First check time dep maps. If both are time dep, safe to dereference
  // unique_ptr
  if (lhs.is_time_dependent() != rhs.is_time_dependent()) {
    return false;
  }
  bool equal = lhs.radius_ == rhs.radius_ and lhs.center_ == rhs.center_ and
               lhs.abutting_directions_ == rhs.abutting_directions_;
  if (lhs.is_time_dependent() and rhs.is_time_dependent()) {
    equal = equal and *lhs.grid_to_inertial_map_ == *rhs.grid_to_inertial_map_;
  }

  return equal;
}

template <size_t VolumeDim>
bool operator!=(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ExcisionSphere<VolumeDim>& sphere) {
  os << "ExcisionSphere:\n";
  os << "  Radius: " << sphere.radius() << "\n";
  os << "  Center: " << sphere.center() << "\n";
  os << "  Abutting directions: " << sphere.abutting_directions() << "\n";
  return os;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                    \
  template class ExcisionSphere<GET_DIM(data)>;                   \
  template bool operator==(const ExcisionSphere<GET_DIM(data)>&,  \
                           const ExcisionSphere<GET_DIM(data)>&); \
  template bool operator!=(const ExcisionSphere<GET_DIM(data)>&,  \
                           const ExcisionSphere<GET_DIM(data)>&); \
  template std::ostream& operator<<(std::ostream&,                \
                                    const ExcisionSphere<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
