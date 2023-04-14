// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ExcisionSphere.hpp"

#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>
#include <unordered_map>

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
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | radius_;
    p | center_;
    p | abutting_directions_;
  }
}

template <size_t VolumeDim>
bool operator==(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) {
  return lhs.radius() == rhs.radius() and lhs.center() == rhs.center() and
         lhs.abutting_directions() == rhs.abutting_directions();
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
