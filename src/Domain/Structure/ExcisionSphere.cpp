// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ExcisionSphere.hpp"

#include <cstddef>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>
#include <unordered_map>

#include "Domain/Structure/Direction.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
ExcisionSphere<VolumeDim>::ExcisionSphere(
    const double radius, const std::array<double, VolumeDim> center,
    std::unordered_map<size_t, Direction<VolumeDim>> abutting_directions)
    : radius_(radius),
      center_(center),
      abutting_directions_(std::move(abutting_directions)) {
  ASSERT(radius_ > 0.0,
         "The ExcisionSphere must have a radius greater than zero.");
}

template <size_t VolumeDim>
void ExcisionSphere<VolumeDim>::pup(PUP::er& p) {
  p | radius_;
  p | center_;
  p | abutting_directions_;
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
