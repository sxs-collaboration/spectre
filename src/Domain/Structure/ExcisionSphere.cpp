// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ExcisionSphere.hpp"

#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
ExcisionSphere<VolumeDim>::ExcisionSphere(
    const double radius, const std::array<double, VolumeDim> center) noexcept
    : radius_(radius), center_(center) {
  ASSERT(radius_ > 0.0,
         "The ExcisionSphere must have a radius greater than zero.");
}

template <size_t VolumeDim>
void ExcisionSphere<VolumeDim>::pup(PUP::er& p) noexcept {
  p | radius_;
  p | center_;
}

template <size_t VolumeDim>
bool operator==(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) noexcept {
  return lhs.radius() == rhs.radius() and lhs.center() == rhs.center();
}

template <size_t VolumeDim>
bool operator!=(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ExcisionSphere<VolumeDim>& sphere) noexcept {
  os << "ExcisionSphere:\n";
  os << "  Radius: " << sphere.radius() << "\n";
  os << "  Center: " << sphere.center() << "\n";
  return os;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template class ExcisionSphere<GET_DIM(data)>;                            \
  template bool operator==(const ExcisionSphere<GET_DIM(data)>&,           \
                           const ExcisionSphere<GET_DIM(data)>&) noexcept; \
  template bool operator!=(const ExcisionSphere<GET_DIM(data)>&,           \
                           const ExcisionSphere<GET_DIM(data)>&) noexcept; \
  template std::ostream& operator<<(                                       \
      std::ostream&, const ExcisionSphere<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
