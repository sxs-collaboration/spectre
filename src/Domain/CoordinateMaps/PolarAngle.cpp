// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/PolarAngle.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> PolarAngle::operator()(
    const std::array<T, 1>& xi) const {
  return {{acos(-xi[0])}};
}

std::optional<std::array<double, 1>> PolarAngle::inverse(
    const std::array<double, 1>& theta) const {
  return {{-cos(theta[0])}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> PolarAngle::jacobian(
    const std::array<T, 1>& xi) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<ReturnType, 1, Frame::NoFrame> result{};
  get<0, 0>(result) = 1. / sqrt(1. - square(xi[0]));
  return result;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>
PolarAngle::inv_jacobian(const std::array<T, 1>& xi) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<ReturnType, 1, Frame::NoFrame> result{};
  get<0, 0>(result) = sqrt(1. - square(xi[0]));
  return result;
}

void PolarAngle::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
}

bool operator!=(const PolarAngle& lhs, const PolarAngle& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1>               \
  PolarAngle::operator()(const std::array<DTYPE(data), 1>& xi) const;        \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame> \
  PolarAngle::jacobian(const std::array<DTYPE(data), 1>& xi) const;          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame> \
  PolarAngle::inv_jacobian(const std::array<DTYPE(data), 1>& xi) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE_DTYPE
}  // namespace domain::CoordinateMaps
