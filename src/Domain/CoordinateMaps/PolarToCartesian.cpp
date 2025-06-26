// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/PolarToCartesian.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {
PolarToCartesian::PolarToCartesian() = default;
PolarToCartesian::PolarToCartesian(PolarToCartesian&&) = default;
PolarToCartesian::PolarToCartesian(const PolarToCartesian&) = default;
PolarToCartesian& PolarToCartesian::operator=(const PolarToCartesian&) =
    default;
PolarToCartesian& PolarToCartesian::operator=(PolarToCartesian&&) = default;

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 2> PolarToCartesian::operator()(
    const std::array<T, 2>& source_coords) const {
  const auto& [r, phi] = source_coords;
  return {{r * cos(phi), r * sin(phi)}};
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::optional<std::array<double, 2>> PolarToCartesian::inverse(
    const std::array<double, 2>& target_coords) const {
  const auto& [x, y] = target_coords;
  if (UNLIKELY(y == 0.0 and x == 0.0)) {
    return std::array{0.0, 0.0};
  } else {
    const double r = std::hypot(x, y);
    const double phi = atan2(y, x);
    return std::array{r, phi < 0.0 ? phi + 2.0 * M_PI : phi};
  }
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>
PolarToCartesian::jacobian(const std::array<T, 2>& source_coords) const {
  const auto& [r, phi] = source_coords;
  using DataType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<DataType, 2, Frame::NoFrame> jacobian_matrix{
      make_with_value<DataType>(dereference_wrapper(r), 0.0)};
  const auto& cos_phi = get<0, 0>(jacobian_matrix) = cos(phi);
  const auto& sin_phi = get<1, 0>(jacobian_matrix) = sin(phi);
  get<0, 1>(jacobian_matrix) = -r * sin_phi;
  get<1, 1>(jacobian_matrix) = r * cos_phi;
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>
PolarToCartesian::inv_jacobian(const std::array<T, 2>& source_coords) const {
  const auto& [r, phi] = source_coords;
  using DataType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<DataType, 2, Frame::NoFrame> inv_jacobian_matrix{
      make_with_value<DataType>(dereference_wrapper(r), 0.0)};
  const auto& cos_phi = get<0, 0>(inv_jacobian_matrix) = cos(phi);
  const auto& sin_phi = get<0, 1>(inv_jacobian_matrix) = sin(phi);
  const auto& one_over_r = get<1, 1>(inv_jacobian_matrix) = 1.0 / r;
  get<1, 0>(inv_jacobian_matrix) = -one_over_r * sin_phi;
  get<1, 1>(inv_jacobian_matrix) *= cos_phi;
  return inv_jacobian_matrix;
}

void PolarToCartesian::pup(PUP::er& /*p*/) {}

bool operator==(const PolarToCartesian& /*lhs*/,
                const PolarToCartesian& /*rhs*/) {
  return true;
}

bool operator!=(const PolarToCartesian& lhs, const PolarToCartesian& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                                            \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 2>                \
  PolarToCartesian::operator()(                                               \
      const std::array<DTYPE(data), 2>& source_coords) const;                 \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 2, Frame::NoFrame>  \
  PolarToCartesian::jacobian(const std::array<DTYPE(data), 2>& source_coords) \
      const;                                                                  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 2, Frame::NoFrame>  \
  PolarToCartesian::inv_jacobian(                                             \
      const std::array<DTYPE(data), 2>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
}  // namespace domain::CoordinateMaps
