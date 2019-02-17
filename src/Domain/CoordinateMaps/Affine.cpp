// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Affine.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

Affine::Affine(const double A, const double B, const double a, const double b)
    : A_(A),
      B_(B),
      a_(a),
      b_(b),
      length_of_domain_(B - A),
      length_of_range_(b - a),
      jacobian_(length_of_range_ / length_of_domain_),
      inverse_jacobian_(length_of_domain_ / length_of_range_),
      is_identity_(A == a and B == b) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> Affine::operator()(
    const std::array<T, 1>& source_coords) const noexcept {
  return {{(length_of_range_ * source_coords[0] + a_ * B_ - b_ * A_) /
           length_of_domain_}};
}

boost::optional<std::array<double, 1>> Affine::inverse(
    const std::array<double, 1>& target_coords) const noexcept {
  return {{{(length_of_domain_ * target_coords[0] - a_ * B_ + b_ * A_) /
            length_of_range_}}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> Affine::jacobian(
    const std::array<T, 1>& source_coords) const noexcept {
  return make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), jacobian_);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> Affine::inv_jacobian(
    const std::array<T, 1>& source_coords) const noexcept {
  return make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), inverse_jacobian_);
}

void Affine::pup(PUP::er& p) {
  p | A_;
  p | B_;
  p | a_;
  p | b_;
  p | length_of_domain_;
  p | length_of_range_;
  p | jacobian_;
  p | inverse_jacobian_;
  p | is_identity_;
}

bool operator==(const CoordinateMaps::Affine& lhs,
                const CoordinateMaps::Affine& rhs) noexcept {
  return lhs.A_ == rhs.A_ and lhs.B_ == rhs.B_ and lhs.a_ == rhs.a_ and
         lhs.b_ == rhs.b_ and lhs.length_of_domain_ == rhs.length_of_domain_ and
         lhs.length_of_range_ == rhs.length_of_range_ and
         lhs.jacobian_ == rhs.jacobian_ and
         lhs.inverse_jacobian_ == rhs.inverse_jacobian_ and
         lhs.is_identity_ == rhs.is_identity_;
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1> Affine::       \
  operator()(const std::array<DTYPE(data), 1>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>  \
  Affine::jacobian(const std::array<DTYPE(data), 1>& source_coords)           \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>  \
  Affine::inv_jacobian(const std::array<DTYPE(data), 1>& source_coords)       \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
