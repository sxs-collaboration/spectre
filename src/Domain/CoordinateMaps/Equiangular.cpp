// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Equiangular.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

Equiangular::Equiangular(const double A, const double B, const double a,
                         const double b) noexcept
    : A_(A),
      B_(B),
      a_(a),
      b_(b),
      length_of_domain_over_m_pi_4_((B - A) / M_PI_4),
      length_of_range_(b - a),
      m_pi_4_over_length_of_domain_(1.0 / length_of_domain_over_m_pi_4_),
      one_over_length_of_range_(1.0 / length_of_range_),
      linear_jacobian_times_m_pi_4_(length_of_range_ /
                                    length_of_domain_over_m_pi_4_),
      linear_inverse_jacobian_over_m_pi_4_(length_of_domain_over_m_pi_4_ /
                                           length_of_range_) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> Equiangular::operator()(
    const std::array<T, 1>& source_coords) const noexcept {
  return {
      {0.5 * (a_ + b_ +
              length_of_range_ * tan(m_pi_4_over_length_of_domain_ *
                                     (-B_ - A_ + 2.0 * source_coords[0])))}};
}

boost::optional<std::array<double, 1>> Equiangular::inverse(
    const std::array<double, 1>& target_coords) const noexcept {
  return {{{0.5 * (A_ + B_ +
                   length_of_domain_over_m_pi_4_ *
                       atan(one_over_length_of_range_ *
                            (-a_ - b_ + 2.0 * target_coords[0])))}}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> Equiangular::jacobian(
    const std::array<T, 1>& source_coords) const noexcept {
  const tt::remove_cvref_wrap_t<T> tan_variable =
      tan(m_pi_4_over_length_of_domain_ * (-B_ - A_ + 2.0 * source_coords[0]));
  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);
  get<0, 0>(jacobian_matrix) =
      linear_jacobian_times_m_pi_4_ * (1.0 + square(tan_variable));
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>
Equiangular::inv_jacobian(const std::array<T, 1>& source_coords) const
    noexcept {
  const tt::remove_cvref_wrap_t<T> tan_variable =
      tan(m_pi_4_over_length_of_domain_ * (-B_ - A_ + 2.0 * source_coords[0]));
  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);
  get<0, 0>(inv_jacobian_matrix) =
      linear_inverse_jacobian_over_m_pi_4_ / (1.0 + square(tan_variable));
  return inv_jacobian_matrix;
}

void Equiangular::pup(PUP::er& p) noexcept {
  p | A_;
  p | B_;
  p | a_;
  p | b_;
  p | length_of_domain_over_m_pi_4_;
  p | length_of_range_;
  p | m_pi_4_over_length_of_domain_;
  p | one_over_length_of_range_;
  p | linear_jacobian_times_m_pi_4_;
  p | linear_inverse_jacobian_over_m_pi_4_;
}

bool operator==(const CoordinateMaps::Equiangular& lhs,
                const CoordinateMaps::Equiangular& rhs) noexcept {
  return lhs.A_ == rhs.A_ and lhs.B_ == rhs.B_ and lhs.a_ == rhs.a_ and
         lhs.b_ == rhs.b_ and
         lhs.length_of_domain_over_m_pi_4_ ==
             rhs.length_of_domain_over_m_pi_4_ and
         lhs.length_of_range_ == rhs.length_of_range_ and
         lhs.m_pi_4_over_length_of_domain_ ==
             rhs.m_pi_4_over_length_of_domain_ and
         lhs.one_over_length_of_range_ == rhs.one_over_length_of_range_ and
         lhs.linear_jacobian_times_m_pi_4_ ==
             rhs.linear_jacobian_times_m_pi_4_ and
         lhs.linear_inverse_jacobian_over_m_pi_4_ ==
             rhs.linear_inverse_jacobian_over_m_pi_4_;
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1> Equiangular::  \
  operator()(const std::array<DTYPE(data), 1>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>  \
  Equiangular::jacobian(const std::array<DTYPE(data), 1>& source_coords)      \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>  \
  Equiangular::inv_jacobian(const std::array<DTYPE(data), 1>& source_coords)  \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
