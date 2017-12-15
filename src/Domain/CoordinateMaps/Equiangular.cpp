// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Equiangular.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

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
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 1> Equiangular::
operator()(const std::array<T, 1>& source_coords) const noexcept {
  return {
      {0.5 * (a_ + b_ +
              length_of_range_ * tan(m_pi_4_over_length_of_domain_ *
                                     (-B_ - A_ + 2.0 * source_coords[0])))}};
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 1>
Equiangular::inverse(const std::array<T, 1>& target_coords) const noexcept {
  return {{0.5 * (A_ + B_ +
                  length_of_domain_over_m_pi_4_ *
                      atan(one_over_length_of_range_ *
                           (-a_ - b_ + 2.0 * target_coords[0])))}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Equiangular::jacobian(const std::array<T, 1>& source_coords) const noexcept {
  const std::decay_t<tt::remove_reference_wrapper_t<T>> tan_variable =
      tan(m_pi_4_over_length_of_domain_ * (-B_ - A_ + 2.0 * source_coords[0]));
  return Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>{
      linear_jacobian_times_m_pi_4_ * (1.0 + square(tan_variable))};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Equiangular::inv_jacobian(const std::array<T, 1>& source_coords) const
    noexcept {
  const std::decay_t<tt::remove_reference_wrapper_t<T>> tan_variable =
      tan(m_pi_4_over_length_of_domain_ * (-B_ - A_ + 2.0 * source_coords[0]));
  return Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>{
      linear_inverse_jacobian_over_m_pi_4_ / (1.0 + square(tan_variable))};
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

#define INSTANTIATE(_, data)                                                 \
  template std::array<DTYPE(data), 1> Equiangular::operator()(               \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 1>&        \
          source_coords) const noexcept;                                     \
  template std::array<DTYPE(data), 1> Equiangular::operator()(               \
      const std::array<DTYPE(data), 1>& source_coords) const noexcept;       \
  template std::array<DTYPE(data), 1> Equiangular::inverse(                  \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 1>&        \
          target_coords) const noexcept;                                     \
  template std::array<DTYPE(data), 1> Equiangular::inverse(                  \
      const std::array<DTYPE(data), 1>& target_coords) const noexcept;       \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,      \
                  index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,      \
                             SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>     \
  Equiangular::jacobian(                                                     \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 1>&        \
          source_coords) const noexcept;                                     \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,      \
                  index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,      \
                             SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>     \
  Equiangular::jacobian(const std::array<DTYPE(data), 1>& source_coords)     \
      const noexcept;                                                        \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,      \
                  index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,      \
                             SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>     \
  Equiangular::inv_jacobian(                                                 \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 1>&        \
          source_coords) const noexcept;                                     \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,      \
                  index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,      \
                             SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>     \
  Equiangular::inv_jacobian(const std::array<DTYPE(data), 1>& source_coords) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
