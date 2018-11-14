// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"

#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace gr {

KerrSchildCoords::KerrSchildCoords(const double bh_mass,
                                   const double bh_dimless_spin) noexcept
    : spin_a_(bh_mass * bh_dimless_spin) {
  ASSERT(bh_mass > 0,
         "The mass must be positive. The value given was " << bh_mass << ".");
  ASSERT(bh_dimless_spin > -1.0 and bh_dimless_spin < 1.0,
         "The dimensionless spin must be in the range (-1, 1). The value given "
         "was "
             << bh_dimless_spin << ".");
}

void KerrSchildCoords::pup(PUP::er& p) noexcept { p | spin_a_; }

template <typename DataType>
tnsr::Ij<DataType, 3, Frame::NoFrame> KerrSchildCoords::jacobian_matrix(
    const DataType& x, const DataType& y, const DataType& z) const noexcept {
  auto result = make_with_value<tnsr::Ij<DataType, 3, Frame::NoFrame>>(x, 0.0);

  const double a_squared = square(spin_a_);
  DataType prefactor = 0.5 * (square(x) + square(y) + square(z) - a_squared);
  prefactor += sqrt(square(prefactor) + a_squared * square(z));
  const DataType r = sqrt(prefactor);
  prefactor += a_squared;
  const DataType prefactor_dth = sqrt(prefactor / (square(x) + square(y))) / r;
  prefactor = 1.0 / prefactor;

  get<0, 0>(result) = prefactor * (x * r + spin_a_ * y);
  get<0, 1>(result) = z * prefactor_dth * x;
  get<0, 2>(result) = -y;
  get<1, 0>(result) = prefactor * (y * r - spin_a_ * x);
  get<1, 1>(result) = z * prefactor_dth * y;
  get<1, 2>(result) = x;
  get<2, 0>(result) = z / r;
  get<2, 1>(result) = -1.0 / prefactor_dth;
  // get<2, 2>(result) vanishes identically

  return result;
}

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial>
KerrSchildCoords::cartesian_from_spherical_ks(
    const tnsr::I<DataType, 3, Frame::NoFrame>& spatial_vector,
    const tnsr::I<DataType, 3, Frame::Inertial>& cartesian_coords) const
    noexcept {
  auto result = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(
      cartesian_coords, 0.0);

  for (size_t s = 0; s < get_size(get<0>(cartesian_coords)); ++s) {
    const double& x = get_element(get<0>(cartesian_coords), s);
    const double& y = get_element(get<1>(cartesian_coords), s);
    const double& z = get_element(get<2>(cartesian_coords), s);

    // For a point off the z-axis, transforming with Jacobian is well defined.
    if (LIKELY(not(equal_within_roundoff(x, 0.0) and
                   equal_within_roundoff(y, 0.0)))) {
      const auto jacobian = jacobian_matrix(x, y, z);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          get_element(result.get(i), s) +=
              jacobian.get(i, j) * get_element(spatial_vector.get(j), s);
        }
      }
    } else {
      // On the z-axis, the Jacobian provides the multivalued transformation
      //
      // v^x = (+-)(r cos\phi + a sin\phi) v^\theta
      // v^y = (+-)(r sin\phi - a cos\phi) v^\theta
      // v^z = (+-) v^r
      //
      // where +/- accounts for points z > 0 or z < 0, respectively.
      // A well defined transformation is possible if v^\theta vanishes at the
      // z-axis, in which case the transformed vector points along the z-axis.
      ASSERT(equal_within_roundoff(get_element(get<1>(spatial_vector), s), 0.0),
             "The input vector must have a vanishing theta component on the "
             "z-axis in order to perform a single-valued transformation. The "
             "vector passed has v^theta = "
                 << get_element(get<1>(spatial_vector), s) << " at z =  " << z
                 << " on the z-axis.");
      get_element(get<2>(result), s) =
          ((z > 0.0) ? get_element(get<0>(spatial_vector), s)
                     : -1.0 * get_element(get<0>(spatial_vector), s));
    }
  }
  return result;
}

template <typename DataType>
Scalar<DataType> KerrSchildCoords::r_coord_squared(
    const tnsr::I<DataType, 3, Frame::Inertial>& cartesian_coords) const
    noexcept {
  const double a_squared = square(spin_a_);
  const DataType temp =
      0.5 * (get(dot_product(cartesian_coords, cartesian_coords)) - a_squared);
  return Scalar<DataType>{
      temp + sqrt(square(temp) + a_squared * square(get<2>(cartesian_coords)))};
}

bool operator==(const KerrSchildCoords& lhs,
                const KerrSchildCoords& rhs) noexcept {
  return lhs.spin_a_ == rhs.spin_a_;
}

bool operator!=(const KerrSchildCoords& lhs,
                const KerrSchildCoords& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template Scalar<DTYPE(data)> KerrSchildCoords::r_coord_squared(       \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& cartesian_coords) \
      const noexcept;                                                   \
  template tnsr::I<DTYPE(data), 3, Frame::Inertial>                     \
  KerrSchildCoords::cartesian_from_spherical_ks(                        \
      const tnsr::I<DTYPE(data), 3, Frame::NoFrame>& spatial_vector,    \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& cartesian_coords) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace gr
/// \endcond
