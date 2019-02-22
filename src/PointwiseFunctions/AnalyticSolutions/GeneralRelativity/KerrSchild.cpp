// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"

#include <cmath> // IWYU pragma: keep
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <complex>

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace gr {
namespace Solutions {

KerrSchild::KerrSchild(const double mass,
                       KerrSchild::Spin::type dimensionless_spin,
                       KerrSchild::Center::type center,
                       const OptionContext& context)
    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      dimensionless_spin_(std::move(dimensionless_spin)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center))  // NOLINT
{
  const double spin_magnitude = magnitude(dimensionless_spin_);
  if (spin_magnitude > 1.0) {
    PARSE_ERROR(context,
                "Spin magnitude must be < 1. Given spin: "
                    << dimensionless_spin_ << " with magnitude "
                    << spin_magnitude);
  }
  if (mass_ < 0.0) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

void KerrSchild::pup(PUP::er& p) noexcept {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<KerrSchild::tags<DataType>>
KerrSchild::variables(const tnsr::I<DataType, 3>& x, const double /*t*/,
                      tags<DataType> /*meta*/) const noexcept {
  // Input spin is dimensionless spin.  But below we use `spin_a` = the
  // Kerr spin parameter `a`, which is `J/M` where `J` is the angular
  // momentum.
  const auto spin_a = dimensionless_spin_ * mass_;

  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  TempBuffer<tmpl::list<
      ::Tags::TempI<0, 3, Frame::Inertial, DataType>,
      ::Tags::TempScalar<1, DataType>, ::Tags::TempScalar<2, DataType>,
      ::Tags::TempScalar<3, DataType>, ::Tags::TempScalar<4, DataType>,
      ::Tags::TempScalar<5, DataType>, ::Tags::TempScalar<6, DataType>,
      ::Tags::Tempi<7, 3, Frame::Inertial, DataType>,
      ::Tags::TempScalar<8, DataType>, ::Tags::TempScalar<9, DataType>,
      ::Tags::TempScalar<10, DataType>, ::Tags::TempScalar<11, DataType>,
      ::Tags::Tempi<12, 3, Frame::Inertial, DataType>,
      ::Tags::TempScalar<13, DataType>, ::Tags::TempScalar<14, DataType>,
      ::Tags::Tempi<15, 3, Frame::Inertial, DataType>,
      ::Tags::Tempi<16, 3, Frame::Inertial, DataType>,
      ::Tags::Tempij<17, 3, Frame::Inertial, DataType>,
      ::Tags::TempScalar<18, DataType>>>
      buffer(get_size(get<0>(x)));

  // Assign understandable names to parts of the buffer.
  auto& x_minus_center =
      get<::Tags::TempI<0, 3, Frame::Inertial, DataType>>(buffer);
  auto& a_dot_x =
      get(get<::Tags::TempScalar<1, DataType>>(buffer));
  auto& a_dot_x_squared =
      get(get<::Tags::TempScalar<2, DataType>>(buffer));
  auto& half_xsq_minus_asq =
      get(get<::Tags::TempScalar<3, DataType>>(buffer));
  auto& r_squared =
      get(get<::Tags::TempScalar<4, DataType>>(buffer));
  auto& a_dot_x_over_rsquared =
      get(get<::Tags::TempScalar<5, DataType>>(buffer));
  auto& deriv_log_r_denom =
      get(get<::Tags::TempScalar<6, DataType>>(buffer));
  auto& deriv_log_r =
      get<::Tags::Tempi<7, 3, Frame::Inertial, DataType>>(buffer);
  auto& H = get(get<::Tags::TempScalar<9, DataType>>(buffer));
  auto& H_denom =
      get(get<::Tags::TempScalar<8, DataType>>(buffer));
  auto& temp1 =
      get(get<::Tags::TempScalar<10, DataType>>(buffer));
  auto& temp2 =
      get(get<::Tags::TempScalar<11, DataType>>(buffer));
  auto& a_cross_x =
      get<::Tags::Tempi<12, 3, Frame::Inertial, DataType>>(buffer);
  auto& denom =
      get(get<::Tags::TempScalar<13, DataType>>(buffer));
  auto& r = get(get<::Tags::TempScalar<14, DataType>>(buffer));
  auto& deriv_H = get<::Tags::Tempi<15, 3, Frame::Inertial, DataType>>(buffer);
  auto& null_form =
      get<::Tags::Tempi<16, 3, Frame::Inertial, DataType>>(buffer);
  auto& deriv_null_form =
      get<::Tags::Tempij<17, 3, Frame::Inertial, DataType>>(buffer);
  auto& lapse_squared =
      get(get<::Tags::TempScalar<18, DataType>>(buffer));

  x_minus_center = x;
  for (size_t d = 0; d < 3; ++d) {
    x_minus_center.get(d) -= gsl::at(center_, d);
  }

  a_dot_x = spin_a[0] * get<0>(x_minus_center) +
                   spin_a[1] * get<1>(x_minus_center) +
                   spin_a[2] * get<2>(x_minus_center);
  a_dot_x_squared = square(a_dot_x);
  half_xsq_minus_asq =
      0.5 * (square(get<0>(x_minus_center)) +
             square(get<1>(x_minus_center)) +
             square(get<2>(x_minus_center)) - a_squared);

  r_squared =
      half_xsq_minus_asq +
      sqrt(square(half_xsq_minus_asq) + a_dot_x_squared);
  a_dot_x_over_rsquared = a_dot_x / r_squared;

  deriv_log_r_denom =
      0.5 / (r_squared - half_xsq_minus_asq);

  for (size_t i = 0; i < 3; ++i) {
    deriv_log_r.get(i) =
        deriv_log_r_denom *
        (x_minus_center.get(i) +
         gsl::at(spin_a, i) * a_dot_x_over_rsquared);
  }

  H_denom = 1.0 / (square(r_squared) + a_dot_x_squared);
  H = mass_ * sqrt(r_squared) * r_squared * H_denom;

  temp1 =
      H * (3.0 - 4.0 * square(r_squared) * H_denom);
  temp2 = H * (2.0 * H_denom * a_dot_x);
  for (size_t i = 0; i < 3; ++i) {
    deriv_H.get(i) = temp1 * deriv_log_r.get(i) -
                            temp2 * gsl::at(spin_a, i);
  }

  get<0>(a_cross_x) = spin_a[1] * get<2>(x_minus_center) -
                             spin_a[2] * get<1>(x_minus_center);
  get<1>(a_cross_x) = spin_a[2] * get<0>(x_minus_center) -
                             spin_a[0] * get<2>(x_minus_center);
  get<2>(a_cross_x) = spin_a[0] * get<1>(x_minus_center) -
                             spin_a[1] * get<0>(x_minus_center);

  denom = 1.0 / (r_squared + a_squared);
  r = sqrt(r_squared);

  temp1 = a_dot_x / r;
  for (size_t i = 0; i < 3; ++i) {
    null_form.get(i) =
        denom *
        (r * x_minus_center.get(i) - a_cross_x.get(i) +
         temp1 * gsl::at(spin_a, i));
  }

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      deriv_null_form.get(j, i) =
          denom * (gsl::at(spin_a, i) * gsl::at(spin_a, j) / r +
                          (x_minus_center.get(i) -
                           2.0 * r * null_form.get(i) -
                           a_dot_x_over_rsquared * gsl::at(spin_a, i)) *
                              deriv_log_r.get(j) * r);
      if (i == j) {
        deriv_null_form.get(j, i) += denom * r;
      } else {  //  add denom*epsilon^ijk a_k
        size_t k = (j + 1) % 3;
        if (k == i) {  // j+1 = i (cyclic), so choose minus sign
          k++;
          k = k % 3;  // and set k to be neither i nor j
          deriv_null_form.get(j, i) -= denom * gsl::at(spin_a, k);
        } else {  // i+1 = j (cyclic), so choose plus sign
          deriv_null_form.get(j, i) += denom * gsl::at(spin_a, k);
        }
      }
    }
  }

  // Here null_vector_0 is simply -1, but if you have a boosted solution,
  // then null_vector_0 can be something different, so we leave it coded
  // in instead of eliminating it.
  double null_vector_0 = -1.0;
  lapse_squared =
      1.0 / (1.0 + 2.0 * H * square(null_vector_0));

  // Need the following quantities we computed above to construct the
  // Kerr-Schild solution:
  // - lapse_squared
  // - H
  // - deriv_H
  // - null_vector_0
  // - null_form
  // - deriv_null_form

  auto result =
      make_with_value<tuples::tagged_tuple_from_typelist<tags<DataType>>>(x,
                                                                          0.0);

  get(get<gr::Tags::Lapse<DataType>>(result)) = sqrt(lapse_squared);

  {
    temp1 = -square(null_vector_0) *
                   get(get<gr::Tags::Lapse<DataType>>(result)) *
                   lapse_squared;
    for (size_t i = 0; i < 3; ++i) {
      get<DerivLapse<DataType>>(result).get(i) =
          temp1 * deriv_H.get(i);
    }
  }

  {
    temp1 =
        -2.0 * H * null_vector_0 * lapse_squared;
    for (size_t i = 0; i < 3; ++i) {
      get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(result).get(i) =
          temp1 * null_form.get(i);
    }
  }

  for (size_t m = 0; m < 3; ++m) {
    for (size_t i = 0; i < 3; ++i) {
      get<DerivShift<DataType>>(result).get(m, i) =
          4.0 * H * null_form.get(i) *
              square(lapse_squared) * cube(null_vector_0) *
              deriv_H.get(m) -
          2.0 * lapse_squared * null_vector_0 *
              (null_form.get(i) * deriv_H.get(m) +
               H * deriv_null_form.get(m, i));
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(result).get(
        i, i) = 1.;
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(result).get(
          i, j) +=
          2.0 * H * null_form.get(i) * null_form.get(j);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      for (size_t m = 0; m < 3; ++m) {
        get<DerivSpatialMetric<DataType>>(result).get(m, i, j) =
            2.0 * null_form.get(i) * null_form.get(j) *
                deriv_H.get(m) +
            2.0 * H *
                (null_form.get(i) * deriv_null_form.get(m, j) +
                 null_form.get(j) * deriv_null_form.get(m, i));
      }
    }
  }

  auto det_and_inverse = determinant_and_inverse(
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(result));

  get(get<gr::Tags::SqrtDetSpatialMetric<DataType>>(result)) =
      sqrt(get(det_and_inverse.first));

  get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>>(result) =
      det_and_inverse.second;

  get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>>(result) =
      gr::extrinsic_curvature(
          get<gr::Tags::Lapse<DataType>>(result),
          get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(result),
          get<DerivShift<DataType>>(result),
          get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(result),
          get<::Tags::dt<
              gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>>(result),
          get<DerivSpatialMetric<DataType>>(result));

  return result;
}
}  // namespace Solutions
}  // namespace gr

template tuples::tagged_tuple_from_typelist<
    gr::Solutions::KerrSchild::tags<DataVector>>
gr::Solutions::KerrSchild::variables(
    const tnsr::I<DataVector, 3>& x, const double /*t*/,
    KerrSchild::tags<DataVector> /*meta*/) const noexcept;
template tuples::tagged_tuple_from_typelist<
    gr::Solutions::KerrSchild::tags<double>>
gr::Solutions::KerrSchild::variables(const tnsr::I<double, 3>& x,
                                     const double /*t*/,
                                     KerrSchild::tags<double> /*meta*/) const
    noexcept;
/// \endcond
