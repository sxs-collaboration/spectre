// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/KerrSchild.hpp"

#include <numeric>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "Utilities/Gsl.hpp"

namespace EinsteinSolutions {

KerrSchild::KerrSchild(const double mass, KerrSchild::Spin::type spin,
                       KerrSchild::Center::type center,
                       const OptionContext& context)
    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      spin_(std::move(spin)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center))  // NOLINT
{
  const double spin_magnitude = magnitude(spin_);
  if (spin_magnitude > 1.0) {
    PARSE_ERROR(context,
                "Spin magnitude must be < 1. Given spin: "
                    << spin_ << " with magnitude " << spin_magnitude);
  }
  if (mass_ < 0.0) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

void KerrSchild::pup(PUP::er& p) noexcept {
  p | mass_;
  p | spin_;
  p | center_;
}

template <typename DataType>
tuples::TaggedTupleTypelist<KerrSchild::tags<DataType>> KerrSchild::solution(
    const tnsr::I<DataType, 3>& x, const double /*t*/) const noexcept {
  const auto a_squared =
      std::inner_product(spin_.begin(), spin_.end(), spin_.begin(), 0.);

  const auto x_minus_center = [&x, this ]() noexcept {
    auto l_x_minus_center = x;
    for (size_t d = 0; d < 3; ++d) {
      l_x_minus_center.get(d) -= gsl::at(center_, d);
    }
    return l_x_minus_center;
  }
  ();

  const DataType a_dot_x = spin_[0] * get<0>(x_minus_center) +
                           spin_[1] * get<1>(x_minus_center) +
                           spin_[2] * get<2>(x_minus_center);
  const DataType a_dot_x_squared = square(a_dot_x);
  const DataType half_xsq_minus_asq =
      0.5 * (square(get<0>(x_minus_center)) + square(get<1>(x_minus_center)) +
             square(get<2>(x_minus_center)) - a_squared);
  const DataType r_squared =
      half_xsq_minus_asq + sqrt(square(half_xsq_minus_asq) + a_dot_x_squared);
  const DataType a_dot_x_over_rsquared = a_dot_x / r_squared;

  const DataType deriv_log_r_denom = 0.5 / (r_squared - half_xsq_minus_asq);

  const auto deriv_log_r = [
    &deriv_log_r_denom, &x_minus_center, &a_dot_x_over_rsquared, this
  ]() noexcept {
    auto l_deriv_log_r =
        make_with_value<tnsr::i<DataType, 3>>(x_minus_center, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      l_deriv_log_r.get(i) =
          deriv_log_r_denom *
          (x_minus_center.get(i) + gsl::at(spin_, i) * a_dot_x_over_rsquared);
    }
    return l_deriv_log_r;
  }
  ();

  const DataType H_denom = 1.0 / (square(r_squared) + a_dot_x_squared);
  const DataType H = mass_ * sqrt(r_squared) * r_squared * H_denom;

  const auto deriv_H =
      [&H, &r_squared, &H_denom, &a_dot_x, &deriv_log_r, this ]() noexcept {
    auto l_deriv_H = make_with_value<tnsr::i<DataType, 3>>(H_denom, 0.0);
    const DataType temp1 = H * (3.0 - 4.0 * square(r_squared) * H_denom);
    const DataType temp2 = H * (2.0 * H_denom * a_dot_x);
    for (size_t i = 0; i < 3; ++i) {
      l_deriv_H.get(i) = temp1 * deriv_log_r.get(i) - temp2 * gsl::at(spin_, i);
    }
    return l_deriv_H;
  }
  ();

  const auto a_cross_x = [](const std::array<double, 3>& a,
                            const tnsr::I<DataType, 3>& coord) noexcept {
    auto l_a_cross_x = make_with_value<tnsr::i<DataType, 3>>(coord, 0.0);
    get<0>(l_a_cross_x) = a[1] * get<2>(coord) - a[2] * get<1>(coord);
    get<1>(l_a_cross_x) = a[2] * get<0>(coord) - a[0] * get<2>(coord);
    get<2>(l_a_cross_x) = a[0] * get<1>(coord) - a[1] * get<0>(coord);
    return l_a_cross_x;
  }
  (spin_, x_minus_center);

  const DataType denom = 1.0 / (r_squared + a_squared);
  const DataType r = sqrt(r_squared);

  const auto null_form = [
    &x, &a_dot_x, &r, &denom, &x_minus_center, &a_cross_x, this
  ]() noexcept {
    auto l_null_form = make_with_value<tnsr::i<DataType, 3>>(x, 0.0);
    const DataType temp = a_dot_x / r;
    for (size_t i = 0; i < 3; ++i) {
      l_null_form.get(i) =
          denom * (r * x_minus_center.get(i) - a_cross_x.get(i) +
                   temp * gsl::at(spin_, i));
    }
    return l_null_form;
  }
  ();

  const auto deriv_null_form = [
    &denom, &x_minus_center, &r, &null_form, &a_dot_x_over_rsquared,
    &deriv_log_r, this
  ]() noexcept {
    auto l_deriv_null_form = make_with_value<tnsr::ij<DataType, 3>>(r, 0.0);
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        l_deriv_null_form.get(j, i) =
            denom * (gsl::at(spin_, i) * gsl::at(spin_, j) / r +
                     (x_minus_center.get(i) - 2.0 * r * null_form.get(i) -
                      a_dot_x_over_rsquared * gsl::at(spin_, i)) *
                         deriv_log_r.get(j) * r);
        if (i == j) {
          l_deriv_null_form.get(j, i) += denom * r;
        } else {  //  add denom*epsilon^ijk a_k
          size_t k = (j + 1) % 3;
          if (k == i) {  // j+1 = i (cyclic), so choose minus sign
            k++;
            k = k % 3;  // and set k to be neither i nor j
            l_deriv_null_form.get(j, i) -= denom * gsl::at(spin_, k);
          } else {  // i+1 = j (cyclic), so choose plus sign
            l_deriv_null_form.get(j, i) += denom * gsl::at(spin_, k);
          }
        }
      }
    }
    return l_deriv_null_form;
  }
  ();

  // Here null_vector_0 is simply -1, but if you have a boosted solution,
  // then null_vector_0 can be something different, so we leave it coded
  // in instead of eliminating it.
  const constexpr double null_vector_0 = -1.0;
  const DataType lapse_squared = 1.0 / (1.0 + 2.0 * H * square(null_vector_0));

  auto result = make_with_value<tuples::TaggedTuple<
      gr::Tags::Lapse<3, Frame::Inertial, DataType>,
      gr::Tags::DtLapse<3, Frame::Inertial, DataType>, deriv_lapse<DataType>,
      gr::Tags::Shift<3, Frame::Inertial, DataType>,
      gr::Tags::DtShift<3, Frame::Inertial, DataType>, deriv_shift<DataType>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
      gr::Tags::DtSpatialMetric<3, Frame::Inertial, DataType>,
      deriv_spatial_metric<DataType>>>(x, 0.0);

  get(get<gr::Tags::Lapse<3, Frame::Inertial, DataType>>(result)) =
      sqrt(lapse_squared);

  {
    const DataType temp =
        -square(null_vector_0) *
        get(get<gr::Tags::Lapse<3, Frame::Inertial, DataType>>(result)) *
        lapse_squared;
    for (size_t i = 0; i < 3; ++i) {
      get<deriv_lapse<DataType>>(result).get(i) = temp * deriv_H.get(i);
    }
  }

  {
    const DataType temp = -2.0 * H * null_vector_0 * lapse_squared;
    for (size_t i = 0; i < 3; ++i) {
      get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(result).get(i) =
          temp * null_form.get(i);
    }
  }

  for (size_t m = 0; m < 3; ++m) {
    for (size_t i = 0; i < 3; ++i) {
      get<deriv_shift<DataType>>(result).get(m, i) =
          4.0 * H * null_form.get(i) * square(lapse_squared) *
              cube(null_vector_0) * deriv_H.get(m) -
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
          i, j) += 2.0 * H * null_form.get(i) * null_form.get(j);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      for (size_t m = 0; m < 3; ++m) {
        get<deriv_spatial_metric<DataType>>(result).get(m, i, j) =
            2.0 * null_form.get(i) * null_form.get(j) * deriv_H.get(m) +
            2.0 * H * (null_form.get(i) * deriv_null_form.get(m, j) +
                       null_form.get(j) * deriv_null_form.get(m, i));
      }
    }
  }

  return result;
}
}  // namespace EinsteinSolutions

template tuples::TaggedTupleTypelist<
    EinsteinSolutions::KerrSchild::tags<DataVector>>
EinsteinSolutions::KerrSchild::solution(const tnsr::I<DataVector, 3>& x,
                                        const double /*t*/) const noexcept;
template tuples::TaggedTupleTypelist<
    EinsteinSolutions::KerrSchild::tags<double>>
EinsteinSolutions::KerrSchild::solution(const tnsr::I<double, 3>& x,
                                        const double /*t*/) const noexcept;
