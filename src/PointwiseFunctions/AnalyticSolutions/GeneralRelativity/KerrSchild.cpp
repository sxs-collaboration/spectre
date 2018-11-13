// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
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

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace gr {
namespace Solutions {

namespace {
template <typename DataType>
struct KerrSchildBuffer;

template <>
struct KerrSchildBuffer<double> {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit KerrSchildBuffer(const size_t /*size*/) noexcept {}

  tnsr::I<double, 3> x_minus_center;
  double a_dot_x;
  double a_dot_x_squared;
  double half_xsq_minus_asq;
  double r_squared;
  double a_dot_x_over_rsquared;
  double deriv_log_r_denom;
  tnsr::i<double, 3> deriv_log_r;
  double H_denom;
  double temp1;
  double temp2;
  tnsr::i<double, 3> a_cross_x;
  double denom;
  double r;

  double H;
  tnsr::i<double, 3> deriv_H;
  tnsr::i<double, 3> null_form;
  tnsr::ij<double, 3> deriv_null_form;
  double null_vector_0 = -1.0;
  double lapse_squared;
};

template <>
struct KerrSchildBuffer<DataVector> {
  private:
   // We make one giant allocation so that we don't thrash the heap. This is
   // important if we call Kerr-Schild frequently such as when applying analytic
   // boundary conditions.
   Variables<tmpl::list<
       ::Tags::TempI<0, 3>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
       ::Tags::TempScalar<3>, ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
       ::Tags::TempScalar<6>, ::Tags::Tempi<7, 3>, ::Tags::TempScalar<8>,
       ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::TempScalar<11>,
       ::Tags::Tempi<12, 3>, ::Tags::TempScalar<13>, ::Tags::TempScalar<14>,
       ::Tags::Tempi<15, 3>, ::Tags::Tempi<16, 3>, ::Tags::Tempij<17, 3>,
       ::Tags::TempScalar<18>>>
       temp_buffer_;

  public:
   explicit KerrSchildBuffer(const size_t size) noexcept
       : temp_buffer_(size),
         x_minus_center(get<::Tags::TempI<0, 3>>(temp_buffer_)),
         a_dot_x(get(get<::Tags::TempScalar<1>>(temp_buffer_))),
         a_dot_x_squared(get(get<::Tags::TempScalar<2>>(temp_buffer_))),
         half_xsq_minus_asq(get(get<::Tags::TempScalar<3>>(temp_buffer_))),
         r_squared(get(get<::Tags::TempScalar<4>>(temp_buffer_))),
         a_dot_x_over_rsquared(get(get<::Tags::TempScalar<5>>(temp_buffer_))),
         deriv_log_r_denom(get(get<::Tags::TempScalar<6>>(temp_buffer_))),
         deriv_log_r(get<::Tags::Tempi<7, 3>>(temp_buffer_)),
         H(get(get<::Tags::TempScalar<9>>(temp_buffer_))),
         H_denom(get(get<::Tags::TempScalar<8>>(temp_buffer_))),
         temp1(get(get<::Tags::TempScalar<10>>(temp_buffer_))),
         temp2(get(get<::Tags::TempScalar<11>>(temp_buffer_))),
         a_cross_x(get<::Tags::Tempi<12, 3>>(temp_buffer_)),
         denom(get(get<::Tags::TempScalar<13>>(temp_buffer_))),
         r(get(get<::Tags::TempScalar<14>>(temp_buffer_))),
         deriv_H(get<::Tags::Tempi<15, 3>>(temp_buffer_)),
         null_form(get<::Tags::Tempi<16, 3>>(temp_buffer_)),
         deriv_null_form(get<::Tags::Tempij<17, 3>>(temp_buffer_)),
         lapse_squared(get(get<::Tags::TempScalar<18>>(temp_buffer_))) {}

   tnsr::I<DataVector, 3>& x_minus_center;
   DataVector& a_dot_x;
   DataVector& a_dot_x_squared;
   DataVector& half_xsq_minus_asq;
   DataVector& r_squared;
   DataVector& a_dot_x_over_rsquared;
   DataVector& deriv_log_r_denom;
   tnsr::i<DataVector, 3>& deriv_log_r;
   DataVector& H;
   DataVector& H_denom;
   DataVector& temp1;
   DataVector& temp2;
   tnsr::i<DataVector, 3>& a_cross_x;
   DataVector& denom;
   DataVector& r;
   tnsr::i<DataVector, 3>& deriv_H;
   tnsr::i<DataVector, 3>& null_form;
   tnsr::ij<DataVector, 3>& deriv_null_form;
   double null_vector_0 = -1.0;
   DataVector& lapse_squared;
};
}  // namespace

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

  KerrSchildBuffer<DataType> buffer(get_size(get<0>(x)));

  buffer.x_minus_center = x;
  for (size_t d = 0; d < 3; ++d) {
    buffer.x_minus_center.get(d) -= gsl::at(center_, d);
  }

  buffer.a_dot_x = spin_a[0] * get<0>(buffer.x_minus_center) +
                   spin_a[1] * get<1>(buffer.x_minus_center) +
                   spin_a[2] * get<2>(buffer.x_minus_center);
  buffer.a_dot_x_squared = square(buffer.a_dot_x);
  buffer.half_xsq_minus_asq =
      0.5 * (square(get<0>(buffer.x_minus_center)) +
             square(get<1>(buffer.x_minus_center)) +
             square(get<2>(buffer.x_minus_center)) - a_squared);

  buffer.r_squared =
      buffer.half_xsq_minus_asq +
      sqrt(square(buffer.half_xsq_minus_asq) + buffer.a_dot_x_squared);
  buffer.a_dot_x_over_rsquared = buffer.a_dot_x / buffer.r_squared;

  buffer.deriv_log_r_denom =
      0.5 / (buffer.r_squared - buffer.half_xsq_minus_asq);

  for (size_t i = 0; i < 3; ++i) {
    buffer.deriv_log_r.get(i) =
        buffer.deriv_log_r_denom *
        (buffer.x_minus_center.get(i) +
         gsl::at(spin_a, i) * buffer.a_dot_x_over_rsquared);
  }

  buffer.H_denom = 1.0 / (square(buffer.r_squared) + buffer.a_dot_x_squared);
  buffer.H = mass_ * sqrt(buffer.r_squared) * buffer.r_squared * buffer.H_denom;

  buffer.temp1 =
      buffer.H * (3.0 - 4.0 * square(buffer.r_squared) * buffer.H_denom);
  buffer.temp2 = buffer.H * (2.0 * buffer.H_denom * buffer.a_dot_x);
  for (size_t i = 0; i < 3; ++i) {
    buffer.deriv_H.get(i) = buffer.temp1 * buffer.deriv_log_r.get(i) -
                            buffer.temp2 * gsl::at(spin_a, i);
  }

  get<0>(buffer.a_cross_x) = spin_a[1] * get<2>(buffer.x_minus_center) -
                             spin_a[2] * get<1>(buffer.x_minus_center);
  get<1>(buffer.a_cross_x) = spin_a[2] * get<0>(buffer.x_minus_center) -
                             spin_a[0] * get<2>(buffer.x_minus_center);
  get<2>(buffer.a_cross_x) = spin_a[0] * get<1>(buffer.x_minus_center) -
                             spin_a[1] * get<0>(buffer.x_minus_center);

  buffer.denom = 1.0 / (buffer.r_squared + a_squared);
  buffer.r = sqrt(buffer.r_squared);

  buffer.temp1 = buffer.a_dot_x / buffer.r;
  for (size_t i = 0; i < 3; ++i) {
    buffer.null_form.get(i) =
        buffer.denom *
        (buffer.r * buffer.x_minus_center.get(i) - buffer.a_cross_x.get(i) +
         buffer.temp1 * gsl::at(spin_a, i));
  }

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      buffer.deriv_null_form.get(j, i) =
          buffer.denom * (gsl::at(spin_a, i) * gsl::at(spin_a, j) / buffer.r +
                          (buffer.x_minus_center.get(i) -
                           2.0 * buffer.r * buffer.null_form.get(i) -
                           buffer.a_dot_x_over_rsquared * gsl::at(spin_a, i)) *
                              buffer.deriv_log_r.get(j) * buffer.r);
      if (i == j) {
        buffer.deriv_null_form.get(j, i) += buffer.denom * buffer.r;
      } else {  //  add denom*epsilon^ijk a_k
        size_t k = (j + 1) % 3;
        if (k == i) {  // j+1 = i (cyclic), so choose minus sign
          k++;
          k = k % 3;  // and set k to be neither i nor j
          buffer.deriv_null_form.get(j, i) -= buffer.denom * gsl::at(spin_a, k);
        } else {  // i+1 = j (cyclic), so choose plus sign
          buffer.deriv_null_form.get(j, i) += buffer.denom * gsl::at(spin_a, k);
        }
      }
    }
  }

  // Here null_vector_0 is simply -1, but if you have a boosted solution,
  // then null_vector_0 can be something different, so we leave it coded
  // in instead of eliminating it.
  buffer.null_vector_0 = -1.0;
  buffer.lapse_squared =
      1.0 / (1.0 + 2.0 * buffer.H * square(buffer.null_vector_0));

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

  get(get<gr::Tags::Lapse<DataType>>(result)) = sqrt(buffer.lapse_squared);

  {
    buffer.temp1 = -square(buffer.null_vector_0) *
                   get(get<gr::Tags::Lapse<DataType>>(result)) *
                   buffer.lapse_squared;
    for (size_t i = 0; i < 3; ++i) {
      get<DerivLapse<DataType>>(result).get(i) =
          buffer.temp1 * buffer.deriv_H.get(i);
    }
  }

  {
    buffer.temp1 =
        -2.0 * buffer.H * buffer.null_vector_0 * buffer.lapse_squared;
    for (size_t i = 0; i < 3; ++i) {
      get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(result).get(i) =
          buffer.temp1 * buffer.null_form.get(i);
    }
  }

  for (size_t m = 0; m < 3; ++m) {
    for (size_t i = 0; i < 3; ++i) {
      get<DerivShift<DataType>>(result).get(m, i) =
          4.0 * buffer.H * buffer.null_form.get(i) *
              square(buffer.lapse_squared) * cube(buffer.null_vector_0) *
              buffer.deriv_H.get(m) -
          2.0 * buffer.lapse_squared * buffer.null_vector_0 *
              (buffer.null_form.get(i) * buffer.deriv_H.get(m) +
               buffer.H * buffer.deriv_null_form.get(m, i));
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(result).get(
        i, i) = 1.;
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(result).get(
          i, j) +=
          2.0 * buffer.H * buffer.null_form.get(i) * buffer.null_form.get(j);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      for (size_t m = 0; m < 3; ++m) {
        get<DerivSpatialMetric<DataType>>(result).get(m, i, j) =
            2.0 * buffer.null_form.get(i) * buffer.null_form.get(j) *
                buffer.deriv_H.get(m) +
            2.0 * buffer.H *
                (buffer.null_form.get(i) * buffer.deriv_null_form.get(m, j) +
                 buffer.null_form.get(j) * buffer.deriv_null_form.get(m, i));
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
