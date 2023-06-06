// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"               // IWYU pragma: keep
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

// Find the alpha parameter that corresponds to the weak-field solution,
// since this is the solution we get when we set \psi = 1 initially
double compute_alpha(const double density, const double radius) {
  const double alpha_source = sqrt(2. * M_PI * density / 3.) * radius;
  return RootFinder::toms748(
      [alpha_source](const double a) {
        const double a_square = pow<2>(a);
        const double pow_2_one_plus_a_square = pow<2>(1. + a_square);
        return alpha_source - a_square * pow<3>(a) /
                                  (pow_2_one_plus_a_square * (1. + a_square));
      },
      sqrt(5.), 1.0 / alpha_source,
      // Choose a precision of 14 base-10 digits for no particular reason
      1.0e-14, 1.0e-15);
}

template <typename DataType>
Scalar<DataType> compute_piecewise(const Scalar<DataType>& r, double radius,
                                   double inner_value, double outer_value);
template <>
Scalar<double> compute_piecewise(const Scalar<double>& r, const double radius,
                                 const double inner_value,
                                 const double outer_value) {
  return Scalar<double>(get(r) < radius ? inner_value : outer_value);
}
template <>
Scalar<DataVector> compute_piecewise(const Scalar<DataVector>& r,
                                     const double radius,
                                     const double inner_value,
                                     const double outer_value) {
  return Scalar<DataVector>(inner_value - (inner_value - outer_value) *
                                              step_function(get(r) - radius));
}

}  // namespace

namespace Xcts::Solutions {

ConstantDensityStar::ConstantDensityStar(const double density,
                                         const double radius,
                                         const Options::Context& context)
    : density_(density), radius_(radius) {
  const double critical_density =
      3. * pow<5>(5.) / (2. * pow<6>(6.) * M_PI * pow<2>(radius));
  if (density <= critical_density) {
    alpha_ = compute_alpha(density, radius);
  } else {
    PARSE_ERROR(context,
                "A ConstantDensityStar has no solutions for a density below "
                "the critical density ("
                    << critical_density << ").");
  }
}

void ConstantDensityStar::pup(PUP::er& p) {
  elliptic::analytic_data::AnalyticSolution::pup(p);
  p | density_;
  p | radius_;
  if (p.isUnpacking()) {
    alpha_ = compute_alpha(density_, radius_);
  }
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
ConstantDensityStar::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const {
  const DataType r = get(magnitude(x));
  const double inner_prefactor =
      sqrt(alpha_ * radius_) / std::pow(2. * M_PI * density_ / 3., 0.25);
  const double alpha_times_radius_square = square(alpha_ * radius_);
  const double beta = inner_prefactor / sqrt(1. + square(alpha_)) - radius_;
  auto conformal_factor = make_with_value<Scalar<DataType>>(r, 0.);
  for (size_t i = 0; i < get_size(r); i++) {
    if (get_element(r, i) <= radius_) {
      get_element(get(conformal_factor), i) =
          inner_prefactor /
          sqrt(square(get_element(r, i)) + alpha_times_radius_square);
    } else {
      get_element(get(conformal_factor), i) = beta / get_element(r, i) + 1.;
    }
  }
  return {std::move(conformal_factor)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>
ConstantDensityStar::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<::Tags::deriv<
    Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>>
ConstantDensityStar::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>>> /*meta*/) const {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
ConstantDensityStar::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>
ConstantDensityStar::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const {
  return {compute_piecewise(magnitude(x), radius_, density_, 0.)};
}

PUP::able::PUP_ID ConstantDensityStar::my_PUP_ID = 0;  // NOLINT

bool operator==(const ConstantDensityStar& lhs,
                const ConstantDensityStar& rhs) {
  return lhs.density() == rhs.density() and lhs.radius() == rhs.radius();
}

bool operator!=(const ConstantDensityStar& lhs,
                const ConstantDensityStar& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>      \
  ConstantDensityStar::variables(                                             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                        \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>>) const;            \
  template tuples::TaggedTuple<                                               \
      ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>              \
  ConstantDensityStar::variables(                                             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                        \
      tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>)  \
      const;                                                                  \
  template tuples::TaggedTuple<                                               \
      ::Tags::Initial<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>, \
                                    tmpl::size_t<3>, Frame::Inertial>>>       \
  ConstantDensityStar::variables(                                             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                        \
      tmpl::list<::Tags::Initial<                                             \
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,             \
                        tmpl::size_t<3>, Frame::Inertial>>>) const;           \
  template tuples::TaggedTuple<                                               \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>          \
  ConstantDensityStar::variables(                                             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                        \
      tmpl::list<                                                             \
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>)     \
      const;                                                                  \
  template tuples::TaggedTuple<gr::Tags::EnergyDensity<DTYPE(data)>>          \
  ConstantDensityStar::variables(                                             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                        \
      tmpl::list<gr::Tags::EnergyDensity<DTYPE(data)>>) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts::Solutions
