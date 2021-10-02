// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianEuler/KhInstability.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianEuler::AnalyticData {

template <size_t Dim>
KhInstability<Dim>::KhInstability(
    const double adiabatic_index, const double strip_bimedian_height,
    const double strip_thickness, const double strip_density,
    const double strip_velocity, const double background_density,
    const double background_velocity, const double pressure,
    const double perturbation_amplitude, const double perturbation_width)
    : adiabatic_index_(adiabatic_index),
      strip_bimedian_height_(strip_bimedian_height),
      strip_half_thickness_(0.5 * strip_thickness),
      strip_density_(strip_density),
      strip_velocity_(strip_velocity),
      background_density_(background_density),
      background_velocity_(background_velocity),
      pressure_(pressure),
      perturbation_amplitude_(perturbation_amplitude),
      perturbation_width_(perturbation_width),
      equation_of_state_(adiabatic_index) {
  ASSERT(strip_density_ > 0.0 and background_density_ > 0.0,
         "The mass density must be positive everywhere. Inner "
         "density: "
             << strip_density_ << ", Outer density: " << background_density_
             << ".");
  ASSERT(pressure_ > 0.0, "The pressure must be positive. The value given was "
                              << pressure_ << ".");
  ASSERT(perturbation_width_ > 0.0,
         "The damping factor must be positive. The value given was "
             << perturbation_width_ << ".");
  ASSERT(strip_thickness > 0.0,
         "The strip thickness must be positive. The value given was "
             << strip_thickness << ".");
}

template <size_t Dim>
void KhInstability<Dim>::pup(PUP::er& p) {
  p | adiabatic_index_;
  p | strip_bimedian_height_;
  p | strip_half_thickness_;
  p | strip_density_;
  p | strip_velocity_;
  p | background_density_;
  p | background_velocity_;
  p | pressure_;
  p | perturbation_amplitude_;
  p | perturbation_width_;
  p | equation_of_state_;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>> KhInstability<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/) const {
  auto result = make_with_value<Scalar<DataType>>(x, 0.0);
  const size_t n_pts = get_size(get<0>(x));
  for (size_t s = 0; s < n_pts; ++s) {
    get_element(get(result), s) =
        abs(get_element(get<Dim - 1>(x), s) - strip_bimedian_height_) <
                strip_half_thickness_
            ? strip_density_
            : background_density_;
  }
  return {std::move(result)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>
KhInstability<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/) const {
  auto result =
      make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(x, 0.0);

  const size_t n_pts = get_size(get<0>(x));
  for (size_t s = 0; s < n_pts; ++s) {
    get_element(get<0>(result), s) =
        abs(get_element(get<Dim - 1>(x), s) - strip_bimedian_height_) <
                strip_half_thickness_
            ? strip_velocity_
            : background_velocity_;
  }

  const double one_over_two_sigma_squared = 0.5 / square(perturbation_width_);
  const double strip_lower_bound =
      strip_bimedian_height_ - strip_half_thickness_;
  const double strip_upper_bound =
      strip_bimedian_height_ + strip_half_thickness_;
  get<Dim - 1>(result) = exp(-one_over_two_sigma_squared *
                             square(get<Dim - 1>(x) - strip_lower_bound)) +
                         exp(-one_over_two_sigma_squared *
                             square(get<Dim - 1>(x) - strip_upper_bound));
  get<Dim - 1>(result) *= perturbation_amplitude_ * sin(4.0 * M_PI * get<0>(x));

  return {std::move(result)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
KhInstability<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<Tags::MassDensity<DataType>>(
          variables(x, tmpl::list<Tags::MassDensity<DataType>>{})),
      get<Tags::Pressure<DataType>>(
          variables(x, tmpl::list<Tags::Pressure<DataType>>{})));
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> KhInstability<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x,
    tmpl::list<Tags::Pressure<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, pressure_);
}

template <size_t Dim>
bool operator==(const KhInstability<Dim>& lhs, const KhInstability<Dim>& rhs) {
  // No comparison for equation_of_state_. Comparing adiabatic_index_ should
  // suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.strip_bimedian_height_ == rhs.strip_bimedian_height_ and
         lhs.strip_half_thickness_ == rhs.strip_half_thickness_ and
         lhs.strip_density_ == rhs.strip_density_ and
         lhs.strip_velocity_ == rhs.strip_velocity_ and
         lhs.background_density_ == rhs.background_density_ and
         lhs.background_velocity_ == rhs.background_velocity_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.perturbation_width_ == rhs.perturbation_width_;
}

template <size_t Dim>
bool operator!=(const KhInstability<Dim>& lhs, const KhInstability<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                           \
  template class KhInstability<DIM(data)>;                   \
  template bool operator==(const KhInstability<DIM(data)>&,  \
                           const KhInstability<DIM(data)>&); \
  template bool operator!=(const KhInstability<DIM(data)>&,  \
                           const KhInstability<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (2, 3))

#define INSTANTIATE_SCALARS(_, data)                                 \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >            \
      KhInstability<DIM(data)>::variables(                           \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x, \
          tmpl::list<TAG(data) < DTYPE(data)> >) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (2, 3), (double, DataVector),
                        (Tags::MassDensity, Tags::SpecificInternalEnergy,
                         Tags::Pressure))

#define INSTANTIATE_VELOCITY(_, data)                                        \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),           \
                               Frame::Inertial> >                            \
      KhInstability<DIM(data)>::variables(                                   \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,         \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data), Frame::Inertial> >) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (2, 3), (double, DataVector),
                        (Tags::Velocity))

#undef INSTANTIATE_VELOCITY
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_CLASS
#undef TAG
#undef DTYPE
#undef DIM
}  // namespace NewtonianEuler::AnalyticData
