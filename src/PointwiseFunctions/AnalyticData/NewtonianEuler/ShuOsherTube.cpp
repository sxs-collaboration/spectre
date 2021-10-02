// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianEuler/ShuOsherTube.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

namespace NewtonianEuler::AnalyticData {
ShuOsherTube::ShuOsherTube(const double jump_position,
                           const double mass_density_l, const double velocity_l,
                           const double pressure_l, const double velocity_r,
                           const double pressure_r, const double epsilon,
                           const double lambda)
    : mass_density_l_(mass_density_l),
      velocity_l_(velocity_l),
      pressure_l_(pressure_l),
      jump_position_(jump_position),
      epsilon_(epsilon),
      lambda_(lambda),
      velocity_r_(velocity_r),
      pressure_r_(pressure_r) {
  ASSERT(mass_density_l_ > 0.0,
         "The left mass_density must be greater than 0 but is "
             << mass_density_l_);
  ASSERT(pressure_l_ > 0.0,
         "The left pressure must be greater than 0 but is " << pressure_l_);
  ASSERT(pressure_r_ > 0.0,
         "The right pressure must be greater than 0 but is " << pressure_r_);
  ASSERT(epsilon_ > 0.0,
         "The sinusoid amplitude must be greater than 0 but is " << epsilon_);
  ASSERT(epsilon_ < 1.0,
         "The sinusoid amplitude must be less than 1 but is " << epsilon_);
}

void ShuOsherTube::pup(PUP::er& p) {
  p | mass_density_l_;
  p | velocity_l_;
  p | pressure_l_;
  p | jump_position_;
  p | epsilon_;
  p | lambda_;
  p | velocity_r_;
  p | pressure_r_;
  p | adiabatic_index_;
  p | equation_of_state_;
}

bool operator==(const ShuOsherTube& lhs, const ShuOsherTube& rhs) {
  // No comparison for equation_of_state_. Comparing adiabatic_index_ should
  // suffice.
  return lhs.mass_density_l_ == rhs.mass_density_l_ and
         lhs.velocity_l_ == rhs.velocity_l_ and
         lhs.pressure_l_ == rhs.pressure_l_ and
         lhs.jump_position_ == rhs.jump_position_ and
         lhs.epsilon_ == rhs.epsilon_ and lhs.lambda_ == rhs.lambda_ and
         lhs.velocity_r_ == rhs.velocity_r_ and
         lhs.pressure_r_ == rhs.pressure_r_ and
         lhs.adiabatic_index_ == rhs.adiabatic_index_;
}

bool operator!=(const ShuOsherTube& lhs, const ShuOsherTube& rhs) {
  return not(lhs == rhs);
}

template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>> ShuOsherTube::variables(
    const tnsr::I<DataType, 1, Frame::Inertial>& x,
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/) const {
  auto mass_density = make_with_value<Scalar<DataType>>(x, 0.0);
  for (size_t s = 0; s < get_size(get<0>(x)); ++s) {
    const double x_s = get_element(get<0>(x), s);
    get_element(get(mass_density), s) =
        x_s < jump_position_ ? mass_density_l_
                             : 1.0 + epsilon_ * sin(lambda_ * x_s);
  }
  return mass_density;
}

template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, 1, Frame::Inertial>>
ShuOsherTube::variables(
    const tnsr::I<DataType, 1, Frame::Inertial>& x,
    tmpl::list<Tags::Velocity<DataType, 1, Frame::Inertial>> /*meta*/) const {
  auto velocity =
      make_with_value<tnsr::I<DataType, 1, Frame::Inertial>>(x, 0.0);
  for (size_t s = 0; s < get_size(get<0>(x)); ++s) {
    const double x_s = get_element(get<0>(x), s);
    get_element(get<0>(velocity), s) =
        x_s < jump_position_ ? velocity_l_ : velocity_r_;
  }
  return velocity;
}

template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> ShuOsherTube::variables(
    const tnsr::I<DataType, 1, Frame::Inertial>& x,
    tmpl::list<Tags::Pressure<DataType>> /*meta*/) const {
  auto pressure = make_with_value<Scalar<DataType>>(x, 0.0);
  for (size_t s = 0; s < get_size(get<0>(x)); ++s) {
    const double x_s = get_element(get<0>(x), s);
    get_element(get(pressure), s) =
        x_s < jump_position_ ? pressure_l_ : pressure_r_;
  }
  return pressure;
}

template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
ShuOsherTube::variables(
    const tnsr::I<DataType, 1, Frame::Inertial>& x,
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<Tags::MassDensity<DataType>>(
          variables(x, tmpl::list<Tags::MassDensity<DataType>>{})),
      get<Tags::Pressure<DataType>>(
          variables(x, tmpl::list<Tags::Pressure<DataType>>{})));
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_SCALARS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)> >                    \
      ShuOsherTube::variables(                                               \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x_shifted, \
          tmpl::list<TAG(data) < DTYPE(data)> >) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (1), (double, DataVector),
                        (Tags::MassDensity, Tags::Pressure,
                         Tags::SpecificInternalEnergy))

#define INSTANTIATE_VELOCITY(_, data)                                        \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data),           \
                               Frame::Inertial> >                            \
      ShuOsherTube::variables(                                               \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x_shifted, \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data), Frame::Inertial> >) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (1), (double, DataVector),
                        (Tags::Velocity))

#undef DIM
#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VELOCITY
}  // namespace NewtonianEuler::AnalyticData
