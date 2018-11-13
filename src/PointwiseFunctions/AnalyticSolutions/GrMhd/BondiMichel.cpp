// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
/// \cond
namespace grmhd {
namespace Solutions {

BondiMichel::BondiMichel(
    const Mass::type mass, const SonicRadius::type sonic_radius,
    const SonicDensity::type sonic_density,
    const PolytropicExponent::type polytropic_exponent,
    const MagFieldStrength::type mag_field_strength) noexcept
    : mass_(mass),
      sonic_radius_(sonic_radius),
      sonic_density_(sonic_density),
      polytropic_exponent_(polytropic_exponent),
      mag_field_strength_(mag_field_strength),
      // From Rezzola and Zanotti (2013)
      // Rezzola and Zanotti (2013) Eq. 11.94(a)
      sonic_fluid_speed_squared_(0.5 * mass_ / sonic_radius_),
      // Rezzola and Zanotti (2013) Eq. 11.94(b)
      sonic_sound_speed_squared_(sonic_fluid_speed_squared_ /
                                 (1.0 - 3.0 * sonic_fluid_speed_squared_)),
      // Rezzola and Zanotti (2013) Eq. 11.101
      mass_accretion_rate_over_four_pi_(square(sonic_radius_) * sonic_density *
                                        sqrt(sonic_fluid_speed_squared_)),
      background_spacetime_{mass_, std::array<double, 3>{{0.0, 0.0, 0.0}},
                            std::array<double, 3>{{0.0, 0.0, 0.0}}} {
  const double gamma_minus_one = polytropic_exponent_ - 1.0;
  // Rezzola and Zanotti (2013) Eq. 11.96
  const double sonic_newtonian_sound_speed_squared =
      gamma_minus_one * sonic_sound_speed_squared_ /
      (gamma_minus_one - sonic_sound_speed_squared_);
  polytropic_constant_ = sonic_newtonian_sound_speed_squared *
                         pow(sonic_density_, 1.0 - polytropic_exponent_) /
                         polytropic_exponent_;
  equation_of_state_ = EquationsOfState::PolytropicFluid<true>{
      polytropic_constant_, polytropic_exponent_};
  const double sonic_specific_enthalpy_squared_minus_one =
      sonic_newtonian_sound_speed_squared / gamma_minus_one;

  // Rezzola and Zanotti (2013) Eq. 11.97
  bernoulli_constant_squared_minus_one_ =
      -3.0 * sonic_fluid_speed_squared_ *
          square(1.0 + sonic_specific_enthalpy_squared_minus_one) +
      sonic_specific_enthalpy_squared_minus_one *
          (2.0 + sonic_specific_enthalpy_squared_minus_one);

  // Inverse of Rezzola and Zanotti (2013) Eq 11.99 without truncation
  const double sound_speed_at_infinity_squared =
      gamma_minus_one + (sonic_sound_speed_squared_ - gamma_minus_one) *
                            sqrt(1.0 + 3.0 * sonic_sound_speed_squared_);

  // Rezzola and Zanotti (2013) Eq. 11.102 is a good approximation
  // only for `sonic_radius` >> 2.0 * mass_, The exact form has a
  // factor of sqrt(1.0 + 3.0 * sonic_sound_speed_squared_)
  rest_mass_density_at_infinity_ =
      sonic_density_ * pow(sound_speed_at_infinity_squared /
                               (sonic_sound_speed_squared_ *
                                sqrt(1.0 + 3.0 * sonic_sound_speed_squared_)),
                           1.0 / gamma_minus_one);
}

void BondiMichel::pup(PUP::er& p) noexcept {
  p | mass_;
  p | sonic_radius_;
  p | sonic_density_;
  p | polytropic_exponent_;
  p | mag_field_strength_;
  p | sonic_fluid_speed_squared_;
  p | sonic_sound_speed_squared_;
  p | polytropic_constant_;
  p | mass_accretion_rate_over_four_pi_;
  p | bernoulli_constant_squared_minus_one_;
  p | rest_mass_density_at_infinity_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
BondiMichel::IntermediateVars<DataType>::IntermediateVars(
    const double rest_mass_density_at_infinity,
    const double in_mass_accretion_rate_over_four_pi, const double in_mass,
    const double in_polytropic_constant, const double in_polytropic_exponent,
    const double in_bernoulli_constant_squared_minus_one,
    const double in_sonic_radius, const double in_sonic_density,
    const tnsr::I<DataType, 3>& x, const bool need_spacetime,
    const gr::Solutions::KerrSchild& background_spacetime) noexcept
    : radius((magnitude(x)).get()),
      rest_mass_density(make_with_value<DataType>(x, 0.0)),
      mass_accretion_rate_over_four_pi(in_mass_accretion_rate_over_four_pi),
      mass(in_mass),
      polytropic_constant(in_polytropic_constant),
      polytropic_exponent(in_polytropic_exponent),
      bernoulli_constant_squared_minus_one(
          in_bernoulli_constant_squared_minus_one),
      sonic_radius(in_sonic_radius),
      sonic_density(in_sonic_density) {
  // NOLINTNEXTLINE(clang-analyzer-core)
  for (size_t i = 0; i < get_size(rest_mass_density); i++) {
    const double current_radius = get_element(radius, i);
    // Near the sonic radius, a second root to the Bernoulli
    // root function appears. Within the sonic radius, the
    // upper bound of
    // `mass_accretion_rate_over_four_pi_ * sqrt(2.0 /
    // (mass_* cube(current_radius)))` selects the correct one
    // of two possible roots. Beyond the sonic radius, this
    // becomes the lower bound provided to the root finder.
    const double sonic_bound = mass_accretion_rate_over_four_pi *
                               sqrt(2.0 / (mass * cube(current_radius)));
    get_element(rest_mass_density, i) =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(
            [&current_radius, this ](const double guess_for_rho) noexcept {
              return bernoulli_root_function(guess_for_rho, current_radius);
            },
            current_radius < sonic_radius ? rest_mass_density_at_infinity
                                          : sonic_bound,
            current_radius < sonic_radius ? sonic_bound : sonic_density, 1.e-15,
            1.e-15);
  }
  if (need_spacetime) {
    kerr_schild_soln = background_spacetime.variables(
        x, 0.0, gr::Solutions::KerrSchild::tags<DataType>{});
  }
}

template <typename DataType>
double BondiMichel::IntermediateVars<DataType>::bernoulli_root_function(
    const double rest_mass_density_guess, const double current_radius) const
    noexcept {
  const double gamma_minus_one = polytropic_exponent - 1.0;
  const double polytropic_index_times_newtonian_sound_speed_squared =
      (polytropic_exponent * polytropic_constant *
       pow(rest_mass_density_guess, gamma_minus_one)) /
      gamma_minus_one;
  const double specific_enthalpy_squared_minus_one =
      polytropic_index_times_newtonian_sound_speed_squared *
      (2.0 + polytropic_index_times_newtonian_sound_speed_squared);
  const double specific_enthalpy_squared =
      specific_enthalpy_squared_minus_one + 1.0;
  // As the bernoulli constant is 1.0 + a small number, it is better numerically
  // to compute it as (small_number_1) + (1.0 + small_number_1) * small_number_2
  // as opposed to (1.0 + small_number_1)^2 * (1.0 + small_number_2) - 1.0.
  // Computing it in this way allows the numerical root-finding method to work
  // even for large radii.
  return specific_enthalpy_squared_minus_one +
         specific_enthalpy_squared *
             (-2.0 * mass / current_radius +
              square(mass_accretion_rate_over_four_pi /
                     (square(current_radius) * rest_mass_density_guess))) -
         bernoulli_constant_squared_minus_one;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  return {Scalar<DataType>{DataType{vars.rest_mass_density}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> BondiMichel::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  return {Scalar<DataType>{
      DataType{polytropic_constant_ *
               pow(vars.rest_mass_density, polytropic_exponent_)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  return {Scalar<DataType>{
      DataType{polytropic_constant_ *
               pow(vars.rest_mass_density, polytropic_exponent_ - 1.0) /
               (polytropic_exponent_ - 1.0)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  return {Scalar<DataType>{DataType{
      1.0 + polytropic_exponent_ * polytropic_constant_ *
                pow(vars.rest_mass_density, polytropic_exponent_ - 1.0) /
                (polytropic_exponent_ - 1.0)}}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
    const IntermediateVars<DataType>& /*vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  // Rezzola and Zanotti (2013) Eq. 11.79
  const DataType abs_fluid_four_velocity_u_r =
      mass_accretion_rate_over_four_pi_ /
      (square(vars.radius) * vars.rest_mass_density);
  auto lorentz_factor = make_with_value<Scalar<DataType>>(x, 1.0);
  // Rezzola and Zanotti (2013) Eq. 11.87
  for (size_t i = 0; i < get_size(lorentz_factor.get()); i++) {
    double two_m_over_r = 2.0 * mass_ / get_element(vars.radius, i);
    get_element(lorentz_factor.get(), i) =
        ((1.0 + two_m_over_r) *
             square(get_element(abs_fluid_four_velocity_u_r, i)) +
         1.0) /
        ((two_m_over_r * get_element(abs_fluid_four_velocity_u_r, i) +
          sqrt(square(get_element(abs_fluid_four_velocity_u_r, i)) + 1.0 -
               two_m_over_r)) *
         (sqrt(1.0 + two_m_over_r)));
  }
  return lorentz_factor;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  // Rezzola and Zanotti (2013) Eq. 11.79
  const DataType abs_fluid_four_velocity_u_r =
      mass_accretion_rate_over_four_pi_ /
      (square(vars.radius) * vars.rest_mass_density);
  auto fluid_four_velocity_u_t = make_with_value<DataType>(x, 0.0);
  for (size_t i = 0; i < get_size(fluid_four_velocity_u_t); i++) {
    double two_m_over_r = 2.0 * mass_ / get_element(vars.radius, i);
    get_element(fluid_four_velocity_u_t, i) =
        ((1.0 + two_m_over_r) *
             square(get_element(abs_fluid_four_velocity_u_r, i)) +
         1.0) /
        (two_m_over_r * get_element(abs_fluid_four_velocity_u_r, i) +
         sqrt(square(get_element(abs_fluid_four_velocity_u_r, i)) + 1.0 -
              two_m_over_r));
  }

  // Rezzola and Zanotti (2013) Eq. 7.22
  const DataType eulerian_radial_velocity_over_radius =
      sqrt(1.0 + 2.0 * mass_ / vars.radius) *
      (-abs_fluid_four_velocity_u_r / fluid_four_velocity_u_t +
       2.0 * mass_ / (vars.radius + 2.0 * mass_)) /
      vars.radius;
  result.get(0) = eulerian_radial_velocity_over_radius * x.get(0);
  result.get(1) = eulerian_radial_velocity_over_radius * x.get(1);
  result.get(2) = eulerian_radial_velocity_over_radius * x.get(2);
  return result;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
BondiMichel::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
    const IntermediateVars<DataType>& vars) const noexcept {
  auto result = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  const DataType mag_field_strength_factor =
      mag_field_strength_ / (cube(vars.radius) * sqrt(1.0 + 2.0 / vars.radius));
  result.get(0) = mag_field_strength_factor * x.get(0);
  result.get(1) = mag_field_strength_factor * x.get(1);
  result.get(2) = mag_field_strength_factor * x.get(2);
  return result;
}

bool operator==(const BondiMichel& lhs, const BondiMichel& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.mass_ == rhs.mass_ and lhs.sonic_radius_ == rhs.sonic_radius_ and
         lhs.sonic_density_ == rhs.sonic_density_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_ and
         lhs.mag_field_strength_ == rhs.mag_field_strength_ and
         lhs.sonic_fluid_speed_squared_ == rhs.sonic_fluid_speed_squared_ and
         lhs.sonic_sound_speed_squared_ == rhs.sonic_sound_speed_squared_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.mass_accretion_rate_over_four_pi_ ==
             rhs.mass_accretion_rate_over_four_pi_ and
         lhs.bernoulli_constant_squared_minus_one_ ==
             rhs.bernoulli_constant_squared_minus_one_ and
         lhs.rest_mass_density_at_infinity_ ==
             rhs.rest_mass_density_at_infinity_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const BondiMichel& lhs, const BondiMichel& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                  \
  template class BondiMichel::IntermediateVars<DTYPE(data)>;                  \
  template tuples::TaggedTuple<hydro::Tags::RestMassDensity<DTYPE(data)>>     \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::RestMassDensity<DTYPE(data)>> /*meta*/,         \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>    \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> meta,            \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<hydro::Tags::Pressure<DTYPE(data)>>            \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::Pressure<DTYPE(data)>> /*meta*/,                \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::SpecificInternalEnergy<DTYPE(data)>>                       \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DTYPE(data)>> /*meta*/,  \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<hydro::Tags::LorentzFactor<DTYPE(data)>>       \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::LorentzFactor<DTYPE(data)>> /*meta*/,           \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::SpatialVelocity<DTYPE(data), 3, Frame::Inertial>>          \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::SpatialVelocity<DTYPE(data), 3,                 \
                                              Frame::Inertial>> /*meta*/,     \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::MagneticField<DTYPE(data), 3, Frame::Inertial>>            \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,                   \
                                            Frame::Inertial>> /*meta*/,       \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept; \
  template tuples::TaggedTuple<                                               \
      hydro::Tags::DivergenceCleaningField<DTYPE(data)>>                      \
  BondiMichel::variables(                                                     \
      const tnsr::I<DTYPE(data), 3>& x,                                       \
      tmpl::list<hydro::Tags::DivergenceCleaningField<DTYPE(data)>> /*meta*/, \
      const BondiMichel::IntermediateVars<DTYPE(data)>& vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace Solutions
}  // namespace grmhd
/// \endcond
