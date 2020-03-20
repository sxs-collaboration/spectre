// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

namespace {

// Any of the two functions of the pressure and the initial data
// in Eqns. (4.6) and (4.7) of Toro.
template <size_t Dim>
struct FunctionOfPressureAndData {
  FunctionOfPressureAndData(const typename NewtonianEuler::Solutions::
                                RiemannProblem<Dim>::InitialData& data,
                            const double adiabatic_index) noexcept
      : state_pressure_(data.pressure_),
        adiabatic_index_(adiabatic_index),
        constant_a_(data.constant_a_),
        constant_b_(data.constant_b_) {
    prefactor_ = 2.0 * data.sound_speed_ / (adiabatic_index_ - 1.0);
    prefactor_deriv_ = data.sound_speed_ / adiabatic_index_ / state_pressure_;
    exponent_ = 0.5 * (adiabatic_index_ - 1.0) / adiabatic_index_;
    exponent_deriv_ = -0.5 * (adiabatic_index_ + 1.0) / adiabatic_index_;
  }

  double operator()(const double pressure) const noexcept {
    // Value depends on whether the initial state is a shock
    // (pressure > pressure of initial state) or a rarefaction wave
    // (pressure <= pressure of initial state)
    return pressure > state_pressure_
               ? (pressure - state_pressure_) *
                     sqrt(constant_a_ / (pressure + constant_b_))
               : prefactor_ *
                     (pow(pressure / state_pressure_, exponent_) - 1.0);
  }

  // First derivative with respect to the pressure.
  double deriv(const double pressure) const noexcept {
    return pressure > state_pressure_
               ? 0.5 * sqrt(constant_a_) *
                     (pressure + state_pressure_ + 2.0 * constant_b_) /
                     pow(pressure + constant_b_, 1.5)
               : prefactor_deriv_ * pow(pressure / state_pressure_, exponent_);
  }

 private:
  double state_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double constant_a_ = std::numeric_limits<double>::signaling_NaN();
  double constant_b_ = std::numeric_limits<double>::signaling_NaN();

  // Auxiliary variables for computing the rarefaction wave
  double prefactor_ = std::numeric_limits<double>::signaling_NaN();
  double prefactor_deriv_ = std::numeric_limits<double>::signaling_NaN();
  double exponent_ = std::numeric_limits<double>::signaling_NaN();
  double exponent_deriv_ = std::numeric_limits<double>::signaling_NaN();
};

}  // namespace

/// \cond
namespace NewtonianEuler {
namespace Solutions {

template <size_t Dim>
RiemannProblem<Dim>::RiemannProblem(
    const double adiabatic_index, const double initial_position,
    const double left_mass_density,
    const std::array<double, Dim>& left_velocity, const double left_pressure,
    const double right_mass_density,
    const std::array<double, Dim>& right_velocity, const double right_pressure,
    const double pressure_star_tol) noexcept
    : adiabatic_index_(adiabatic_index),
      initial_position_(initial_position),
      left_initial_data_(left_mass_density, left_velocity, left_pressure,
                         adiabatic_index, propagation_axis_),
      right_initial_data_(right_mass_density, right_velocity, right_pressure,
                          adiabatic_index, propagation_axis_),
      pressure_star_tol_(pressure_star_tol),
      equation_of_state_(adiabatic_index) {
  const double delta_u = right_initial_data_.normal_velocity_ -
                         left_initial_data_.normal_velocity_;

  // The pressure positivity condition must be met (Eqn. 4.40 of Toro).
  ASSERT(2.0 *
                     (left_initial_data_.sound_speed_ +
                      right_initial_data_.sound_speed_) /
                     (adiabatic_index_ - 1.0) -
                 delta_u >
             0.0,
         "The pressure positivity condition must be met. Initial data do not "
         "satisfy this criterion.");

  // Before evaluating the solution at any (x, t), the type of solution
  // (shock or rarefaction) on each side of the contact discontinuity
  // must be sorted out. To do so, the first step is to compute the
  // state variables in the star region by solving a transcendental equation,
  // which is what all the math in this constructor is about.

  // The initial guess to obtain p_* is given by the
  // Two-Shock approximation (Eqn. (4.47) in Toro), which gives best results
  // overall. Other options are also possible (see Section 4.3.2 of Toro.)
  const double guess_for_pressure_star = [&delta_u, this ](
      const InitialData& left, const InitialData& right) noexcept {
    // Eqn. (4.47) of Toro: guess derived from a linearized solution
    // based on primitive variables.
    double p_pv =
        0.5 * (left.pressure_ + right.pressure_ -
               0.25 * delta_u * (left.mass_density_ + right.mass_density_) *
                   (left.sound_speed_ + right.sound_speed_));
    p_pv = std::max(pressure_star_tol_, p_pv);

    // Eqns. (4.48) of Toro: Two-Shock wave approximation.
    const double g_left = sqrt(left.constant_a_ / (p_pv + left.constant_b_));
    const double g_right = sqrt(right.constant_a_ / (p_pv + right.constant_b_));
    return std::max(pressure_star_tol_, (g_left * left.pressure_ +
                                         g_right * right.pressure_ - delta_u) /
                                            (g_left + g_right));
  }
  (left_initial_data_, right_initial_data_);

  // Compute bracket for root finder according to value of the function whose
  // root we want (Eqn. 4.39 of Toro.)
  const FunctionOfPressureAndData<Dim> f_of_p_left(left_initial_data_,
                                                   adiabatic_index_);
  const FunctionOfPressureAndData<Dim> f_of_p_right(right_initial_data_,
                                                    adiabatic_index_);
  const auto p_minmax =
      std::minmax(left_initial_data_.pressure_, right_initial_data_.pressure_);
  const double f_min =
      f_of_p_left(p_minmax.first) + f_of_p_right(p_minmax.first) + delta_u;
  const double f_max =
      f_of_p_left(p_minmax.second) + f_of_p_right(p_minmax.second) + delta_u;

  double pressure_lower;
  double pressure_upper;
  if (f_min > 0.0 and f_max > 0.0) {
    pressure_lower = 0.0;
    pressure_upper = p_minmax.first;
  } else if (f_min < 0.0 and f_max > 0.0) {
    pressure_lower = p_minmax.first;
    pressure_upper = p_minmax.second;
  } else {
    pressure_lower = p_minmax.second;
    pressure_upper = 10.0 * pressure_lower;  // Arbitrary upper bound < \infty
  }

  // Now get pressure by solving transcendental equation. Newton-Raphson is OK.
  const auto f_of_p_and_deriv =
      [&f_of_p_left, &f_of_p_right, &delta_u ](const double pressure) noexcept {
    // Function of pressure in Eqn. (4.5) of Toro.
    return std::make_pair(
        f_of_p_left(pressure) + f_of_p_right(pressure) + delta_u,
        f_of_p_left.deriv(pressure) + f_of_p_right.deriv(pressure));
  };
  try {
    pressure_star_ = RootFinder::newton_raphson(
        f_of_p_and_deriv, guess_for_pressure_star, pressure_lower,
        pressure_upper, -log10(pressure_star_tol_));
  } catch (std::exception& exception) {
    ERROR(
        "Failed to find p_* with Newton-Raphson root finder. Got "
        "exception message:\n"
        << exception.what()
        << "\nIf the residual is small you can change the tolerance for the "
           "root finder in the input file.");
  }

  // Calculated p_*, u_* is obtained from Eqn. (4.9) in Toro.
  velocity_star_ =
      0.5 * (left_initial_data_.normal_velocity_ +
             right_initial_data_.normal_velocity_ -
             f_of_p_left(pressure_star_) + f_of_p_right(pressure_star_));
}

template <size_t Dim>
void RiemannProblem<Dim>::pup(PUP::er& p) noexcept {
  p | adiabatic_index_;
  p | initial_position_;
  p | left_initial_data_;
  p | right_initial_data_;
  p | pressure_star_tol_;
  p | pressure_star_;
  p | velocity_star_;
  p | equation_of_state_;
}

template <size_t Dim>
RiemannProblem<Dim>::InitialData::InitialData(
    const double mass_density, const std::array<double, Dim>& velocity,
    const double pressure, const double adiabatic_index,
    const size_t propagation_axis) noexcept
    : mass_density_(mass_density), velocity_(velocity), pressure_(pressure) {
  ASSERT(mass_density_ > 0.0,
         "The mass density must be positive. Value given: " << mass_density);
  ASSERT(pressure_ > 0.0,
         "The pressure must be positive. Value given: " << pressure);

  sound_speed_ = sqrt(adiabatic_index * pressure / mass_density);
  normal_velocity_ = gsl::at(velocity, propagation_axis);
  constant_a_ = 2.0 / (adiabatic_index + 1.0) / mass_density;
  constant_b_ = (adiabatic_index - 1.0) * pressure / (adiabatic_index + 1.0);
}

template <size_t Dim>
void RiemannProblem<Dim>::InitialData::pup(PUP::er& p) noexcept {
  p | mass_density_;
  p | velocity_;
  p | pressure_;
  p | sound_speed_;
  p | normal_velocity_;
  p | constant_a_;
  p | constant_b_;
}

template <size_t Dim>
RiemannProblem<Dim>::Wave::Wave(const InitialData& data,
                                const double pressure_star,
                                const double velocity_star,
                                const double adiabatic_index,
                                const Side& side) noexcept
    : pressure_ratio_(pressure_star / data.pressure_),
      is_shock_(pressure_ratio_ > 1.0),
      data_(data),
      shock_(data, pressure_ratio_, adiabatic_index, side),
      rarefaction_(data, pressure_ratio_, velocity_star, adiabatic_index,
                   side) {}

template <size_t Dim>
double RiemannProblem<Dim>::Wave::mass_density(const double x_shifted,
                                               const double t) const noexcept {
  return (is_shock_ == true ? shock_.mass_density(x_shifted, t, data_)
                            : rarefaction_.mass_density(x_shifted, t, data_));
}

template <size_t Dim>
double RiemannProblem<Dim>::Wave::normal_velocity(
    const double x_shifted, const double t, const double velocity_star) const
    noexcept {
  return (
      is_shock_ == true
          ? shock_.normal_velocity(x_shifted, t, data_, velocity_star)
          : rarefaction_.normal_velocity(x_shifted, t, data_, velocity_star));
}

template <size_t Dim>
double RiemannProblem<Dim>::Wave::pressure(const double x_shifted,
                                           const double t,
                                           const double pressure_star) const
    noexcept {
  return (is_shock_ == true
              ? shock_.pressure(x_shifted, t, data_, pressure_star)
              : rarefaction_.pressure(x_shifted, t, data_, pressure_star));
}

template <size_t Dim>
RiemannProblem<Dim>::Shock::Shock(const InitialData& data,
                                  const double pressure_ratio,
                                  const double adiabatic_index,
                                  const Side& side) noexcept
    : direction_(side == Side::Left ? -1.0 : 1.0) {
  const double gamma_mm = adiabatic_index - 1.0;
  const double gamma_pp = adiabatic_index + 1.0;
  const double gamma_mm_over_gamma_pp = gamma_mm / gamma_pp;

  mass_density_star_ = data.mass_density_ *
                       (pressure_ratio + gamma_mm_over_gamma_pp) /
                       (pressure_ratio * gamma_mm_over_gamma_pp + 1.0);

  shock_speed_ =
      data.normal_velocity_ +
      direction_ * data.sound_speed_ *
          sqrt(0.5 * (gamma_pp * pressure_ratio + gamma_mm) / adiabatic_index);
}

template <size_t Dim>
double RiemannProblem<Dim>::Shock::mass_density(const double x_shifted,
                                                const double t,
                                                const InitialData& data) const
    noexcept {
  return mass_density_star_ +
         (data.mass_density_ - mass_density_star_) *
             step_function(direction_ * (x_shifted - shock_speed_ * t));
}

template <size_t Dim>
double RiemannProblem<Dim>::Shock::normal_velocity(
    const double x_shifted, const double t, const InitialData& data,
    const double velocity_star) const noexcept {
  return velocity_star +
         (data.normal_velocity_ - velocity_star) *
             step_function(direction_ * (x_shifted - shock_speed_ * t));
}

template <size_t Dim>
double RiemannProblem<Dim>::Shock::pressure(const double x_shifted,
                                            const double t,
                                            const InitialData& data,
                                            const double pressure_star) const
    noexcept {
  return pressure_star +
         (data.pressure_ - pressure_star) *
             step_function(direction_ * (x_shifted - shock_speed_ * t));
}

template <size_t Dim>
RiemannProblem<Dim>::Rarefaction::Rarefaction(const InitialData& data,
                                              const double pressure_ratio,
                                              const double velocity_star,
                                              const double adiabatic_index,
                                              const Side& side) noexcept
    : direction_(side == Side::Left ? -1.0 : 1.0) {
  gamma_mm_ = adiabatic_index - 1.0;
  gamma_pp_ = adiabatic_index + 1.0;
  mass_density_star_ =
      data.mass_density_ * pow(pressure_ratio, 1.0 / adiabatic_index);
  sound_speed_star_ =
      data.sound_speed_ *
      pow(pressure_ratio, 0.5 * (adiabatic_index - 1.0) / adiabatic_index);
  head_speed_ = data.normal_velocity_ + direction_ * data.sound_speed_;
  tail_speed_ = velocity_star + direction_ * sound_speed_star_;
}

template <size_t Dim>
double RiemannProblem<Dim>::Rarefaction::mass_density(
    const double x_shifted, const double t, const InitialData& data) const
    noexcept {
  const double s = (t > 0.0 ? (x_shifted / t) : 0.0);
  return direction_ * (x_shifted - tail_speed_ * t) < 0.0
             ? mass_density_star_
             : (direction_ * (x_shifted - tail_speed_ * t) >= 0.0 and
                        direction_ * (x_shifted - head_speed_ * t) < 0.0
                    ? (data.mass_density_ *
                       pow((2.0 - direction_ * gamma_mm_ *
                                      (data.normal_velocity_ - s) /
                                      data.sound_speed_) /
                               gamma_pp_,
                           2.0 / gamma_mm_))
                    : data.mass_density_);
}

template <size_t Dim>
double RiemannProblem<Dim>::Rarefaction::normal_velocity(
    const double x_shifted, const double t, const InitialData& data,
    const double velocity_star) const noexcept {
  const double s = (t > 0.0 ? (x_shifted / t) : 0.0);
  return direction_ * (x_shifted - tail_speed_ * t) < 0.0
             ? velocity_star
             : (direction_ * (x_shifted - tail_speed_ * t) >= 0.0 and
                        direction_ * (x_shifted - head_speed_ * t) < 0.0
                    ? (2.0 *
                       (0.5 * gamma_mm_ * data.normal_velocity_ + s -
                        direction_ * data.sound_speed_) /
                       gamma_pp_)
                    : data.normal_velocity_);
}

template <size_t Dim>
double RiemannProblem<Dim>::Rarefaction::pressure(
    const double x_shifted, const double t, const InitialData& data,
    const double pressure_star) const noexcept {
  const double s = (t > 0.0 ? (x_shifted / t) : 0.0);
  return direction_ * (x_shifted - tail_speed_ * t) < 0.0
             ? pressure_star
             : (direction_ * (x_shifted - tail_speed_ * t) >= 0.0 and
                        direction_ * (x_shifted - head_speed_ * t) < 0.0
                    ? (data.pressure_ *
                       pow((2.0 - direction_ * gamma_mm_ *
                                      (data.normal_velocity_ - s) /
                                      data.sound_speed_) /
                               gamma_pp_,
                           2.0 * (gamma_mm_ + 1.0) / gamma_mm_))
                    : data.pressure_);
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::MassDensity<DataType>> RiemannProblem<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted, const double t,
    tmpl::list<Tags::MassDensity<DataType>> /*meta*/, const Wave& left,
    const Wave& right) const noexcept {
  auto mass_density = make_with_value<Scalar<DataType>>(x_shifted, 0.0);
  const double u_star_times_t = velocity_star_ * t;
  for (size_t s = 0; s < get_size(get<0>(x_shifted)); ++s) {
    const double x_shifted_s = get_element(x_shifted.get(propagation_axis_), s);
    get_element(get(mass_density), s) =
        (x_shifted_s < u_star_times_t ? left.mass_density(x_shifted_s, t)
                                      : right.mass_density(x_shifted_s, t));
  }
  return mass_density;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Velocity<DataType, Dim>>
RiemannProblem<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted, const double t,
    tmpl::list<Tags::Velocity<DataType, Dim>> /*meta*/,
    const Wave& left, const Wave& right) const noexcept {
  auto velocity = make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(
      get<0>(x_shifted), 0.0);

  const double u_star_times_t = velocity_star_ * t;
  for (size_t s = 0; s < get_size(get<0>(x_shifted)); ++s) {
    const double x_shifted_s = get_element(x_shifted.get(propagation_axis_), s);

    size_t index = propagation_axis_ % Dim;
    get_element(velocity.get(index), s) =
        (x_shifted_s < u_star_times_t
             ? left.normal_velocity(x_shifted_s, t, velocity_star_)
             : right.normal_velocity(x_shifted_s, t, velocity_star_));

    if (Dim > 1) {
      index = (propagation_axis_ + 1) % Dim;
      get_element(velocity.get(index), s) =
          (x_shifted_s < u_star_times_t
               ? gsl::at(left_initial_data_.velocity_, index)
               : gsl::at(right_initial_data_.velocity_, index));
    }

    if (Dim > 2) {
      index = (propagation_axis_ + 2) % Dim;
      get_element(velocity.get(index), s) =
          (x_shifted_s < u_star_times_t
               ? gsl::at(left_initial_data_.velocity_, index)
               : gsl::at(right_initial_data_.velocity_, index));
    }
  }
  return velocity;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::Pressure<DataType>> RiemannProblem<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted, const double t,
    tmpl::list<Tags::Pressure<DataType>> /*meta*/, const Wave& left,
    const Wave& right) const noexcept {
  auto pressure = make_with_value<Scalar<DataType>>(x_shifted, 0.0);
  const double u_star_times_t = velocity_star_ * t;
  for (size_t s = 0; s < get_size(get<0>(x_shifted)); ++s) {
    const double x_shifted_s = get_element(x_shifted.get(propagation_axis_), s);
    get_element(get(pressure), s) =
        (x_shifted_s < u_star_times_t
             ? left.pressure(x_shifted_s, t, pressure_star_)
             : right.pressure(x_shifted_s, t, pressure_star_));
  }
  return pressure;
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>
RiemannProblem<Dim>::variables(
    const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted, const double t,
    tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
    const Wave& left, const Wave& right) const noexcept {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<Tags::MassDensity<DataType>>(
          variables(x_shifted, t, tmpl::list<Tags::MassDensity<DataType>>{},
                    left, right)),
      get<Tags::Pressure<DataType>>(variables(
          x_shifted, t, tmpl::list<Tags::Pressure<DataType>>{}, left, right)));
}

template <size_t Dim>
bool operator==(const RiemannProblem<Dim>& lhs,
                const RiemannProblem<Dim>& rhs) noexcept {
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.initial_position_ == rhs.initial_position_ and
         lhs.left_initial_data_ == rhs.left_initial_data_ and
         lhs.right_initial_data_ == rhs.right_initial_data_ and
         lhs.pressure_star_tol_ == rhs.pressure_star_tol_ and
         lhs.pressure_star_ == rhs.pressure_star_ and
         lhs.velocity_star_ == rhs.velocity_star_;
}

template <size_t Dim>
bool operator!=(const RiemannProblem<Dim>& lhs,
                const RiemannProblem<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_CLASS(_, data)                                     \
  template class RiemannProblem<DIM(data)>;                            \
  template bool operator==(const RiemannProblem<DIM(data)>&,           \
                           const RiemannProblem<DIM(data)>&) noexcept; \
  template bool operator!=(const RiemannProblem<DIM(data)>&,           \
                           const RiemannProblem<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (1, 2, 3))

#define INSTANTIATE_SCALARS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>                     \
      RiemannProblem<DIM(data)>::variables(                                  \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x_shifted, \
          const double t, tmpl::list<TAG(data) < DTYPE(data)>>,              \
          const Wave& left, const Wave& right) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (1, 2, 3), (double, DataVector),
                        (Tags::MassDensity, Tags::Pressure,
                         Tags::SpecificInternalEnergy))

#define INSTANTIATE_VELOCITY(_, data)                                        \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), DIM(data)>>          \
      RiemannProblem<DIM(data)>::variables(                                  \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x_shifted, \
          const double t,                                                    \
          tmpl::list<TAG(data) < DTYPE(data), DIM(data)>>,  \
          const Wave& left, const Wave& right) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VELOCITY, (1, 2, 3), (double, DataVector),
                        (Tags::Velocity))

#undef DIM
#undef DTYPE
#undef TAG
#undef INSTANTIATE_CLASS
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VELOCITY

}  // namespace Solutions
}  // namespace NewtonianEuler
/// \endcond
