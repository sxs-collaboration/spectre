// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"

#include <cmath>
#include <numeric>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "NumericalAlgorithms/Spectral/Clenshaw.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

double evaluate_cosine_series(const std::vector<double>& coefficients,
                              const double& cosx) {
  // cos(nx) = cos((n-1)*x)*2*cosx - 1* cos((n-2)*x)
  return Spectral::evaluate_clenshaw(coefficients, 2.0 * cosx, -1.0, cosx,
                                     2.0 * cosx * cosx - 1.0);
}
double evaluate_sine_series(const std::vector<double>& coefficients, double x,
                            const double& cosx) {
  // sin(nx) = sin((n-1)*x) * 2*cos(x) - 1 * sin((n-2)x)
  auto sinx = sin(x);
  return Spectral::evaluate_clenshaw(coefficients, 2.0 * cosx, -1.0, sinx,
                                     2.0 * cosx * sinx);
}
}  // namespace

namespace EquationsOfState {
template <typename LowDensityEoS>
Enthalpy<LowDensityEoS>::Coefficients::Coefficients(
    std::vector<double> in_polynomial_coefficients,
    std::vector<double> in_sin_coefficients,
    std::vector<double> in_cos_coefficients, double in_trig_scale,
    double in_reference_density, double in_exponential_constant)
    : polynomial_coefficients(std::move(in_polynomial_coefficients)),
      sin_coefficients(std::move(in_sin_coefficients)),
      cos_coefficients(std::move(in_cos_coefficients)),
      trig_scale(in_trig_scale),
      reference_density(in_reference_density) {
  if (not std::isnan(in_exponential_constant)) {
    // used to construct the pressuure coefficient
    has_exponential_prefactor = true;
    exponential_external_constant = in_exponential_constant;
  } else {
    has_exponential_prefactor = false;
    exponential_external_constant =
        std::numeric_limits<double>::signaling_NaN();
  }
}
template <typename LowDensityEoS>
bool Enthalpy<LowDensityEoS>::Coefficients::operator==(
    const Coefficients& rhs) const {
  return polynomial_coefficients == rhs.polynomial_coefficients and
         sin_coefficients == rhs.sin_coefficients and
         cos_coefficients == rhs.cos_coefficients and
         trig_scale == rhs.trig_scale and
         reference_density == rhs.reference_density and
         has_exponential_prefactor == rhs.has_exponential_prefactor and
         (has_exponential_prefactor ? exponential_external_constant ==
                                          rhs.exponential_external_constant
                                    : true);
}
namespace {
std::vector<double> operator-(const std::vector<double>& lhs,
                              const std::vector<double>& rhs) {
  ASSERT(lhs.size() == rhs.size(), "Incorrect Sizes in vector addition");
  std::vector<double> result(lhs.size());
  for (size_t index = 0; index < lhs.size(); index++) {
    result[index] = lhs[index] - rhs[index];
  }
  return result;
}
}  // namespace
template <typename LowDensityEoS>
typename Enthalpy<LowDensityEoS>::Coefficients
Enthalpy<LowDensityEoS>::compute_pressure_coefficients(
    const typename Enthalpy<LowDensityEoS>::Coefficients& enthalpy,
    const typename Enthalpy<LowDensityEoS>::Coefficients& energy_density) {
  // precompute the coefficients of p = rho * h  - e

  auto polynomial_coefficients =
      enthalpy.polynomial_coefficients - energy_density.polynomial_coefficients;
  auto sin_coefficients =
      enthalpy.sin_coefficients - energy_density.sin_coefficients;
  auto cos_coefficients =
      enthalpy.cos_coefficients - energy_density.cos_coefficients;
  auto trig_scale = enthalpy.trig_scale;
  auto reference_density = enthalpy.reference_density;
  auto exponential_external_constant =
      -energy_density.exponential_external_constant;
  return Enthalpy<LowDensityEoS>::Coefficients(
      polynomial_coefficients, sin_coefficients, cos_coefficients, trig_scale,
      reference_density, exponential_external_constant);
}

// Given an expansion h(x) = sum_i f_i(x) , compute int_a^x sum_i f_i(x) e^x +
// F(a) where f_i(x) could be one of the basis functions  used, i.e. x^i/i!,
// sin(ikx) or cos(ikx)
template <typename LowDensityEoS>
typename Enthalpy<LowDensityEoS>::Coefficients
Enthalpy<LowDensityEoS>::Coefficients::compute_exponential_integral(
    const std::pair<double, double>& initial_condition,
    const double minimum_density) {
  // This is used to compute the energy density coefficients
  if (has_exponential_prefactor) {
    // No code currently calls this
    ERROR(
        "Attempting to exponentially integrate an EoS decomposition series "
        "with an exponential prefactor!  This is not yet implemented, if you "
        "need this, please file an issue to get it added.");
  }
  std::vector<double> integral_poly_coeffs(polynomial_coefficients.size(), 0.0);
  std::vector<double> integral_sin_coeffs(sin_coefficients.size(), 0.0);
  std::vector<double> integral_cos_coeffs(cos_coefficients.size(), 0.0);
  Enthalpy::Coefficients exponential_integral_coefficients = *this;
  // Preprocessing to put coefficients in better basis c_i z^i  = c_i' z^i/i!
  std::vector<double> taylor_series_coefficients(
      polynomial_coefficients.size());
  for (size_t i = 0; i < integral_poly_coeffs.size(); i++) {
    taylor_series_coefficients[i] =
        polynomial_coefficients[i] * static_cast<double>(factorial(i));
  }
  // i indexes terms of the integrand, each of which contributes i+1 terms (of
  // degree r <= i) to the integral indexed by r.  Therefore, r indexes terms
  // of the integral.
  for (size_t i = 0; i < taylor_series_coefficients.size(); i++) {
    for (size_t r = 0; r <= i; r++) {
      integral_poly_coeffs[r] +=
          ((i - r) % 2 == 0 ? 1.0 : -1.0) * taylor_series_coefficients[i];
    }
  }
  // restore the default normalization for the coefficients
  for (size_t r = 0; r < integral_poly_coeffs.size(); r++) {
    integral_poly_coeffs[r] /= static_cast<double>(factorial(r));
  }

  // note again sum starts from 0, basis functions are sin([j+1]kx)
  for (size_t j = 0; j < sin_coefficients.size(); j++) {
    // contribution from the sine terms
    double k = trig_scale;
    integral_sin_coeffs[j] +=
        1.0 / (square(j + 1) * square(k) + 1.0) * sin_coefficients[j];
    integral_cos_coeffs[j] -= (static_cast<double>(j + 1) * k) /
                              (square(j + 1) * square(k) + 1.0) *
                              sin_coefficients[j];
    // contribution from the cosine terms
    integral_cos_coeffs[j] +=
        1.0 / (square(j + 1) * square(k) + 1.0) * cos_coefficients[j];
    integral_sin_coeffs[j] += (static_cast<double>(j + 1) * k) /
                              (square(j + 1) * square(k) + 1.0) *
                              cos_coefficients[j];
  }
  exponential_integral_coefficients.has_exponential_prefactor = true;
  exponential_integral_coefficients.exponential_external_constant = 0.0;
  exponential_integral_coefficients.polynomial_coefficients =
      std::move(integral_poly_coeffs);
  exponential_integral_coefficients.sin_coefficients =
      std::move(integral_sin_coeffs);
  exponential_integral_coefficients.cos_coefficients =
      std::move(integral_cos_coeffs);
  const double new_constant =
      initial_condition.second -
      evaluate_coefficients(exponential_integral_coefficients,
                            initial_condition.first, minimum_density);
  exponential_integral_coefficients.exponential_external_constant =
      new_constant;
  return exponential_integral_coefficients;
}
template <typename LowDensityEoS>
typename Enthalpy<LowDensityEoS>::Coefficients
Enthalpy<LowDensityEoS>::Coefficients::compute_derivative() {
  std::vector<double> derivative_poly_coeffs(polynomial_coefficients.size());
  std::vector<double> derivative_sin_coeffs(sin_coefficients.size());
  std::vector<double> derivative_cos_coeffs(cos_coefficients.size());
  Enthalpy::Coefficients derivative_coefficients = *this;
  // d/dz e^z \sum_i a_i f(z) = e^z \sum_i a_i f_i(z)  + e^z \sum_i a_i f_i'(z)
  if (has_exponential_prefactor) {
    // Currently unused, but may be useful in the future
    ERROR(
        "This branch is untested, it may be "
        "used to compute derivatives of internal"
        "energy (or related quantities) in the future. ");
    derivative_poly_coeffs = polynomial_coefficients;
    derivative_sin_coeffs = sin_coefficients;
    derivative_cos_coeffs = cos_coefficients;
    derivative_poly_coeffs[polynomial_coefficients.size() - 1] = 0.0;
    for (size_t i = 0; i < polynomial_coefficients.size() - 1; i++) {
      derivative_poly_coeffs[i] +=
          polynomial_coefficients[i + 1] * static_cast<double>(i + 1);
    }
    // Again sum starts from 0
    for (size_t j = 0; j < sin_coefficients.size(); j++) {
      derivative_cos_coeffs[j] +=
          sin_coefficients[j] * (static_cast<double>(j + 1) * trig_scale);
      derivative_sin_coeffs[j] +=
          -cos_coefficients[j] * (static_cast<double>(j + 1) * trig_scale);
    }
  } else {  // There is no exponential prefactor
    // The final coefficient will be zero because nothing differentiates to it
    derivative_poly_coeffs[polynomial_coefficients.size() - 1] = 0.0;
    for (size_t i = 0; i < polynomial_coefficients.size() - 1; i++) {
      derivative_poly_coeffs[i] =
          polynomial_coefficients[i + 1] * static_cast<double>(i + 1);
    }
    for (size_t j = 0; j < sin_coefficients.size(); j++) {
      derivative_cos_coeffs[j] =
          sin_coefficients[j] * (static_cast<double>(j + 1) * trig_scale);
      derivative_sin_coeffs[j] =
          -cos_coefficients[j] * (static_cast<double>(j + 1) * trig_scale);
    }
  }
  derivative_coefficients.polynomial_coefficients =
      std::move(derivative_poly_coeffs);
  derivative_coefficients.sin_coefficients = std::move(derivative_sin_coeffs);
  derivative_coefficients.cos_coefficients = std::move(derivative_cos_coeffs);
  return derivative_coefficients;
}

template <typename LowDensityEoS>
void Enthalpy<LowDensityEoS>::Coefficients::pup(PUP::er& p) {
  p | polynomial_coefficients;
  p | sin_coefficients;
  p | cos_coefficients;
  p | trig_scale;
  p | reference_density;
  p | has_exponential_prefactor;
  p | exponential_external_constant;
}
template <typename LowDensityEoS>
Enthalpy<LowDensityEoS>::Enthalpy(
    const double reference_density, const double max_density,
    const double min_density, const double trig_scale,
    const std::vector<double>& polynomial_coefficients,
    const std::vector<double>& sin_coefficients,
    const std::vector<double>& cos_coefficients,
    const LowDensityEoS& low_density_eos, const double transition_delta_epsilon)
    : reference_density_(reference_density),
      minimum_density_(min_density),
      maximum_density_(max_density),
      low_density_eos_(low_density_eos),
      coefficients_(polynomial_coefficients, sin_coefficients, cos_coefficients,
                    trig_scale, reference_density)

{
  minimum_enthalpy_ = specific_enthalpy_from_density(minimum_density_);
  // Compute based on low density behavior
  double min_energy_density =
      minimum_density_ +
      get(low_density_eos_.specific_internal_energy_from_density(
          Scalar<double>(minimum_density_))) *
          minimum_density_ +
      transition_delta_epsilon;
  exponential_integral_coefficients_ =
      coefficients_.compute_exponential_integral(
          {x_from_density(min_density), min_energy_density}, minimum_density_);
  derivative_coefficients_ = coefficients_.compute_derivative();
  pressure_coefficients_ = compute_pressure_coefficients(
      coefficients_, exponential_integral_coefficients_);
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename LowDensityEoS>,
                                     Enthalpy<LowDensityEoS>, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename LowDensityEoS>,
                                     Enthalpy<LowDensityEoS>, DataVector, 1)

template <typename LowDensityEoS>
std::unique_ptr<EquationOfState<true, 1>> Enthalpy<LowDensityEoS>::get_clone()
    const {
  auto clone = std::make_unique<Enthalpy>(*this);
  return std::unique_ptr<EquationOfState<true, 1>>(std::move(clone));
}

template <typename LowDensityEoS>
bool Enthalpy<LowDensityEoS>::is_equal(
    const EquationOfState<true, 1>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Enthalpy<LowDensityEoS>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}
template <typename LowDensityEoS>
bool Enthalpy<LowDensityEoS>::operator==(
    const Enthalpy<LowDensityEoS>& rhs) const {
  return low_density_eos_ == rhs.low_density_eos_ and
         coefficients_ == rhs.coefficients_ and
         exponential_integral_coefficients_ ==
             rhs.exponential_integral_coefficients_;
  // Don't need to check the derivative coefficients
}
template <typename LowDensityEoS>
bool Enthalpy<LowDensityEoS>::operator!=(
    const Enthalpy<LowDensityEoS>& rhs) const {
  return not(*this == rhs);
}

template <typename LowDensityEoS>
Enthalpy<LowDensityEoS>::Enthalpy(CkMigrateMessage* msg)
    : EquationOfState<true, 1>(msg) {}

template <typename LowDensityEoS>
void Enthalpy<LowDensityEoS>::pup(PUP::er& p) {
  EquationOfState<true, 1>::pup(p);
  p | reference_density_;
  p | maximum_density_;
  p | minimum_density_;
  p | minimum_enthalpy_;
  p | low_density_eos_;
  p | coefficients_;
  p | exponential_integral_coefficients_;
  p | derivative_coefficients_;
  p | pressure_coefficients_;
}

template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::x_from_density(
    const double rest_mass_density) const {
  ASSERT(rest_mass_density > 0.0, "Density must be greater than zero");
  return log(rest_mass_density / reference_density_);
}
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::density_from_x(const double x) const {
  return reference_density_ * exp(x);
}
// Only works for rho > rho_min
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::energy_density_from_log_density(
    const double x, const double rest_mass_density) const {
  return evaluate_coefficients(exponential_integral_coefficients_, x,
                               rest_mass_density);
}

// Evaluate the function represented by the coefficinets at x  = log(rho/rho_0)
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::evaluate_coefficients(
    const Enthalpy<LowDensityEoS>::Coefficients& coefficients, const double x,
    const double exponential_prefactor) {
  const double k_times_x = coefficients.trig_scale * x;
  const double polynomial_contribution =
      evaluate_polynomial(coefficients.polynomial_coefficients, x);
  double value = polynomial_contribution;
  // A couple of edge cases
  switch (coefficients.sin_coefficients.size()) {
    case 0:
      break;
    case 1: {
      // One sine and one cosine term
      value += (coefficients.sin_coefficients[0] * sin(k_times_x) +
                coefficients.cos_coefficients[0] * cos(k_times_x));
      break;
    }
    default: {
      // Use Clenshaw's method to evaluate
      const double coskx = cos(k_times_x);
      const double sin_contribution =
          evaluate_sine_series(coefficients.sin_coefficients, k_times_x, coskx);
      const double cos_contribution =
          evaluate_cosine_series(coefficients.cos_coefficients, coskx);
      value += (sin_contribution + cos_contribution);
      break;
    }
  }
  if (coefficients.has_exponential_prefactor) {
    // multiply by some constant, typically rho(x) = rho_0 * exp(x)
    // add an additional constant external to the previous prefactor
    return value * exponential_prefactor +
           coefficients.exponential_external_constant;
  }
  return value;
}
template <typename LowDensityEoS>
template <typename DataType>
Scalar<DataType> Enthalpy<LowDensityEoS>::pressure_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{pressure_from_density(get(rest_mass_density))};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] = pressure_from_density(get(rest_mass_density)[i]);
    }
    return result;
  }
}
template <typename LowDensityEoS>
template <class DataType>
Scalar<DataType> Enthalpy<LowDensityEoS>::rest_mass_density_from_enthalpy_impl(
    const Scalar<DataType>& specific_enthalpy) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{
        rest_mass_density_from_enthalpy(get(specific_enthalpy))};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(specific_enthalpy, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] =
          rest_mass_density_from_enthalpy(get(specific_enthalpy)[i]);
    }
    return result;
  }
}

template <typename LowDensityEoS>
template <class DataType>
Scalar<DataType>
Enthalpy<LowDensityEoS>::specific_internal_energy_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{
        specific_internal_energy_from_density(get(rest_mass_density))};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] =
          specific_internal_energy_from_density(get(rest_mass_density)[i]);
    }
    return result;
  }
}

template <typename LowDensityEoS>
template <class DataType>
Scalar<DataType> Enthalpy<LowDensityEoS>::chi_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{chi_from_density(get(rest_mass_density))};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] = chi_from_density(get(rest_mass_density)[i]);
    }
    return result;
  }
}

template <typename LowDensityEoS>
template <class DataType>
Scalar<DataType>
Enthalpy<LowDensityEoS>::kappa_times_p_over_rho_squared_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  return make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
}
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::chi_from_density(
    const double rest_mass_density) const {
  if (Enthalpy::in_low_density_domain(rest_mass_density)) {
    return get(
        low_density_eos_.chi_from_density(Scalar<double>(rest_mass_density)));
  } else {
    const double x = x_from_density(rest_mass_density);
    return evaluate_coefficients(derivative_coefficients_, x);
  }
}

template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::specific_internal_energy_from_density(
    const double rest_mass_density) const {
  if (Enthalpy::in_low_density_domain(rest_mass_density)) {
    return get(low_density_eos_.specific_internal_energy_from_density(
        Scalar<double>(rest_mass_density)));
  } else {
    return 1.0 / rest_mass_density *
               energy_density_from_log_density(
                   x_from_density(rest_mass_density), rest_mass_density) -
           1.0;
  }
}
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::specific_enthalpy_from_density(
    const double rest_mass_density) const {
  if (Enthalpy::in_low_density_domain(rest_mass_density)) {
    return get(hydro::relativistic_specific_enthalpy(
        Scalar<double>(rest_mass_density),
        low_density_eos_.specific_internal_energy_from_density(
            Scalar<double>(rest_mass_density)),
        low_density_eos_.pressure_from_density(
            Scalar<double>(rest_mass_density))));
  } else {
    return evaluate_coefficients(coefficients_,
                                 x_from_density(rest_mass_density));
  }
}

// P(x) = rho(x)h(x) - e(x) with h the specific enthalpy, and e
// the energy density.
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::pressure_from_log_density(
    const double x, const double density) const {
  return evaluate_coefficients(pressure_coefficients_, x, density);
}
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::pressure_from_density(
    const double rest_mass_density) const {
  if (in_low_density_domain(rest_mass_density)) {
    return get(low_density_eos_.pressure_from_density(
        Scalar<double>(rest_mass_density)));
  } else {
    const double x = x_from_density(rest_mass_density);
    return pressure_from_log_density(x, rest_mass_density);
  }
}

// Solve for h(rho)=h0, which requires rootfinding for this EoS
template <typename LowDensityEoS>
double Enthalpy<LowDensityEoS>::rest_mass_density_from_enthalpy(
    const double specific_enthalpy) const {
  if (specific_enthalpy <= minimum_enthalpy_) {
    return get(low_density_eos_.rest_mass_density_from_enthalpy(
        Scalar<double>(specific_enthalpy)));
  } else {
    // Root-finding appropriate between reference density and maximum density
    // We can use x=0 and x=x_max as bounds
    const auto f = [this, &specific_enthalpy](const double density) {
      const auto x = x_from_density(density);
      return evaluate_coefficients(coefficients_, x) - specific_enthalpy;
    };
    return RootFinder::toms748(f, minimum_density_, maximum_density_, 1.0e-14,
                               1.0e-15);
  }
}
template <typename LowDensityEoS>
PUP::able::PUP_ID EquationsOfState::Enthalpy<LowDensityEoS>::my_PUP_ID = 0;

template class EquationsOfState::Enthalpy<Spectral>;
template class EquationsOfState::Enthalpy<PolytropicFluid<true>>;
template class EquationsOfState::Enthalpy<EquationsOfState::Enthalpy<Spectral>>;
template class EquationsOfState::Enthalpy<
    EquationsOfState::Enthalpy<EquationsOfState::Enthalpy<Spectral>>>;
}  // namespace EquationsOfState
