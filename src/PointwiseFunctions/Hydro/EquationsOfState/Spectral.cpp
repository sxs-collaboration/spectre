// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
std::vector<double> compute_integral_coefficients(
    const std::vector<double>& gamma_coefficients) {
  std::vector<double> result = gamma_coefficients;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] /= (1.0 + i);
  }
  return result;
}
}  // namespace

namespace EquationsOfState {
Spectral::Spectral(const double reference_density,
                   const double reference_pressure,
                   std::vector<double> coefficients, const double upper_density)
    : reference_density_(reference_density),
      reference_pressure_(reference_pressure),
      integral_coefficients_(compute_integral_coefficients(coefficients)),
      gamma_coefficients_(std::move(coefficients)),
      x_max_(log(upper_density / reference_density)),
      gamma_of_x_max_(gamma(x_max_)),
      integral_of_gamma_of_x_max_(integral_of_gamma(x_max_)) {
  // Setup table of specific energies
  // Use 6th order Gauss-Legendre quadrature.
  // Weights and collocation points can be calculated following
  // e.g. the gauleg method from Numerical Recipes
  // (Sec. 4.6.1 in 3rd edition). We only need to consider
  // collocation points with x>0, as the other points are
  // just symmetric with respect to the origin.
  quadrature_weights_ = {0.3607615730481386, 0.4679139345726910,
                         0.1713244923791704};
  quadrature_points_ = {0.6612093864662645, 0.2386191860831969,
                        0.9324695142031521};
  number_of_quadrature_coefs_ = quadrature_weights_.size();
  const auto n_points_epsilon = static_cast<size_t>(ceil(2.0 * x_max_) + 1.0);
  const double delta_x = x_max_ / (n_points_epsilon - 1.0);
  table_of_specific_energies_.resize(n_points_epsilon);
  table_of_specific_energies_[0] =
      reference_pressure_ / reference_density_ / (gamma_coefficients_[0] - 1.0);
  for (size_t i = 1; i < n_points_epsilon; i++) {
    table_of_specific_energies_[i] = table_of_specific_energies_[i - 1];
    // Use Gaussian quadrature to calculate the specific internal
    // energy at the next point, separated from the old point
    // by delta_x in log(density)
    // From the 1st law of thermodynamics,
    // d(epsilon)/dx = (Pressure)/(reference_density)*exp(-x)
    const double x0 = (i - 1.) * delta_x;
    for (size_t q = 0; q < number_of_quadrature_coefs_; q++) {
      const double xp = x0 + (1.0 + quadrature_points_[q]) * (delta_x / 2.0);
      const double xm = x0 + (1.0 - quadrature_points_[q]) * (delta_x / 2.0);
      table_of_specific_energies_[i] +=
          quadrature_weights_[q] * delta_x / (2.0 * reference_density) *
          (exp(-xp) * pressure_from_log_density(xp) +
           exp(-xm) * pressure_from_log_density(xm));
    }
  }
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(, Spectral, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(, Spectral, DataVector, 1)

std::unique_ptr<EquationOfState<true, 1>> Spectral::get_clone() const {
  auto clone = std::make_unique<Spectral>(*this);
  return std::unique_ptr<EquationOfState<true, 1>>(std::move(clone));
}

bool Spectral::operator==(const Spectral& rhs) const {
  return reference_density_ == rhs.reference_density_ and
         reference_pressure_ == rhs.reference_pressure_;
}

bool Spectral::operator!=(const Spectral& rhs) const {
  return not(*this == rhs);
}

bool Spectral::is_equal(const EquationOfState<true, 1>& rhs) const {
  const auto& derived_ptr = dynamic_cast<const Spectral* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

Spectral::Spectral(CkMigrateMessage* msg) : EquationOfState<true, 1>(msg) {}

void Spectral::pup(PUP::er& p) {
  EquationOfState<true, 1>::pup(p);
  p | reference_density_;
  p | reference_pressure_;
  p | integral_coefficients_;
  p | gamma_coefficients_;
  p | x_max_;
  p | gamma_of_x_max_;
  p | integral_of_gamma_of_x_max_;
  p | number_of_quadrature_coefs_;
  p | quadrature_weights_;
  p | quadrature_points_;
  p | table_of_specific_energies_;
}

// this evaluates the power series
// Int_0^x Gamma(xx) dxx = Sum_{n=0}^N \frac{gamma_n}{n+1} xx^{n+1}
double Spectral::integral_of_gamma(const double x) const {
  return evaluate_polynomial(integral_coefficients_, x) * x;
}

template <typename DataType>
Scalar<DataType> Spectral::pressure_from_density_impl(
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

template <class DataType>
Scalar<DataType> Spectral::rest_mass_density_from_enthalpy_impl(
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

template <class DataType>
Scalar<DataType> Spectral::specific_internal_energy_from_density_impl(
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

template <class DataType>
Scalar<DataType> Spectral::chi_from_density_impl(
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

template <class DataType>
Scalar<DataType> Spectral::kappa_times_p_over_rho_squared_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  return make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
}

// this evaluates the power series
// Gamma(x) = Sum_{n=0}^N gamma_n x^n
double Spectral::gamma(const double x) const {
  return evaluate_polynomial(gamma_coefficients_, x);
}

double Spectral::chi_from_density(const double rest_mass_density) const {
  const double x = log(rest_mass_density / reference_density_);
  const double P = pressure_from_log_density(x);
  const double chi = P / rest_mass_density *
                     (x <= 0.0 ? integral_coefficients_[0]
                               : (x < x_max_ ? gamma(x) : gamma(x_max_)));
  return chi;
}

double Spectral::specific_internal_energy_from_density(
    const double rest_mass_density) const {
  const double x = log(rest_mass_density / reference_density_);
  if (x <= 0.) {
    return reference_pressure_ / reference_density_ /
           (gamma_coefficients_[0] - 1.0) *
           exp((gamma_coefficients_[0] - 1.0) * x);
  }
  const size_t n_points_epsilon = table_of_specific_energies_.size();
  const double delta_x = x_max_ / (n_points_epsilon - 1.0);
  double specific_energy = 0.0;
  if (x >= x_max_) {
    // Analytic integral at constant gamma_coefficient above x_max_
    specific_energy = table_of_specific_energies_[n_points_epsilon - 1];
    specific_energy += pressure_from_log_density(x_max_) / reference_density_ *
                       exp(-x_max_) *
                       std::expm1((gamma_of_x_max_ - 1.0) * (x - x_max_)) /
                       (gamma_of_x_max_ - 1.0);
    return specific_energy;
  } else {
    const auto table_index = static_cast<size_t>(floor(x / delta_x));
    const double x0 = delta_x * table_index;
    specific_energy = table_of_specific_energies_[table_index];
    // Gaussian quadrature integral
    for (size_t q = 0; q < number_of_quadrature_coefs_; q++) {
      const double xp = (1.0 + quadrature_points_[q]) * (x - x0) / 2. + x0;
      const double xm = (1.0 - quadrature_points_[q]) * (x - x0) / 2. + x0;
      const double Pp = pressure_from_log_density(xp);
      const double Pm = pressure_from_log_density(xm);
      specific_energy += quadrature_weights_[q] * (x - x0) / 2. *
                         (Pp * exp(-xp) + Pm * exp(-xm)) / reference_density_;
    }
    return specific_energy;
  }
}

double Spectral::specific_enthalpy_from_density(
    const double rest_mass_density) const {
  return 1.0 + pressure_from_density(rest_mass_density) / rest_mass_density +
         specific_internal_energy_from_density(rest_mass_density);
}

// P = P_0 exp(Int_0^x Gamma(xx) dxx)
// where
// Gamma(x) = gamma_0                        for  x < 0
//            Sum_{n=0}^N gamma_n x^n             0 < x < x_{max}
//            Sum_{n=0}^N gamma_n x_{max}^n       x > x_{max}
double Spectral::pressure_from_log_density(const double x) const {
  const double integral_of_gamma_of_x =
      x <= 0.0 ? integral_coefficients_[0] * x
               : (x < x_max_ ? integral_of_gamma(x)
                             : integral_of_gamma_of_x_max_ +
                                   gamma_of_x_max_ * (x - x_max_));
  return reference_pressure_ * exp(integral_of_gamma_of_x);
}

double Spectral::pressure_from_density(const double rest_mass_density) const {
  const double x = log(rest_mass_density / reference_density_);
  return pressure_from_log_density(x);
}

// Solve for h(rho)=h0, which requires rootfinding for this EoS
double Spectral::rest_mass_density_from_enthalpy(
    const double specific_enthalpy) const {
  const double reference_enthalpy =
      specific_enthalpy_from_density(reference_density_);
  const double upper_density = reference_density_ * exp(x_max_);
  const double enthalpy_of_x_max_ =
      specific_enthalpy_from_density(upper_density);
  if (specific_enthalpy <= reference_enthalpy) {
    double rest_mass_density =
        (specific_enthalpy - 1.0) * (gamma_coefficients_[0] - 1.0) /
        gamma_coefficients_[0] *
        pow(reference_density_, gamma_coefficients_[0]) / reference_pressure_;
    rest_mass_density =
        pow(rest_mass_density, 1.0 / (gamma_coefficients_[0] - 1.0));
    return rest_mass_density;
  } else if (specific_enthalpy >= enthalpy_of_x_max_) {
    // Above maximum density, we also have an analytical expression
    // (h-1-eps_max)*(rho_max/P_max)+1/(Gamma_max-1) = Gamma_max / (Gamma_max-1)
    // * exp((Gamma_max-1)*(x-x_max)) [Can be derived by combining expressions
    // for epsilon(x) and P(x) with x = log(rho)]
    const size_t n_points_epsilon = table_of_specific_energies_.size();
    const double specific_internal_energy_of_x_max_ =
        table_of_specific_energies_[n_points_epsilon - 1];
    const double pressure_of_x_max_ = pressure_from_log_density(x_max_);
    double x_target =
        ((specific_enthalpy - 1.0 - specific_internal_energy_of_x_max_) *
             upper_density / pressure_of_x_max_ +
         1.0 / (gamma_of_x_max_ - 1.0)) *
        (gamma_of_x_max_ - 1.0) / gamma_of_x_max_;
    x_target = log(x_target) / (gamma_of_x_max_ - 1.0) + x_max_;
    return reference_density_ * exp(x_target);
  } else {
    // Root-finding appropriate between reference density and maximum density
    // We can use x=0 and x=x_max as bounds
    const auto f_df_lambda = [this, &specific_enthalpy](const double density) {
      const double f =
          this->specific_enthalpy_from_density(density) - specific_enthalpy;
      const double x = log(density / this->reference_density_);
      const double df = this->pressure_from_log_density(x) /
                        (density * density) * this->gamma(x);
      return std::make_pair(f, df);
    };
    const size_t digits = 14;
    const double intial_guess = 0.5 * (reference_density_ + upper_density);
    const auto root_from_lambda = RootFinder::newton_raphson(
        f_df_lambda, intial_guess, reference_density_, upper_density, digits);
    return root_from_lambda;
  }
}

PUP::able::PUP_ID EquationsOfState::Spectral::my_PUP_ID = 0;

}  // namespace EquationsOfState
