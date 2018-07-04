// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Controller.hpp"

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

template <size_t DerivOrder>
DataVector Controller<DerivOrder>::operator()(
    const DataVector& timescales,
    const std::array<DataVector, DerivOrder + 1>& q_and_derivs,
    const double q_time_offset, const double deriv_time_offset) const noexcept {
  // helper lambda for computing the binomial coefficients
  const auto binomial = [](size_t N, size_t k) noexcept {
    return falling_factorial(N, k) / factorial(k);
  };

  // The coefficients a0_{k} are chosen such that the system is critically
  // damped, so they are given by: a0_{k} = (N choose k) timescales^{k-N}
  std::array<DataVector, DerivOrder> coefs0;
  DataVector inv_timescales = 1.0 / timescales;
  DataVector tau = inv_timescales;
  for (size_t i = DerivOrder; i-- > 0;) {
    gsl::at(coefs0, i) = binomial(DerivOrder, i) * tau;
    // add another power of timescales
    tau *= inv_timescales;
  }

  // correct for time offset
  std::array<DataVector, DerivOrder> coefs;
  for (size_t i = 0; i < DerivOrder; i++) {
    gsl::at(coefs, i) = gsl::at(coefs0, i);
    double q_dt = q_time_offset;
    double deriv_dt = deriv_time_offset;
    double fact_denom = 1.0;
    for (size_t j = i; j-- > 0;) {
      // update the factorial coefficient of this term in the series expansion
      fact_denom *= (i - j);
      const double t_offset = (j == 0 ? q_dt : deriv_dt);
      gsl::at(coefs, i) += gsl::at(coefs0, j) * t_offset / fact_denom;
      // update the time coefficients for the next term in the series expansion
      q_dt *= q_time_offset;
      deriv_dt *= deriv_time_offset;
    }
  }

  // compute denominator associated with correcting for time offset
  // reuse tau DataVector allocation and properly initialize
  DataVector& denom = tau;
  denom = 1.0;
  double q_dt = q_time_offset;
  double deriv_dt = deriv_time_offset;
  double fact_denom = 1.0;
  for (size_t i = DerivOrder; i-- > 0;) {
    // update the factorial coefficient of this term in the series expansion
    fact_denom *= (DerivOrder - i);
    const double t_offset = (i == 0 ? q_dt : deriv_dt);
    denom += gsl::at(coefs0, i) * t_offset / fact_denom;
    // update the time coefficients for the next term in the series expansion
    q_dt *= q_time_offset;
    deriv_dt *= deriv_time_offset;
  }

  // compute control signal
  // reuse inv_timescales DataVector allocation and properly initialize
  DataVector& control_signal = inv_timescales;
  control_signal = coefs[0] * q_and_derivs[0];
  for (size_t i = 1; i < DerivOrder; i++) {
    control_signal += gsl::at(coefs, i) * gsl::at(q_and_derivs, i);
  }

  return control_signal / denom;
}

// explicit instantiations
/// \cond
template class Controller<2>;
/// \endcond
