// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/ScalarSource.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"
#include "PointwiseFunctions/ScalarTensor/RampUpFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace ScalarTensor {

void gauss_bonnet_scalar_source(
    const gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar,
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters, const double mass_psi,
    const std::pair<double, double> start_and_ramp_times, const double time) {
  // Compute the Riemann squared scalar in vacuum
  gr::gauss_bonnet_scalar_in_vacuum(scalar_source, weyl_electric_scalar,
                                    weyl_magnetic_scalar);
  // Multiply by the source coupling function
  multiply_by_negative_deriv_of_coupling_func(
      scalar_source, psi, coupling_parameters, start_and_ramp_times, time);
  // Add mass term
  scalar_source->get() += square(mass_psi) * get(psi);
}

Scalar<DataVector> gauss_bonnet_scalar_source(
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar,
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters, const double mass_psi,
    const std::pair<double, double> start_and_ramp_times, const double time) {
  Scalar<DataVector> result{};
  gauss_bonnet_scalar_source(make_not_null(&result), weyl_electric_scalar,
                             weyl_magnetic_scalar, psi, coupling_parameters,
                             mass_psi, start_and_ramp_times, time);
  return result;
}

void multiply_by_negative_deriv_of_coupling_func(
    const gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters,
    const std::pair<double, double> start_and_ramp_times, const double time) {
  const double linear_coupling_psi = gsl::at(coupling_parameters, 0);
  const double first_coupling_psi = gsl::at(coupling_parameters, 1);
  const double second_coupling_psi = gsl::at(coupling_parameters, 2);
  const auto ones_scalar = make_with_value<Scalar<DataVector>>(psi, 1.0);

  // Ramp up factor
  const double ramp_factor = nonic_ramp_function(time, start_and_ramp_times);

  const double linear_coupling_psi_over_four =
      0.25 * ramp_factor * linear_coupling_psi;
  const double first_coupling_psi_over_four =
      0.25 * ramp_factor * first_coupling_psi;
  const double second_coupling_psi_over_four =
      0.25 * ramp_factor * second_coupling_psi;

  *scalar_source->get() *= -linear_coupling_psi_over_four * ones_scalar.get() -
                           first_coupling_psi_over_four * psi.get() -
                           second_coupling_psi_over_four * cube(psi.get());
}

void multiply_by_negative_second_deriv_of_coupling_func(
    const gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& psi,
    const std::array<double, 3>& coupling_parameters,
    const std::pair<double, double> start_and_ramp_times, const double time) {
  // Linear coupling drops out here
  const double first_coupling_psi = gsl::at(coupling_parameters, 1);
  const double second_coupling_psi = gsl::at(coupling_parameters, 2);
  const auto ones_scalar = make_with_value<Scalar<DataVector>>(psi, 1.0);

  // Ramp up factor
  const double ramp_factor = nonic_ramp_function(time, start_and_ramp_times);

  const double first_coupling_psi_over_four =
      0.25 * ramp_factor * first_coupling_psi;
  const double second_coupling_psi_over_four =
      0.25 * ramp_factor * second_coupling_psi;

  *scalar_source->get() *=
      -first_coupling_psi_over_four * ones_scalar.get() -
      (3.0 * second_coupling_psi_over_four) * square(psi.get());
}

}  // namespace ScalarTensor
