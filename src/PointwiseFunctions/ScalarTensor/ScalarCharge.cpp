// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarCharge.hpp"

#include <cmath>

#include "Utilities/Gsl.hpp"

void ScalarTensor::scalar_charge_integrand(
    const gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::i<DataVector, 3>& phi,
    const tnsr::I<DataVector, 3>& unit_normal_vector) {
  // Project the scalar gradient on the normal vector
  tenex::evaluate(result, phi(ti::i) * unit_normal_vector(ti::I));
  // Multiply by integral prefactor
  get(*result) /= -4.0 * M_PI;
}
