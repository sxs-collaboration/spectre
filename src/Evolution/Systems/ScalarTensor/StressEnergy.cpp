// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/StressEnergy.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor {

void add_stress_energy_term_to_dt_pi(
    const gsl::not_null<tnsr::aa<DataVector, 3_st>*> dt_pi,
    const tnsr::aa<DataVector, 3_st>& trace_reversed_stress_energy,
    const Scalar<DataVector>& lapse) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      dt_pi->get(a, b) -=
          16.0 * M_PI * get(lapse) * trace_reversed_stress_energy.get(a, b);
    }
  }
}

void trace_reversed_stress_energy(
    const gsl::not_null<tnsr::aa<DataVector, 3_st>*> stress_energy,
    const Scalar<DataVector>& pi_scalar,
    const tnsr::i<DataVector, 3_st>& phi_scalar,
    const Scalar<DataVector>& lapse) {
  get<0, 0>(*stress_energy) = square(get(lapse) * get(pi_scalar));
  for (size_t i = 0; i < 3; ++i) {
    stress_energy->get(0, i + 1) =
        -get(lapse) * get(pi_scalar) * phi_scalar.get(i);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      stress_energy->get(i + 1, j + 1) = phi_scalar.get(i) * phi_scalar.get(j);
    }
  }
}

}  // namespace ScalarTensor
