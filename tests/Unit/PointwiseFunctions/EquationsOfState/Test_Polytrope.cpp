// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/EquationsOfState/EOS.hpp"
#include "PointwiseFunctions/EquationsOfState/Polytrope.hpp"
#include "tests/Unit/DataStructures/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {

Scalar<double> expected_baryon_density(
    const Scalar<double>& log_specific_enthalpy) {
  const double gas_constant = 4.3419;

  const double gamma = 2.0000;

  return Scalar<double>{
      pow((expm1(get(log_specific_enthalpy)) / gas_constant -
           expm1(get(log_specific_enthalpy)) / (gas_constant * gamma)),
          (1.0 / (gamma - 1.0)))};
}

void test_baryon_density(double log_specific_enthalpy) {
  std::unique_ptr<EquationOfState> eostest =
      std::make_unique<EquationsOfState::Polytrope>(4.3419, 2.0000);

  Scalar<double> baryon_density =
      eostest->baryon_density(Scalar<double>{log_specific_enthalpy});

  Scalar<double> expected_baryondensity =
      expected_baryon_density(Scalar<double>{log_specific_enthalpy});

  CHECK(baryon_density == expected_baryondensity);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Polytrope",
                  "[Unit][PointwiseFunctions]") {
  for (size_t i = 0; i < 100; i++) {
    test_baryon_density(i * 1.0e-03);
  }
}

}  // end namespace
