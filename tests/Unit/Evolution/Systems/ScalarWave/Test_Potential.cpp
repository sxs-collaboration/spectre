// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/Potential.hpp"

namespace {

template <size_t SpatialDim>
void test_potential_simple() {
  const DataVector dv{1.0, 2.0, 3.0};
  const Scalar<DataVector> psi{dv};
  const double mass2 = 2.0;

  const auto result = ScalarWave::potential<SpatialDim>(psi, mass2);

  const DataVector expected = 0.5 * mass2 * dv * dv;

  CHECK_ITERABLE_APPROX(get(result), expected);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.Potential",
                  "[Unit][Evolution]") {
  test_potential_simple<1>();
  test_potential_simple<2>();
  test_potential_simple<3>();
}
