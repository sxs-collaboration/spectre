// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"

class DataVector;

namespace {
template <size_t Dim>
void test_tags() noexcept {
  CHECK(NewtonianEuler::Tags::MassDensity<DataVector>::name() == "MassDensity");
  CHECK(NewtonianEuler::Tags::MomentumDensity<DataVector, Dim,
                                              Frame::Inertial>::name() ==
        "MomentumDensity");
  CHECK(NewtonianEuler::Tags::MomentumDensity<DataVector, Dim,
                                              Frame::Grid>::name() ==
        "Grid_MomentumDensity");
  CHECK(NewtonianEuler::Tags::EnergyDensity<DataVector>::name() ==
        "EnergyDensity");
  CHECK(NewtonianEuler::Tags::Velocity<DataVector, Dim,
                                       Frame::Inertial>::name() == "Velocity");
  CHECK(
      NewtonianEuler::Tags::Velocity<DataVector, Dim, Frame::Logical>::name() ==
      "Logical_Velocity");
  CHECK(NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>::name() ==
        "SpecificInternalEnergy");
  CHECK(NewtonianEuler::Tags::Pressure<DataVector>::name() == "Pressure");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Tags",
                  "[Unit][Evolution]") {
  test_tags<1>();
  test_tags<2>();
  test_tags<3>();
}
