// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

struct SomeSourceType {};
struct SomeInitialDataType {
  using source_term_type = SomeSourceType;
};

namespace {
template <size_t Dim>
void test_tags() {
  TestHelpers::db::test_simple_tag<NewtonianEuler::Tags::MassDensityCons>(
      "MassDensityCons");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::MassDensity<DataVector>>("MassDensity");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::MomentumDensity<Dim, Frame::Grid>>(
      "Grid_MomentumDensity");
  TestHelpers::db::test_simple_tag<NewtonianEuler::Tags::EnergyDensity>(
      "EnergyDensity");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::Velocity<DataVector, Dim, Frame::Logical>>(
      "Logical_Velocity");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>(
      "SpecificInternalEnergy");
  TestHelpers::db::test_simple_tag<NewtonianEuler::Tags::Pressure<DataVector>>(
      "Pressure");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::SoundSpeed<DataVector>>("SoundSpeed");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::SoundSpeedSquared<DataVector>>("SoundSpeedSquared");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::CharacteristicSpeeds<Dim>>("CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<NewtonianEuler::Tags::VMinus>("VMinus");
  TestHelpers::db::test_simple_tag<NewtonianEuler::Tags::VMomentum<Dim>>(
      "VMomentum");
  TestHelpers::db::test_simple_tag<NewtonianEuler::Tags::VPlus>("VPlus");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::InternalEnergyDensity<DataVector>>(
      "InternalEnergyDensity");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::KineticEnergyDensity<DataVector>>(
      "KineticEnergyDensity");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::MachNumber<DataVector>>("MachNumber");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::RamPressure<DataVector, Dim, Frame::Grid>>(
      "Grid_RamPressure");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::SpecificKineticEnergy<DataVector>>(
      "SpecificKineticEnergy");
  TestHelpers::db::test_base_tag<NewtonianEuler::Tags::SourceTermBase>(
      "SourceTermBase");
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::Tags::SourceTerm<SomeInitialDataType>>("SourceTerm");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Tags",
                  "[Unit][Evolution]") {
  test_tags<1>();
  test_tags<2>();
  test_tags<3>();
}
