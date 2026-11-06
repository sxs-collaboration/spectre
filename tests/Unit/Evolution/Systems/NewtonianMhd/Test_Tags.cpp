// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

struct SomeSourceType {};
struct SomeInitialDataType {
  using source_term_type = SomeSourceType;
};

namespace {
template <size_t Dim>
void test_tags() {
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::MassDensityCons>(
      "MassDensityCons");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::MomentumDensity<Dim, Frame::Grid>>(
      "Grid_MomentumDensity");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::EnergyDensity>(
      "EnergyDensity");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::SoundSpeed<DataVector>>(
      "SoundSpeed");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::CharacteristicSpeeds<Dim>>("CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::VMinus>("VMinus");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::VMomentum<Dim>>(
      "VMomentum");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::VPlus>("VPlus");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::InternalEnergyDensity<DataVector>>(
      "InternalEnergyDensity");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::KineticEnergyDensity<DataVector>>(
      "KineticEnergyDensity");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::MachNumber<DataVector>>(
      "MachNumber");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::RamPressure<DataVector, Dim, Frame::Grid>>(
      "Grid_RamPressure");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::SpecificKineticEnergy<DataVector>>(
      "SpecificKineticEnergy");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::SourceTerm<Dim>>(
      "SourceTerm");
}

// Magnetic field is 3D only
void test_magnetic_tags() {
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::MagneticFieldCons<Frame::Grid>>(
      "Grid_MagneticFieldCons");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::DivergenceCleaningFieldCons>(
      "DivergenceCleaningFieldCons");
  TestHelpers::db::test_simple_tag<NewtonianMhd::Tags::DivergenceCleaningSpeed>(
      "DivergenceCleaningSpeed");
  TestHelpers::db::test_simple_tag<
      NewtonianMhd::Tags::ConstraintDampingParameter>(
      "ConstraintDampingParameter");

  CHECK(TestHelpers::test_option_tag<
            NewtonianMhd::OptionTags::DivergenceCleaningSpeed>("1.3") == 1.3);
  CHECK(
      TestHelpers::test_option_tag<NewtonianMhd::OptionTags::DampingParameter>(
          "0.17") == 0.17);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Tags",
                  "[Unit][Evolution]") {
  test_tags<1>();
  test_tags<2>();
  test_tags<3>();
  test_magnetic_tags();
}
