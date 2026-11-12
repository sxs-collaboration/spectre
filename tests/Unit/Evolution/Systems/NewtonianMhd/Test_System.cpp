// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test_hydrodynamic_variables() {
  using system = NewtonianMhd::System<Dim, false>;
  CHECK(system::name() == "NewtonianEuler");
  CHECK(not system::evolve_magnetic_field);
  CHECK(std::is_same_v<typename system::flux_variables,
                       tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                                  NewtonianMhd::Tags::MomentumDensity<Dim>,
                                  NewtonianMhd::Tags::EnergyDensity>>);
  CHECK(
      std::is_same_v<typename system::primitive_variables_tag::tags_list,
                     tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                hydro::Tags::SpatialVelocity<DataVector, Dim>,
                                hydro::Tags::SpecificInternalEnergy<DataVector>,
                                hydro::Tags::Pressure<DataVector>>>);
}

void test_magnetic_variables() {
  using system = NewtonianMhd::System<3, true>;
  CHECK(system::name() == "NewtonianMhd");
  CHECK(system::evolve_magnetic_field);
  CHECK(std::is_same_v<
        typename system::flux_variables,
        tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                   NewtonianMhd::Tags::MomentumDensity<3>,
                   NewtonianMhd::Tags::EnergyDensity,
                   NewtonianMhd::Tags::MagneticFieldCons<>,
                   NewtonianMhd::Tags::DivergenceCleaningFieldCons>>);
  CHECK(std::is_same_v<
        typename system::primitive_variables_tag::tags_list,
        tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                   hydro::Tags::SpatialVelocity<DataVector, 3>,
                   hydro::Tags::SpecificInternalEnergy<DataVector>,
                   hydro::Tags::Pressure<DataVector>,
                   hydro::Tags::MagneticField<DataVector, 3>,
                   hydro::Tags::DivergenceCleaningField<DataVector>>>);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.System", "[Unit][Evolution]") {
  test_hydrodynamic_variables<1>();
  test_hydrodynamic_variables<2>();
  test_hydrodynamic_variables<3>();
  test_magnetic_variables();
}
