// Distributed under the MIT License
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Reflection.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

template <size_t Dim>
void test() {
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      NewtonianEuler::BoundaryConditions::Reflection<Dim>,
      NewtonianEuler::BoundaryConditions::BoundaryCondition<Dim>,
      NewtonianEuler::System<Dim, NewtonianEuler::Solutions::SmoothFlow<Dim>>,
      tmpl::list<NewtonianEuler::BoundaryCorrections::Rusanov<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.NewtonianEuler.BoundaryConditions.Reflection",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::MassDensityCons>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::MomentumDensity<Dim>>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::EnergyDensity>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianEuler::Tags::MassDensityCons,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianEuler::Tags::MomentumDensity<Dim>,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianEuler::Tags::EnergyDensity,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::Velocity<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>>{
          "error", "mass_density_cons", "momentum_density", "energy_density",
          "flux_mass_density", "flux_momentum_density", "flux_energy_density",
          "velocity", "specific_internal_energy"},
      "Reflection:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      db::DataBox<tmpl::list<>>{}, tuples::TaggedTuple<>{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianEuler.BoundaryConditions.Reflection",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
