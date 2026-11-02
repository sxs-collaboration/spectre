// Distributed under the MIT License
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/SmoothFlow.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

template <size_t Dim>
void test() {
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      NewtonianMhd::BoundaryConditions::Reflection<Dim>,
      NewtonianMhd::BoundaryConditions::BoundaryCondition<Dim>,
      NewtonianMhd::System<Dim>,
      tmpl::list<NewtonianMhd::BoundaryCorrections::Rusanov<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.NewtonianMhd.BoundaryConditions.Reflection",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              NewtonianMhd::Tags::MassDensityCons>,
          helpers::Tags::PythonFunctionName<
              NewtonianMhd::Tags::MomentumDensity<Dim>>,
          helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::EnergyDensity>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianMhd::Tags::MassDensityCons,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianMhd::Tags::MomentumDensity<Dim>,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianMhd::Tags::EnergyDensity, tmpl::size_t<Dim>,
                           Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpatialVelocity<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpecificInternalEnergy<DataVector>>>{
          "error", "mass_density_cons", "momentum_density", "energy_density",
          "flux_mass_density", "flux_momentum_density", "flux_energy_density",
          "velocity", "specific_internal_energy"},
      "Reflection:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      db::DataBox<tmpl::list<>>{}, tuples::TaggedTuple<>{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.BoundaryConditions.Reflection",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
