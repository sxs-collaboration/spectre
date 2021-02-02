// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Dirichlet.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.Burgers.BoundaryConditions.Dirichlet",
                  "[Unit][Burgers]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers/BoundaryConditions/"};
  MAKE_GENERATOR(gen);
  helpers::test_boundary_condition_with_python<
      Burgers::BoundaryConditions::Dirichlet,
      Burgers::BoundaryConditions::BoundaryCondition, Burgers::System,
      tmpl::list<Burgers::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen), "Dirichlet",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<Burgers::Tags::U>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>{
          "error", "u_1", "flux_1"},
      "Dirichlet:\n"
      "  U: 1.0\n",
      Index<0>{1}, db::DataBox<tmpl::list<>>{}, tuples::TaggedTuple<>{});

  helpers::test_boundary_condition_with_python<
      Burgers::BoundaryConditions::Dirichlet,
      Burgers::BoundaryConditions::BoundaryCondition, Burgers::System,
      tmpl::list<Burgers::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen), "Dirichlet",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<Burgers::Tags::U>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>{
          "error", "u_m1", "flux_m1"},
      "Dirichlet:\n"
      "  U: -1.0\n",
      Index<0>{1}, db::DataBox<tmpl::list<>>{}, tuples::TaggedTuple<>{});
}
