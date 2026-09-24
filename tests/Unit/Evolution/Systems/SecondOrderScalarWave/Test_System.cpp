// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

namespace {
template <size_t Dim>
void check_system() {
  using System = SecondOrderScalarWave::System<Dim>;
  static_assert(
      std::is_same_v<
          typename System::variables_tag,
          tmpl::list<
              ::Tags::Variables<tmpl::list<SecondOrderScalarWave::Tags::Psi,
                                           SecondOrderScalarWave::Tags::Pi>>,
              ::Tags::BoundaryVariables<
                  Dim, tmpl::list<evolution::dg::Tags::BoundaryValue<
                           SecondOrderScalarWave::Tags::Psi>>>>>);
  static_assert(evolution::dg::system_has_boundary_variables_v<System>);
  static_assert(
      std::is_same_v<evolution::dg::boundary_variables_tag<System>,
                     ::Tags::BoundaryVariables<
                         Dim, tmpl::list<evolution::dg::Tags::BoundaryValue<
                                  SecondOrderScalarWave::Tags::Psi>>>>);
  CHECK(System::name() == "SecondOrderScalarWave");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SecondOrderScalarWave.System",
                  "[Unit][Evolution]") {
  check_system<1>();
  check_system<2>();
  check_system<3>();
}
