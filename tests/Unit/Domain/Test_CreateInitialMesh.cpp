// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::Initialization {

SPECTRE_TEST_CASE("Unit.Domain.CreateInitialMesh", "[Domain][Unit]") {
  {
    INFO("Single element");
    CHECK(create_initial_mesh({{{3}}}, ElementId<1>{0}) ==
          Mesh<1>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(create_initial_mesh({{{3, 2}}}, ElementId<2>{0}) ==
          Mesh<2>{{{3, 2}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(create_initial_mesh({{{3, 2, 4}}}, ElementId<3>{0}) ==
          Mesh<3>{{{3, 2, 4}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
  }
  {
    INFO("Another element");
    CHECK(create_initial_mesh({{{3}}, {{2}}}, ElementId<1>{1}) ==
          Mesh<1>{2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(create_initial_mesh({{{3, 3}}, {{2, 2}}}, ElementId<2>{1}) ==
          Mesh<2>{2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(create_initial_mesh({{{3, 3, 3}}, {{2, 2, 2}}}, ElementId<3>{1}) ==
          Mesh<3>{2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
  }
  {
    INFO("Unaligned orientation");
    OrientationMap<2> unaligned(
        make_array(Direction<2>::lower_eta(), Direction<2>::upper_xi()));
    CHECK(
        create_initial_mesh({{{2, 3}}, {{4, 5}}}, ElementId<2>{0}, unaligned) ==
        Mesh<2>{{{3, 2}},
                Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto});
    CHECK(
        create_initial_mesh({{{2, 3}}, {{4, 5}}}, ElementId<2>{1}, unaligned) ==
        Mesh<2>{{{5, 4}},
                Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto});
  }
}

}  // namespace domain::Initialization
