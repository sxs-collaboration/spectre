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

SPECTRE_TEST_CASE("Unit.Domain.Structure.CreateInitialMesh", "[Domain][Unit]") {
  {
    INFO("Single element");
    for (const auto& quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3}}}, ElementId<1>{0}, quadrature) ==
            Mesh<1>{3, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 2}}}, ElementId<2>{0}, quadrature) ==
            Mesh<2>{{{3, 2}}, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 2, 4}}}, ElementId<3>{0}, quadrature) ==
            Mesh<3>{{{3, 2, 4}}, Spectral::Basis::Legendre, quadrature});
    }
  }
  {
    INFO("Another element");
    for (const auto& quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3}}, {{2}}}, ElementId<1>{1}, quadrature) ==
            Mesh<1>{2, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 3}}, {{2, 2}}}, ElementId<2>{1},
                                quadrature) ==
            Mesh<2>{2, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 3, 3}}, {{2, 2, 2}}}, ElementId<3>{1},
                                quadrature) ==
            Mesh<3>{2, Spectral::Basis::Legendre, quadrature});
    }
  }
  {
    INFO("Unaligned orientation");
    OrientationMap<2> unaligned(
        make_array(Direction<2>::lower_eta(), Direction<2>::upper_xi()));
    for (const auto& quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{2, 3}}, {{4, 5}}}, ElementId<2>{0},
                                quadrature, unaligned) ==
            Mesh<2>{{{3, 2}}, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{2, 3}}, {{4, 5}}}, ElementId<2>{1},
                                quadrature, unaligned) ==
            Mesh<2>{{{5, 4}}, Spectral::Basis::Legendre, quadrature});
    }
  }
}

}  // namespace domain::Initialization
