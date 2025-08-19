// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Block.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/Topology.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Structure.CreateInitialMesh", "[Domain][Unit]") {
  const ElementId<1> element_id_1d{0};
  const ElementId<2> element_id_2d{0};
  const ElementId<3> element_id_3d{0};
  {
    INFO("From Element");
    const Element<1> interval(element_id_1d, {});
    const Element<2> rectangle(element_id_2d, {});
    const Element<3> brick(element_id_3d, {});
    for (const auto& quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3}}}, interval, quadrature) ==
            Mesh<1>{3, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 2}}}, rectangle, quadrature) ==
            Mesh<2>{{{3, 2}}, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 2, 4}}}, brick, quadrature) ==
            Mesh<3>{{{3, 2, 4}}, Spectral::Basis::Legendre, quadrature});
    }
  }
  {
    INFO("From Block and ElementId");
    const Block<1> interval(nullptr, 0, {});
    const Block<2> rectangle(nullptr, 0, {});
    const Block<3> brick(nullptr, 0, {});
    for (const auto& quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3}}}, interval, element_id_1d, quadrature) ==
            Mesh<1>{3, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 2}}}, rectangle, element_id_2d,
                                quadrature) ==
            Mesh<2>{{{3, 2}}, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 2, 4}}}, brick, element_id_3d,
                                quadrature) ==
            Mesh<3>{{{3, 2, 4}}, Spectral::Basis::Legendre, quadrature});
    }
  }
  {
    INFO("Another element");
    const Element<1> interval(ElementId<1>{1}, {});
    const Element<2> rectangle(ElementId<2>{1}, {});
    const Element<3> brick(ElementId<3>{1}, {});
    for (const auto& quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3}}, {{2}}}, interval, quadrature) ==
            Mesh<1>{2, Spectral::Basis::Legendre, quadrature});
      CHECK(create_initial_mesh({{{3, 3}}, {{2, 2}}}, rectangle, quadrature) ==
            Mesh<2>{2, Spectral::Basis::Legendre, quadrature});
      CHECK(
          create_initial_mesh({{{3, 3, 3}}, {{2, 2, 2}}}, brick, quadrature) ==
          Mesh<3>{2, Spectral::Basis::Legendre, quadrature});
    }
  }
  {
    INFO("annulus");
    const Element<2> annulus(element_id_2d, {}, domain::topologies::annulus);
    for (const auto& legendre_quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3, 4}}}, annulus, legendre_quadrature) ==
            Mesh<2>{
                {{3, 4}},
                std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier},
                std::array{legendre_quadrature,
                           Spectral::Quadrature::Equiangular}});
    }
  }
  {
    INFO("disk");
    const Block<2> disk(nullptr, 0, {}, "", domain::topologies::disk);
    for (const auto& legendre_quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3, 4}}}, disk, element_id_2d,
                                legendre_quadrature) ==
            Mesh<2>{{{3, 4}},
                    std::array{Spectral::Basis::ZernikeB2,
                               Spectral::Basis::ZernikeB2},
                    std::array{Spectral::Quadrature::GaussRadauUpper,
                               Spectral::Quadrature::Equiangular}});
    }
  }
  {
    INFO("cylindrical_shell");
    const Element<3> cylindrical_shell(element_id_3d, {},
                                       domain::topologies::cylindrical_shell);
    for (const auto& legendre_quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(
          create_initial_mesh({{{3, 2, 4}}}, cylindrical_shell,
                              legendre_quadrature) ==
          Mesh<3>{
              {{3, 2, 4}},
              std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier,
                         Spectral::Basis::Legendre},
              std::array{legendre_quadrature, Spectral::Quadrature::Equiangular,
                         legendre_quadrature}});
    }
  }
  {
    INFO("full_cylinder");
    const Block<3> full_cylinder(nullptr, 0, {}, "",
                                 domain::topologies::full_cylinder);
    for (const auto& legendre_quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      CHECK(create_initial_mesh({{{3, 2, 4}}}, full_cylinder, element_id_3d,
                                legendre_quadrature) ==
            Mesh<3>{{{3, 2, 4}},
                    std::array{Spectral::Basis::ZernikeB2,
                               Spectral::Basis::ZernikeB2,
                               Spectral::Basis::Legendre},
                    std::array{Spectral::Quadrature::GaussRadauUpper,
                               Spectral::Quadrature::Equiangular,
                               legendre_quadrature}});
    }
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      create_initial_mesh(
          {{{3, 4}}}, Block<2>{nullptr, 0, {}, "", domain::topologies::disk},
          ElementId<2>{0, {{SegmentId{1, 0}, SegmentId{0, 0}}}},
          Spectral::Quadrature::GaussLobatto),
      Catch::Matchers::ContainsSubstring(
          "Splitting Topology::B2Radial is not yet supported"));
#endif  // SPECTRE_DEBUG
}
}  // namespace domain
