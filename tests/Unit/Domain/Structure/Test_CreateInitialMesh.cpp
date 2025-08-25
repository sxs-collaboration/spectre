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
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3}}}, interval, i1_basis, i1_quadrature) ==
              Mesh<1>{3, i1_basis, i1_quadrature});
        CHECK(create_initial_mesh({{{3, 2}}}, rectangle, i1_basis,
                                  i1_quadrature) ==
              Mesh<2>{{{3, 2}}, i1_basis, i1_quadrature});
        CHECK(create_initial_mesh({{{3, 2, 4}}}, brick, i1_basis,
                                  i1_quadrature) ==
              Mesh<3>{{{3, 2, 4}}, i1_basis, i1_quadrature});
      }
    }
#ifdef SPECTRE_DEBUG
    CHECK_THROWS_WITH(
        create_initial_mesh({{{3}}}, interval, Spectral::Basis::Fourier,
                            Spectral::Quadrature::Gauss),
        Catch::Matchers::ContainsSubstring("Invalid I1 Basis"));
    CHECK_THROWS_WITH(
        create_initial_mesh({{{3}}}, interval, Spectral::Basis::Legendre,
                            Spectral::Quadrature::Equiangular),
        Catch::Matchers::ContainsSubstring("Invalid I1 Quadrature"));
#endif  // SPECTRE_DEBUG
  }
  {
    INFO("From Block and ElementId");
    const Block<1> interval(nullptr, 0, {});
    const Block<2> rectangle(nullptr, 0, {});
    const Block<3> brick(nullptr, 0, {});
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3}}}, interval, element_id_1d, i1_basis,
                                  i1_quadrature) ==
              Mesh<1>{3, i1_basis, i1_quadrature});
        CHECK(create_initial_mesh({{{3, 2}}}, rectangle, element_id_2d,
                                  i1_basis, i1_quadrature) ==
              Mesh<2>{{{3, 2}}, i1_basis, i1_quadrature});
        CHECK(create_initial_mesh({{{3, 2, 4}}}, brick, element_id_3d, i1_basis,
                                  i1_quadrature) ==
              Mesh<3>{{{3, 2, 4}}, i1_basis, i1_quadrature});
      }
    }
  }
  {
    INFO("Another element");
    const Element<1> interval(ElementId<1>{1}, {});
    const Element<2> rectangle(ElementId<2>{1}, {});
    const Element<3> brick(ElementId<3>{1}, {});
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3}}, {{2}}}, interval, i1_basis,
                                  i1_quadrature) ==
              Mesh<1>{2, i1_basis, i1_quadrature});
        CHECK(create_initial_mesh({{{3, 3}}, {{2, 2}}}, rectangle, i1_basis,
                                  i1_quadrature) ==
              Mesh<2>{2, i1_basis, i1_quadrature});
        CHECK(create_initial_mesh({{{3, 3, 3}}, {{2, 2, 2}}}, brick, i1_basis,
                                  i1_quadrature) ==
              Mesh<3>{2, i1_basis, i1_quadrature});
      }
    }
  }
  {
    INFO("annulus");
    const Element<2> annulus(element_id_2d, {}, domain::topologies::annulus);
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(
            create_initial_mesh({{{3, 4}}}, annulus, i1_basis, i1_quadrature) ==
            Mesh<2>{
                {{3, 4}},
                std::array{i1_basis, Spectral::Basis::Fourier},
                std::array{i1_quadrature, Spectral::Quadrature::Equiangular}});
      }
    }
  }
  {
    INFO("disk");
    const Block<2> disk(nullptr, 0, {}, "", domain::topologies::disk);
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3, 4}}}, disk, element_id_2d, i1_basis,
                                  i1_quadrature) ==
              Mesh<2>{{{3, 4}},
                      std::array{Spectral::Basis::ZernikeB2,
                                 Spectral::Basis::ZernikeB2},
                      std::array{Spectral::Quadrature::GaussRadauUpper,
                                 Spectral::Quadrature::Equiangular}});
      }
    }
  }
  {
    INFO("cylindrical_shell");
    const Element<3> cylindrical_shell(element_id_3d, {},
                                       domain::topologies::cylindrical_shell);
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(
            create_initial_mesh({{{3, 2, 4}}}, cylindrical_shell, i1_basis,
                                i1_quadrature) ==
            Mesh<3>{{{3, 2, 4}},
                    std::array{i1_basis, Spectral::Basis::Fourier, i1_basis},
                    std::array{i1_quadrature, Spectral::Quadrature::Equiangular,
                               i1_quadrature}});
      }
    }
  }
  {
    INFO("full_cylinder");
    const Block<3> full_cylinder(nullptr, 0, {}, "",
                                 domain::topologies::full_cylinder);
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3, 2, 4}}}, full_cylinder, element_id_3d,
                                  i1_basis, i1_quadrature) ==
              Mesh<3>{{{3, 2, 4}},
                      std::array{Spectral::Basis::ZernikeB2,
                                 Spectral::Basis::ZernikeB2, i1_basis},
                      std::array{Spectral::Quadrature::GaussRadauUpper,
                                 Spectral::Quadrature::Equiangular,
                                 i1_quadrature}});
      }
    }
  }
  {
    INFO("spheriical_shell");
    const Element<3> spherical_shell(element_id_3d, {},
                                     domain::topologies::spherical_shell);
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3, 2, 4}}}, spherical_shell, i1_basis,
                                  i1_quadrature) ==
              Mesh<3>{{{3, 2, 4}},
                      std::array{i1_basis, Spectral::Basis::SphericalHarmonic,
                                 Spectral::Basis::SphericalHarmonic},
                      std::array{i1_quadrature, Spectral::Quadrature::Gauss,
                                 Spectral::Quadrature::Equiangular}});
      }
    }
  }
  {
    INFO("full_sphere");
    const Block<3> full_sphere(nullptr, 0, {}, "",
                               domain::topologies::full_sphere);
    for (const auto& i1_basis :
         {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto& i1_quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        CHECK(create_initial_mesh({{{3, 2, 4}}}, full_sphere, element_id_3d,
                                  i1_basis, i1_quadrature) ==
              Mesh<3>{{{3, 2, 4}},
                      std::array{Spectral::Basis::ZernikeB3,
                                 Spectral::Basis::ZernikeB3,
                                 Spectral::Basis::ZernikeB3},
                      std::array{Spectral::Quadrature::GaussRadauUpper,
                                 Spectral::Quadrature::Gauss,
                                 Spectral::Quadrature::Equiangular}});
      }
    }
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      create_initial_mesh(
          {{{3, 4}}}, Block<2>{nullptr, 0, {}, "", domain::topologies::disk},
          ElementId<2>{0, {{SegmentId{1, 0}, SegmentId{0, 0}}}},
          Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto),
      Catch::Matchers::ContainsSubstring(
          "Splitting Topology::B2Radial is not yet supported"));
#endif  // SPECTRE_DEBUG
}
}  // namespace domain
