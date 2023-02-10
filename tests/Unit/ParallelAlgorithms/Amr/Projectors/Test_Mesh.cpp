// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "Utilities/Literals.hpp"

namespace {
void test_mesh_1d() {
  const auto legendre = Spectral::Basis::Legendre;
  const auto gauss_lobatto = Spectral::Quadrature::GaussLobatto;
  const auto refine = std::array{amr::Flag::IncreaseResolution};
  const auto coarsen = std::array{amr::Flag::DecreaseResolution};
  const auto split = std::array{amr::Flag::Split};
  const auto join = std::array{amr::Flag::Join};
  const auto stay = std::array{amr::Flag::DoNothing};
  Mesh<1> mesh_3{std::array{3_st}, legendre, gauss_lobatto};
  Mesh<1> mesh_4{std::array{4_st}, legendre, gauss_lobatto};
  CHECK(amr::projectors::mesh(mesh_3, refine) == mesh_4);
  CHECK(amr::projectors::mesh(mesh_4, coarsen) == mesh_3);
  CHECK(amr::projectors::mesh(mesh_3, join) == mesh_3);
  CHECK(amr::projectors::mesh(mesh_3, split) == mesh_3);
  CHECK(amr::projectors::mesh(mesh_3, stay) == mesh_3);

  CHECK(amr::projectors::parent_mesh(std::vector{mesh_3, mesh_4}) == mesh_4);
}

void test_mesh_2d() {
  const auto legendre = Spectral::Basis::Legendre;
  const auto gauss_lobatto = Spectral::Quadrature::GaussLobatto;
  const auto refine_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution};
  const auto refine_stay =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing};
  const auto refine_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution};
  const auto coarsen_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution};
  const auto coarsen_stay =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing};
  const auto coarsen_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution};
  Mesh<2> mesh_3_5{std::array{3_st, 5_st}, legendre, gauss_lobatto};
  Mesh<2> mesh_3_6{std::array{3_st, 6_st}, legendre, gauss_lobatto};
  Mesh<2> mesh_4_5{std::array{4_st, 5_st}, legendre, gauss_lobatto};
  Mesh<2> mesh_4_6{std::array{4_st, 6_st}, legendre, gauss_lobatto};
  CHECK(amr::projectors::mesh(mesh_3_5, refine_refine) == mesh_4_6);
  CHECK(amr::projectors::mesh(mesh_3_5, refine_stay) == mesh_4_5);
  CHECK(amr::projectors::mesh(mesh_3_6, refine_coarsen) == mesh_4_5);
  CHECK(amr::projectors::mesh(mesh_4_5, coarsen_refine) == mesh_3_6);
  CHECK(amr::projectors::mesh(mesh_4_6, coarsen_stay) == mesh_3_6);
  CHECK(amr::projectors::mesh(mesh_4_6, coarsen_coarsen) == mesh_3_5);

  CHECK(amr::projectors::parent_mesh(
            std::vector{mesh_3_5, mesh_4_5, mesh_3_6}) == mesh_4_6);
#ifdef SPECTRE_DEBUG
  Mesh<2> mesh_mismatch_basis{std::array{2_st, 4_st},
                              std::array{legendre, Spectral::Basis::Chebyshev},
                              std::array{gauss_lobatto, gauss_lobatto}};
  CHECK_THROWS_WITH(
      amr::projectors::parent_mesh(std::vector{mesh_3_5, mesh_mismatch_basis}),
      Catch::Contains("AMR does not currently support joining elements with "
                      "different quadratures or bases"));
  Mesh<2> mesh_mismatch_quadrature{
      std::array{2_st, 4_st}, std::array{legendre, legendre},
      std::array{Spectral::Quadrature::Gauss, gauss_lobatto}};
  CHECK_THROWS_WITH(
      amr::projectors::parent_mesh(
          std::vector{mesh_3_5, mesh_mismatch_quadrature}),
      Catch::Contains("AMR does not currently support joining elements with "
                      "different quadratures or bases"));
#endif
}

void test_mesh_3d() {
  const auto legendre = Spectral::Basis::Legendre;
  const auto gauss_lobatto = Spectral::Quadrature::GaussLobatto;
  const auto refine_refine_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto refine_stay_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::IncreaseResolution};
  const auto refine_coarsen_refine =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto coarsen_refine_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto coarsen_stay_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::IncreaseResolution};
  const auto coarsen_coarsen_refine =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::IncreaseResolution};
  const auto refine_refine_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::DecreaseResolution};
  const auto refine_stay_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::DecreaseResolution};
  const auto refine_coarsen_coarsen =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::DecreaseResolution};
  const auto coarsen_refine_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution,
                 amr::Flag::DecreaseResolution};
  const auto coarsen_stay_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing,
                 amr::Flag::DecreaseResolution};
  const auto coarsen_coarsen_coarsen =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution,
                 amr::Flag::DecreaseResolution};
  Mesh<3> mesh_3_5_7{std::array{3_st, 5_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_3_6_7{std::array{3_st, 6_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_5_7{std::array{4_st, 5_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_6_7{std::array{4_st, 6_st, 7_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_3_5_8{std::array{3_st, 5_st, 8_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_3_6_8{std::array{3_st, 6_st, 8_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_5_8{std::array{4_st, 5_st, 8_st}, legendre, gauss_lobatto};
  Mesh<3> mesh_4_6_8{std::array{4_st, 6_st, 8_st}, legendre, gauss_lobatto};
  CHECK(amr::projectors::mesh(mesh_3_5_7, refine_refine_refine) == mesh_4_6_8);
  CHECK(amr::projectors::mesh(mesh_3_5_7, refine_stay_refine) == mesh_4_5_8);
  CHECK(amr::projectors::mesh(mesh_3_6_7, refine_coarsen_refine) == mesh_4_5_8);
  CHECK(amr::projectors::mesh(mesh_4_5_7, coarsen_refine_refine) == mesh_3_6_8);
  CHECK(amr::projectors::mesh(mesh_4_6_7, coarsen_stay_refine) == mesh_3_6_8);
  CHECK(amr::projectors::mesh(mesh_4_6_7, coarsen_coarsen_refine) ==
        mesh_3_5_8);
  CHECK(amr::projectors::mesh(mesh_3_5_8, refine_refine_coarsen) == mesh_4_6_7);
  CHECK(amr::projectors::mesh(mesh_3_5_8, refine_stay_coarsen) == mesh_4_5_7);
  CHECK(amr::projectors::mesh(mesh_3_6_8, refine_coarsen_coarsen) ==
        mesh_4_5_7);
  CHECK(amr::projectors::mesh(mesh_4_5_8, coarsen_refine_coarsen) ==
        mesh_3_6_7);
  CHECK(amr::projectors::mesh(mesh_4_6_8, coarsen_stay_coarsen) == mesh_3_6_7);
  CHECK(amr::projectors::mesh(mesh_4_6_8, coarsen_coarsen_coarsen) ==
        mesh_3_5_7);

  CHECK(amr::projectors::parent_mesh(
            std::vector{mesh_3_5_7, mesh_4_5_8, mesh_3_6_7}) == mesh_4_6_8);
#ifdef SPECTRE_DEBUG
  Mesh<3> mesh_mismatch_basis{
      std::array{2_st, 4_st, 3_st},
      std::array{legendre, Spectral::Basis::Chebyshev, legendre},
      std::array{gauss_lobatto, gauss_lobatto, gauss_lobatto}};
  CHECK_THROWS_WITH(
      amr::projectors::parent_mesh(
          std::vector{mesh_3_5_7, mesh_mismatch_basis}),
      Catch::Contains("AMR does not currently support joining elements with "
                      "different quadratures or bases"));
  Mesh<3> mesh_mismatch_quadrature{
      std::array{2_st, 4_st, 3_st}, std::array{legendre, legendre, legendre},
      std::array{Spectral::Quadrature::Gauss, gauss_lobatto, gauss_lobatto}};
  CHECK_THROWS_WITH(
      amr::projectors::parent_mesh(
          std::vector{mesh_3_5_7, mesh_mismatch_quadrature}),
      Catch::Contains("AMR does not currently support joining elements with "
                      "different quadratures or bases"));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.Mesh", "[ParallelAlgorithms][Unit]") {
  test_mesh_1d();
  test_mesh_2d();
  test_mesh_3d();
}
