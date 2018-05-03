// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>

#include "DataStructures/Index.hpp"
#include "DataStructures/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test_extents_basis_and_quadrature(
    const Mesh<Dim>& mesh, const std::array<size_t, Dim>& extents,
    const std::array<Spectral::Basis, Dim>& basis,
    const std::array<Spectral::Quadrature, Dim>& quadrature) {
  CHECK(mesh.number_of_grid_points() ==
        std::accumulate(extents.begin(), extents.end(), size_t{1},
                        std::multiplies<size_t>()));
  CHECK(mesh.extents() == Index<Dim>{extents});
  CHECK(mesh.basis() == basis);
  CHECK(mesh.quadrature() == quadrature);
  for (size_t d = 0; d < Dim; d++) {
    CHECK(mesh.extents(d) == gsl::at(extents, d));
    CHECK(mesh.basis(d) == gsl::at(basis, d));
    CHECK(mesh.quadrature(d) == gsl::at(quadrature, d));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Mesh", "[DataStructures][Unit]") {
  SECTION("Uniform LGL mesh") {
    const Mesh<1> mesh1d_lgl{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
    test_extents_basis_and_quadrature(mesh1d_lgl, {{3}},
                                      {{Spectral::Basis::Legendre}},
                                      {{Spectral::Quadrature::GaussLobatto}});
    const Mesh<2> mesh2d_lgl{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
    test_extents_basis_and_quadrature(
        mesh2d_lgl, {{3, 3}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::GaussLobatto}});
    const Mesh<3> mesh3d_lgl{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
    test_extents_basis_and_quadrature(
        mesh3d_lgl, {{3, 3, 3}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
          Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::GaussLobatto}});
  }

  SECTION("Explicit choices per dimension") {
    const Mesh<1> mesh1d{{{2}},
                         {{Spectral::Basis::Legendre}},
                         {{Spectral::Quadrature::GaussLobatto}}};
    test_extents_basis_and_quadrature(mesh1d, {{2}},
                                      {{Spectral::Basis::Legendre}},
                                      {{Spectral::Quadrature::GaussLobatto}});
    const auto mesh1d_sliced = mesh1d.slice_away(0);
    test_extents_basis_and_quadrature(mesh1d_sliced, {}, {}, {});
    const Mesh<2> mesh2d{
        {{2, 3}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}};
    test_extents_basis_and_quadrature(
        mesh2d, {{2, 3}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}});
    const auto mesh2d_sliced_0 = mesh2d.slice_away(0);
    test_extents_basis_and_quadrature(mesh2d_sliced_0, {{3}},
                                      {{Spectral::Basis::Legendre}},
                                      {{Spectral::Quadrature::GaussLobatto}});
    const auto mesh2d_sliced_1 = mesh2d.slice_away(1);
    test_extents_basis_and_quadrature(mesh2d_sliced_1, {{2}},
                                      {{Spectral::Basis::Legendre}},
                                      {{Spectral::Quadrature::Gauss}});
    const Mesh<3> mesh3d{
        {{2, 3, 4}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
          Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
          Spectral::Quadrature::GaussLobatto}}};
    test_extents_basis_and_quadrature(
        mesh3d, {{2, 3, 4}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
          Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
          Spectral::Quadrature::GaussLobatto}});
    const auto mesh3d_sliced_0 = mesh3d.slice_away(0);
    test_extents_basis_and_quadrature(
        mesh3d_sliced_0, {{3, 4}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}});
    const auto mesh3d_sliced_1 = mesh3d.slice_away(1);
    test_extents_basis_and_quadrature(
        mesh3d_sliced_1, {{2, 4}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::GaussLobatto}});
    const auto mesh3d_sliced_2 = mesh3d.slice_away(2);
    test_extents_basis_and_quadrature(
        mesh3d_sliced_2, {{2, 3}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
        {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}});
  }

  SECTION("Equality") {
    CHECK(Mesh<1>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} ==
          Mesh<1>{{{3}},
                  {{Spectral::Basis::Legendre}},
                  {{Spectral::Quadrature::GaussLobatto}}});
    CHECK(Mesh<1>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} !=
          Mesh<1>{{{2}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(Mesh<1>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} !=
          Mesh<1>{{{3}},
                  {{Spectral::Basis::Legendre}},
                  {{Spectral::Quadrature::Gauss}}});
    CHECK(Mesh<2>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} ==
          Mesh<2>{{{3, 3}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(Mesh<2>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} ==
          Mesh<2>{{{3, 3}},
                  {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                  {{Spectral::Quadrature::GaussLobatto,
                    Spectral::Quadrature::GaussLobatto}}});
    CHECK(Mesh<2>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} !=
          Mesh<2>{{{3, 2}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(Mesh<2>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} !=
          Mesh<2>{{{3, 3}},
                  {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                  {{Spectral::Quadrature::Gauss,
                    Spectral::Quadrature::GaussLobatto}}});
    CHECK(Mesh<3>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} ==
          Mesh<3>{{{3, 3, 3}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(Mesh<3>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} ==
          Mesh<3>{{{3, 3, 3}},
                  {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                    Spectral::Basis::Legendre}},
                  {{Spectral::Quadrature::GaussLobatto,
                    Spectral::Quadrature::GaussLobatto,
                    Spectral::Quadrature::GaussLobatto}}});
    CHECK(Mesh<3>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} !=
          Mesh<3>{{{3, 2, 3}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(Mesh<3>{3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto} !=
          Mesh<3>{
              {{3, 3, 3}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                Spectral::Basis::Legendre}},
              {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
                Spectral::Quadrature::GaussLobatto}}});
  }
}
SPECTRE_TEST_CASE("Unit.Serialization.Mesh",
                  "[DataStructures][Unit][Serialization]") {
  test_serialization(Mesh<1>{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto});
  test_serialization(Mesh<2>{
      {{3, 2}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}});
  test_serialization(
      Mesh<3>{{{3, 2, 4}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                Spectral::Basis::Legendre}},
              {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
                Spectral::Quadrature::GaussLobatto}}});
}
