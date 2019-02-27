// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test_extents_basis_and_quadrature(
    const Mesh<Dim>& mesh, const std::array<size_t, Dim>& extents,
    const std::array<Spectral::Basis, Dim>& basis,
    const std::array<Spectral::Quadrature, Dim>& quadrature) {
  CAPTURE(Dim);
  CAPTURE(extents);
  CAPTURE(basis);
  CAPTURE(quadrature);
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
    CHECK(gsl::at(mesh.slices(), d) == mesh.slice_through(d));
  }
  CHECK(get_output(mesh) == get_output(extents));
  for (IndexIterator<Dim> index_it(mesh.extents()); index_it; ++index_it) {
    CAPTURE(*index_it);
    Index<Dim> index{};
    for (size_t d = 0; d < Dim; ++d) {
      index[d] = (*index_it)[d];
    }
    CHECK(index_it.collapsed_index() == mesh.storage_index(index));
  }
}

void test_uniform_lgl_mesh() noexcept {
  INFO("Uniform LGL mesh");
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
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
        Spectral::Quadrature::GaussLobatto}});
}

void test_explicit_choices_per_dimension() noexcept {
  INFO("Explicit choices per dimension");
  CHECK(Mesh<0>{}.slice_through() == Mesh<0>{});
  const Mesh<1> mesh1d{{{2}},
                       {{Spectral::Basis::Legendre}},
                       {{Spectral::Quadrature::GaussLobatto}}};
  test_extents_basis_and_quadrature(mesh1d, {{2}},
                                    {{Spectral::Basis::Legendre}},
                                    {{Spectral::Quadrature::GaussLobatto}});
  CHECK(mesh1d.slice_away(0) == Mesh<0>{});
  CHECK(mesh1d.slice_through() == Mesh<0>{});
  CHECK(mesh1d.slice_through(0) == mesh1d);

  const Mesh<2> mesh2d{
      {{2, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}};
  test_extents_basis_and_quadrature(
      mesh2d, {{2, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}});
  CHECK(mesh2d.slice_away(0) == Mesh<1>{3, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto});
  CHECK(mesh2d.slice_away(1) ==
        Mesh<1>{2, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
  CHECK(mesh2d.slice_through() == Mesh<0>{});
  CHECK(mesh2d.slice_through(0) == mesh2d.slice_away(1));
  CHECK(mesh2d.slice_through(1) == mesh2d.slice_away(0));
  CHECK(mesh2d.slice_through(0, 1) == mesh2d);
  CHECK(mesh2d.slice_through(1, 0) ==
        Mesh<2>{{{3, 2}},
                {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                {{Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::Gauss}}});

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
  CHECK(mesh3d.slice_away(0) ==
        Mesh<2>{{{3, 4}},
                {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                {{Spectral::Quadrature::Gauss,
                  Spectral::Quadrature::GaussLobatto}}});
  CHECK(mesh3d.slice_away(1) == Mesh<2>{{{2, 4}},
                                        Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto});
  CHECK(mesh3d.slice_away(2) ==
        Mesh<2>{{{2, 3}},
                {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                {{Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::Gauss}}});
  CHECK(mesh3d.slice_through() == Mesh<0>{});
  CHECK(mesh3d.slice_through(0) == Mesh<1>{2, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto});
  CHECK(mesh3d.slice_through(1) ==
        Mesh<1>{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
  CHECK(mesh3d.slice_through(2) == Mesh<1>{4, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto});
  CHECK(mesh3d.slice_through(0, 1) == mesh3d.slice_away(2));
  CHECK(mesh3d.slice_through(0, 2) == mesh3d.slice_away(1));
  CHECK(mesh3d.slice_through(1, 2) == mesh3d.slice_away(0));
  CHECK(mesh3d.slice_through(0, 1, 2) == mesh3d);
  CHECK(mesh3d.slice_through(2, 0, 1) ==
        Mesh<3>{{{4, 2, 3}},
                {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                  Spectral::Basis::Legendre}},
                {{Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::Gauss}}});
}

void test_equality() noexcept {
  INFO("Equality");
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
  CHECK(
      Mesh<3>{3, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto} !=
      Mesh<3>{{{3, 3, 3}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                Spectral::Basis::Legendre}},
              {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
                Spectral::Quadrature::GaussLobatto}}});
}

void test_serialization() noexcept {
  INFO("Serialization");
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

void test_tags() noexcept {
  INFO("Tags");
  CHECK(Tags::Mesh<1>::name() == "Mesh");
  CHECK(Tags::Mesh<2>::name() == "Mesh");
  CHECK(Tags::Mesh<3>::name() == "Mesh");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Mesh", "[Domain][Unit]") {
  test_uniform_lgl_mesh();
  test_explicit_choices_per_dimension();
  test_equality();
  test_serialization();
  test_tags();
}

// [[OutputRegex, Tried to slice through non-existing dimension]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.Mesh.SliceThroughNonExistingDimension", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Mesh<1> mesh1d{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  mesh1d.slice_through(1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Tried to slice away non-existing dimension]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Mesh.SliceAwayNonExistingDimension",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Mesh<1> mesh1d{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  mesh1d.slice_away(1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Dimensions to slice through contain duplicates]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.Mesh.SliceThroughDuplicateDimensions", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Mesh<3> mesh3d{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  mesh3d.slice_through(2, 1, 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
