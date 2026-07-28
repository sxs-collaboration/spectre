// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

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
  CHECK(get_output(mesh) ==
        std::string{MakeString{} << '[' << get_output(extents) << ','
                                 << get_output(mesh.basis()) << ','
                                 << get_output(mesh.quadrature()) << ']'});
  for (IndexIterator<Dim> index_it(mesh.extents()); index_it; ++index_it) {
    CAPTURE(*index_it);
    Index<Dim> index{};
    for (size_t d = 0; d < Dim; ++d) {
      index[d] = (*index_it)[d];
    }
    CHECK(index_it.collapsed_index() == mesh.storage_index(index));
  }
}

void test_uniform_lgl_mesh() {
  INFO("Uniform LGL mesh");
  CHECK(is_isotropic(Mesh<0>{}));
  const Mesh<1> mesh1d_lgl{3, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  test_extents_basis_and_quadrature(mesh1d_lgl, {{3}},
                                    {{Spectral::Basis::Legendre}},
                                    {{Spectral::Quadrature::GaussLobatto}});
  CHECK(is_isotropic(mesh1d_lgl));
  const Mesh<2> mesh2d_lgl{3, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  test_extents_basis_and_quadrature(
      mesh2d_lgl, {{3, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::GaussLobatto,
        Spectral::Quadrature::GaussLobatto}});
  CHECK(is_isotropic(mesh2d_lgl));
  const Mesh<3> mesh3d_lgl{3, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  test_extents_basis_and_quadrature(
      mesh3d_lgl, {{3, 3, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
        Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
        Spectral::Quadrature::GaussLobatto}});
  CHECK(is_isotropic(mesh3d_lgl));
}

void test_explicit_choices_per_dimension() {
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

void test_interface_mesh() {
  INFO("interface mesh");
  // Non-Bn meshes: on_interface == slice_away
  const Mesh<3> mesh3d{
      {{2, 3, 4}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
        Spectral::Basis::Fourier}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
        Spectral::Quadrature::Equiangular}}};
  CHECK(mesh3d.on_interface(0) == mesh3d.slice_away(0));
  CHECK(mesh3d.on_interface(1) == mesh3d.slice_away(1));
  CHECK(mesh3d.on_interface(2) == mesh3d.slice_away(2));

  // B3 ball: slicing away the radial dimension converts ZernikeB3 angular bases
  // to SphericalHarmonic.
  const Mesh<3> b3_mesh{
      {{4, 5, 9}},
      make_array<3>(Spectral::Basis::ZernikeB3),
      {{Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
  CHECK(b3_mesh.on_interface(0) ==
        Mesh<2>({{5, 9}},
                std::array{Spectral::Basis::SphericalHarmonic,
                           Spectral::Basis::SphericalHarmonic},
                std::array{Spectral::Quadrature::Gauss,
                           Spectral::Quadrature::Equiangular}));

  // B2 disk: slicing away the radial dimension converts ZernikeB2 to Fourier.
  const Mesh<2> b2_mesh{{{3, 7}},
                        make_array<2>(Spectral::Basis::ZernikeB2),
                        {{Spectral::Quadrature::GaussRadauUpper,
                          Spectral::Quadrature::Equiangular}}};
  CHECK(b2_mesh.on_interface(0) == Mesh<1>{7, Spectral::Basis::Fourier,
                                           Spectral::Quadrature::Equiangular});

  // B2 cylinder (mixed bases): only ZernikeB2 dims are converted.
  const Mesh<3> b2_cyl_mesh{
      {{3, 7, 5}},
      {{Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
        Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::GaussRadauUpper,
        Spectral::Quadrature::Equiangular,
        Spectral::Quadrature::GaussLobatto}}};
  CHECK(b2_cyl_mesh.on_interface(0) ==
        Mesh<2>({{7, 5}},
                std::array{Spectral::Basis::Fourier, Spectral::Basis::Legendre},
                std::array{Spectral::Quadrature::Equiangular,
                           Spectral::Quadrature::GaussLobatto}));

  // B2 cylinder (reoriented: radial is last dim). Arises in Pill blocks where
  // the cylinder axis is not aligned with the block's first logical direction.
  const Mesh<3> b2_cyl_rev_mesh{
      {{5, 7, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::ZernikeB2,
        Spectral::Basis::ZernikeB2}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Equiangular,
        Spectral::Quadrature::GaussRadauUpper}}};
  CHECK(b2_cyl_rev_mesh.on_interface(2) ==
        Mesh<2>({{5, 7}},
                std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier},
                std::array{Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::Equiangular}));

#ifdef SPECTRE_DEBUG
  // B3: wrong Dim
  CHECK_THROWS_WITH((Mesh<2>{{{3, 7}},
                             make_array<2>(Spectral::Basis::ZernikeB3),
                             {{Spectral::Quadrature::GaussRadauUpper,
                               Spectral::Quadrature::Gauss}}}
                         .on_interface(0)),
                    Catch::Matchers::ContainsSubstring(
                        "ZernikeB3 should only be used on 3D meshes"));
  // B3: slicing an angular direction
  CHECK_THROWS_WITH(
      (Mesh<3>{
          {{4, 5, 9}},
          make_array<3>(Spectral::Basis::ZernikeB3),
          {{Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
            Spectral::Quadrature::Equiangular}}}
           .on_interface(1)),
      Catch::Matchers::ContainsSubstring(
          "A B3 ball's outer boundary is at the upper end of the radial "
          "dimension"));
  // B3: radial dimension does not use GaussRadauUpper
  CHECK_THROWS_WITH(
      (Mesh<3>{{{4, 5, 9}},
               make_array<3>(Spectral::Basis::ZernikeB3),
               {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
                 Spectral::Quadrature::Equiangular}}}
           .on_interface(0)),
      Catch::Matchers::ContainsSubstring(
          "A B3 ball's outer boundary is at the upper end of the radial "
          "dimension"));
  // B3: non-isotropic bases
  CHECK_THROWS_WITH(
      (Mesh<3>{
          {{4, 5, 9}},
          {{Spectral::Basis::ZernikeB3, Spectral::Basis::Legendre,
            Spectral::Basis::Legendre}},
          {{Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
            Spectral::Quadrature::GaussLobatto}}}
           .on_interface(0)),
      Catch::Matchers::ContainsSubstring(
          "Mesh is using ZernikeB3 basis in a nonisotropic manner"));
  // B2: wrong Dim
  CHECK_THROWS_WITH((Mesh<1>{3, Spectral::Basis::ZernikeB2,
                             Spectral::Quadrature::GaussRadauUpper}
                         .on_interface(0)),
                    Catch::Matchers::ContainsSubstring(
                        "ZernikeB2 should only be used on 2D or 3D meshes"));
  // B2: slicing an angular direction
  CHECK_THROWS_WITH(
      (Mesh<2>{{{3, 7}},
               make_array<2>(Spectral::Basis::ZernikeB2),
               {{Spectral::Quadrature::GaussRadauUpper,
                 Spectral::Quadrature::Equiangular}}}
           .on_interface(1)),
      Catch::Matchers::ContainsSubstring(
          "A B2 disk's outer boundary is at the upper end of the radial "
          "dimension"));
  // B2: radial dimension does not use GaussRadauUpper
  CHECK_THROWS_WITH(
      (Mesh<2>{
          {{3, 7}},
          make_array<2>(Spectral::Basis::ZernikeB2),
          {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}}
           .on_interface(0)),
      Catch::Matchers::ContainsSubstring(
          "A B2 disk's outer boundary is at the upper end of the radial "
          "dimension"));
  // B2: no ZernikeB2 angular basis in face after slicing radial
  CHECK_THROWS_WITH(
      (Mesh<2>{{{3, 7}},
               {{Spectral::Basis::ZernikeB2, Spectral::Basis::Legendre}},
               {{Spectral::Quadrature::GaussRadauUpper,
                 Spectral::Quadrature::Gauss}}}
           .on_interface(0)),
      Catch::Matchers::ContainsSubstring(
          "Expected exactly one ZernikeB2 angular basis remaining in the "
          "face"));
#endif  // SPECTRE_DEBUG
}

void test_equality() {
  INFO("Equality");
  // For a zero-d Mesh any isotropic arguments are ignored...
  CHECK(Mesh<0>{4, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto} ==
        Mesh<0>{2, Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss});
  CHECK(Mesh<0>{4, Spectral::Basis::Chebyshev,
                Spectral::Quadrature::GaussLobatto} ==
        Mesh<0>{{{}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
  CHECK(Mesh<0>{4, Spectral::Basis::Chebyshev,
                Spectral::Quadrature::GaussLobatto} ==
        Mesh<0>{{{}}, {{}}, {{}}});
  CHECK(Mesh<0>{{{}},
                Spectral::Basis::Chebyshev,
                Spectral::Quadrature::GaussLobatto} ==
        Mesh<0>{{{}}, {{}}, {{}}});
  CHECK(Mesh<0>{{{}},
                Spectral::Basis::Chebyshev,
                Spectral::Quadrature::GaussLobatto} ==
        Mesh<0>{{{}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
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

void test_serialization() {
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
  // Mesh should always be 6 bytes.
  constexpr size_t expected_1d_mesh = 2;
  constexpr size_t expected_2d_mesh = 4;
  constexpr size_t expected_3d_mesh = 6;
  static_assert(sizeof(Mesh<1>) == expected_1d_mesh);
  static_assert(sizeof(Mesh<2>) == expected_2d_mesh);
  static_assert(sizeof(Mesh<3>) == expected_3d_mesh);
  CHECK(size_of_object_in_bytes(Mesh<1>{3, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto}) ==
        expected_1d_mesh);
  CHECK(size_of_object_in_bytes(Mesh<2>{3, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto}) ==
        expected_2d_mesh);
  CHECK(size_of_object_in_bytes(Mesh<3>{3, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto}) ==
        expected_3d_mesh);
}

template <size_t Dim>
void test_option_parsing() {
  INFO("Option Parsing creation");

  const std::array<Spectral::Basis, 3> bases{Spectral::Basis::Chebyshev,
                                             Spectral::Basis::Legendre,
                                             Spectral::Basis::FiniteDifference};
  const std::array<Spectral::Quadrature, 4> quadratures{
      Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto,
      Spectral::Quadrature::CellCentered, Spectral::Quadrature::FaceCentered};

  for (size_t extent = 0; extent < 12; extent++) {  // extents
    for (const auto basis : bases) {                // bases
      for (const auto quadrature : quadratures) {   // quadratures
        std::stringstream creation_string;
        creation_string << "Extents: " << extent << "\nBasis: " << basis
                        << "\nQuadrature: " << quadrature;
        const auto mesh =
            TestHelpers::test_creation<Mesh<Dim>>(creation_string.str());
        CHECK(mesh == Mesh<Dim>{extent, basis, quadrature});
      }
    }
  }
}

void test_is_isotropic() {
  // Test non-isotropic meshes.
  CHECK_FALSE(is_isotropic(
      Mesh<2>{{{3, 2}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
              {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}}}));
  CHECK_FALSE(is_isotropic(
      Mesh<2>{{{3, 3}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}},
              {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}}}));
  CHECK_FALSE(is_isotropic(Mesh<2>{
      {{3, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}}));

  CHECK_FALSE(is_isotropic(
      Mesh<3>{{{3, 2}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
              {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}}}));
  CHECK_FALSE(is_isotropic(
      Mesh<3>{{{3, 3}},
              {{Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}},
              {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}}}));
  CHECK_FALSE(is_isotropic(Mesh<3>{
      {{3, 3}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}}}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Mesh",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_uniform_lgl_mesh();
  test_explicit_choices_per_dimension();
  test_interface_mesh();
  test_equality();
  test_serialization();
  test_option_parsing<1>();
  test_option_parsing<2>();
  test_option_parsing<3>();
  test_is_isotropic();

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      (Mesh<1>{2, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}
           .slice_through(1)),
      Catch::Matchers::ContainsSubstring(
          "Tried to slice through non-existing dimension"));
  CHECK_THROWS_WITH(
      (Mesh<1>{2, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}
           .slice_away(1)),
      Catch::Matchers::ContainsSubstring(
          "Tried to slice away non-existing dimension"));
  CHECK_THROWS_WITH(
      (Mesh<3>{2, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}
           .slice_through(2, 1, 1)),
      Catch::Matchers::ContainsSubstring(
          "Dimensions to slice through contain duplicates"));
#endif
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<Mesh<3>>("Extents: 5\n"
                                           "Basis: invalidBasis\n"
                                           "Quadrature: Gauss")),
      Catch::Matchers::ContainsSubstring(
          "Failed to convert \"invalidBasis\" to Spectral::Basis."));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<Mesh<3>>("Extents: 5\n"
                                           "Basis: Chebyshev\n"
                                           "Quadrature: invalidQuadrature")),
      Catch::Matchers::ContainsSubstring(
          "Failed to convert \"invalidQuadrature\" to Spectral::Quadrature."));
}
