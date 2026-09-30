// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
// Evaluates the `Mesh` functions that can be called in Kokkos kernels
constexpr size_t number_of_device_values = 22;
KOKKOS_FUNCTION std::array<size_t, number_of_device_values>
evaluate_mesh_functions(const Mesh<3>& mesh, const Mesh<3>& b3_mesh,
                        const Mesh<2>& b2_mesh) {
  const Mesh<3> other_mesh{std::array<size_t, 3>{{3, 4, 6}}, mesh.basis(),
                           mesh.quadrature()};
  const Mesh<3> copied_mesh = mesh;
  const Mesh<2> isotropic_mesh{4, Spectral::Basis::Legendre,
                               Spectral::Quadrature::Gauss};
  return {mesh.number_of_grid_points(),
          mesh.extents(0),
          mesh.extents(1),
          mesh.extents()[2],
          static_cast<size_t>(mesh.basis(0)),
          static_cast<size_t>(mesh.basis()[1]),
          static_cast<size_t>(mesh.quadrature(0)),
          static_cast<size_t>(mesh.quadrature()[1]),
          mesh.storage_index(Index<3>{1, 2, 3}),
          mesh.slice_away(1).number_of_grid_points(),
          mesh.slice_through(2, 0).extents(0),
          mesh.on_interface(0).number_of_grid_points(),
          mesh.slices()[2].extents(0),
          static_cast<size_t>(is_isotropic(mesh)),
          static_cast<size_t>(is_isotropic(isotropic_mesh)),
          static_cast<size_t>(mesh == copied_mesh),
          static_cast<size_t>(mesh != other_mesh),
          other_mesh.number_of_grid_points(),
          static_cast<size_t>(b3_mesh.on_interface(0).basis(0)),
          static_cast<size_t>(b3_mesh.on_interface(0).quadrature(1)),
          static_cast<size_t>(b2_mesh.on_interface(0).basis(0)),
          static_cast<size_t>(mesh == other_mesh)};
}

// The `Mesh` functions give the same results in a Kokkos kernel as on the host
void test_on_device() {
  const Mesh<3> mesh{
      std::array<size_t, 3>{{3, 4, 5}},
      std::array<Spectral::Basis, 3>{{Spectral::Basis::Legendre,
                                      Spectral::Basis::Chebyshev,
                                      Spectral::Basis::Legendre}},
      std::array<Spectral::Quadrature, 3>{
          {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
           Spectral::Quadrature::GaussLobatto}}};
  const Mesh<3> b3_mesh{
      std::array<size_t, 3>{{4, 5, 9}},
      make_array<3>(Spectral::Basis::ZernikeB3),
      std::array<Spectral::Quadrature, 3>{
          {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
           Spectral::Quadrature::Equiangular}}};
  const Mesh<2> b2_mesh{std::array<size_t, 2>{{4, 7}},
                        make_array<2>(Spectral::Basis::ZernikeB2),
                        std::array<Spectral::Quadrature, 2>{
                            {Spectral::Quadrature::GaussRadauUpper,
                             Spectral::Quadrature::Equiangular}}};
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return evaluate_mesh_functions(mesh, b3_mesh, b2_mesh);
  });
  const auto expected_values = evaluate_mesh_functions(mesh, b3_mesh, b2_mesh);
  // Spot-check some of the values
  CHECK(expected_values[0] == 60);
  CHECK(expected_values[8] == 1 + 3 * (2 + 4 * 3));
  CHECK(expected_values[16] == 1);
  CHECK(expected_values[18] ==
        static_cast<size_t>(Spectral::Basis::SphericalHarmonic));
  CHECK(expected_values[20] == static_cast<size_t>(Spectral::Basis::Fourier));
  CHECK(expected_values[21] == 0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Mesh.Kokkos",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_on_device();
}
