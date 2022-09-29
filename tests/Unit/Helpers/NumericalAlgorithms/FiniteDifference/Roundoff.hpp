// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::fd::reconstruction {
/*!
 * Test that the reconstruction procedure produces positive results in
 * the presence of a very large jump which may cause roundoff error.
 */
template <typename F>
void test_positivity_with_roundoff(const size_t stencil_width,
                                   const F& invoke_recons) {
  const size_t num_fd_points = 4;
  const Mesh<1> mesh{num_fd_points, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  const auto logical_coords = logical_coordinates(mesh);

  std::vector<double> volume_vars(mesh.number_of_grid_points(), 0.0);
  DataVector var(volume_vars.data(), mesh.number_of_grid_points());

  const double small_value = 1.0e-16;
  const double large_value = 1.0e10;
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    const double x = get<0>(logical_coords)[i];
    var[i] = x > 0.0 ? x * large_value : small_value;
  }

  const size_t number_of_ghost_zones = (stencil_width - 1) / 2 + 1;
  DirectionMap<1, std::vector<double>> neighbor_data{};
  neighbor_data[Direction<1>::upper_xi()] =
      std::vector<double>(number_of_ghost_zones, large_value);
  neighbor_data[Direction<1>::lower_xi()] =
      std::vector<double>(number_of_ghost_zones, small_value);

  std::vector<double> reconstructed_upper_side_of_face_vars(
      mesh.number_of_grid_points() + 1);
  std::vector<double> reconstructed_lower_side_of_face_vars(
      mesh.number_of_grid_points() + 1);

  std::array<gsl::span<double>, 1> recons_upper_side_of_face{
      gsl::make_span(reconstructed_upper_side_of_face_vars.data(),
                     reconstructed_upper_side_of_face_vars.size())};
  std::array<gsl::span<double>, 1> recons_lower_side_of_face{
      gsl::make_span(reconstructed_lower_side_of_face_vars.data(),
                     reconstructed_lower_side_of_face_vars.size())};

  DirectionMap<1, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [direction, data] : neighbor_data) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  invoke_recons(make_not_null(&recons_upper_side_of_face),
                make_not_null(&recons_lower_side_of_face),
                gsl::make_span(volume_vars.data(), volume_vars.size()),
                ghost_cell_vars, mesh.extents(), 1);

  for (const auto reconstructed : reconstructed_upper_side_of_face_vars) {
    CHECK(reconstructed > 0.0);
  }
  for (const auto reconstructed : reconstructed_lower_side_of_face_vars) {
    CHECK(reconstructed > 0.0);
  }
}
}  // namespace TestHelpers::fd::reconstruction
