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
#include "Evolution/DgSubcell/SliceData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace TestHelpers::fd::reconstruction {
/*!
 * \brief Test that the reconstruction procedure can exactly reconstruct
 * polynomials of degree equal to the degree of the basis.
 *
 * For example, for a minmod, MC, or other second-order TVD limiter, we must be
 * able to reconstruct linear functions exactly. For a third-order WENO scheme,
 * we must be able to reconstruct quadratic functions exactly. This is done
 * by setting up the function
 *
 * \f{align}{
 *  u=\sum_{i=1}^p x^p + y^p + z^p
 * \f}
 *
 * where \f$(x,y,z)=(\xi,\eta+4,\zeta+8)\f$ and \f$\xi,\eta,\zeta\f$ are the
 * logical coordinates. The translation is done to catch subtle bugs that may
 * arise from indexing in the wrong dimension.
 */
template <size_t Dim, typename F>
void test_reconstruction_is_exact_if_in_basis(const size_t max_degree,
                                              const size_t points_per_dimension,
                                              const size_t stencil_width,
                                              const F& invoke_recons) {
  const size_t number_of_vars = 2;  // arbitrary, 2 is "cheap but not trivial"

  const Mesh<Dim> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  // Compute polynomial on cell centers in FD cluster of points
  const auto set_polynomial = [max_degree](
                                  const gsl::not_null<DataVector*> var1_ptr,
                                  const gsl::not_null<DataVector*> var2_ptr,
                                  const auto& local_logical_coords) {
    *var1_ptr = 0.0;
    *var2_ptr = 100.0;  // some constant offset to distinguish the var values
    for (size_t degree = 1; degree <= max_degree; ++degree) {
      for (size_t i = 0; i < Dim; ++i) {
        *var1_ptr += pow(local_logical_coords.get(i), degree);
        if (number_of_vars == 2) {
          *var2_ptr += pow(local_logical_coords.get(i), degree);
        }
      }
    }
  };
  std::vector<double> volume_vars(mesh.number_of_grid_points() * number_of_vars,
                                  0.0);
  DataVector var1(volume_vars.data(), mesh.number_of_grid_points());
  DataVector var2(volume_vars.data() + mesh.number_of_grid_points(),  // NOLINT
                  mesh.number_of_grid_points());
  set_polynomial(&var1, &var2, logical_coords);

  // Compute the polynomial at the cell center for the neighbor data that we
  // "received".
  //
  // We do this by computing the solution in our entire neighbor, then using
  // slice_data to get the subset of points that are needed.
  DirectionMap<Dim, std::vector<double>> neighbor_data{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    std::vector<double> neighbor_vars(
        mesh.number_of_grid_points() * number_of_vars, 0.0);
    DataVector neighbor_var1(neighbor_vars.data(),
                             mesh.number_of_grid_points());
    DataVector neighbor_var2(
        neighbor_vars.data() + mesh.number_of_grid_points(),  // NOLINT
        mesh.number_of_grid_points());
    set_polynomial(&neighbor_var1, &neighbor_var2, neighbor_logical_coords);

    DirectionMap<Dim, bool> directions_to_slice{};
    directions_to_slice[direction.opposite()] = true;
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        mesh.extents(), (stencil_width - 1) / 2 + 1, directions_to_slice);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    neighbor_data[direction] = sliced_data.at(direction.opposite());
  }

  // Note: reconstructed_num_pts assumes isotropic extents
  const size_t reconstructed_num_pts =
      (mesh.extents(0) + 1) * mesh.slice_away(0).number_of_grid_points();
  std::array<std::vector<double>, Dim> reconstructed_upper_side_of_face_vars =
      make_array<Dim>(
          std::vector<double>(reconstructed_num_pts * number_of_vars));
  std::array<std::vector<double>, Dim> reconstructed_lower_side_of_face_vars =
      make_array<Dim>(
          std::vector<double>(reconstructed_num_pts * number_of_vars));

  std::array<gsl::span<double>, Dim> recons_upper_side_of_face{};
  std::array<gsl::span<double>, Dim> recons_lower_side_of_face{};
  for (size_t i = 0; i < Dim; ++i) {
    auto& upper = gsl::at(reconstructed_upper_side_of_face_vars, i);
    auto& lower = gsl::at(reconstructed_lower_side_of_face_vars, i);
    gsl::at(recons_upper_side_of_face, i) =
        gsl::make_span(upper.data(), upper.size());
    gsl::at(recons_lower_side_of_face, i) =
        gsl::make_span(lower.data(), lower.size());
  }

  DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [direction, data] : neighbor_data) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  invoke_recons(make_not_null(&recons_upper_side_of_face),
                make_not_null(&recons_lower_side_of_face),
                gsl::make_span(volume_vars.data(), volume_vars.size()),
                ghost_cell_vars, mesh.extents(), number_of_vars);

  for (size_t dim = 0; dim < Dim; ++dim) {
    CAPTURE(dim);
    // Since the reconstruction is to the same physical points the values should
    // be the same if the polynomial is representable in the basis of the
    // reconstructor.

    CHECK_ITERABLE_APPROX(gsl::at(reconstructed_upper_side_of_face_vars, dim),
                          gsl::at(reconstructed_lower_side_of_face_vars, dim));

    // Compare to analytic solution on the faces.
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);
    for (size_t i = 1; i < Dim; ++i) {
      logical_coords_face_centered.get(i) =
          logical_coords_face_centered.get(i) + 4.0 * i;
    }

    std::vector<double> expected_volume_vars(
        face_centered_mesh.number_of_grid_points() * number_of_vars, 0.0);
    DataVector expected_var1(expected_volume_vars.data(),
                             face_centered_mesh.number_of_grid_points());
    DataVector expected_var2(
        expected_volume_vars.data() +
            face_centered_mesh.number_of_grid_points(),  // NOLINT
        face_centered_mesh.number_of_grid_points());
    set_polynomial(&expected_var1, &expected_var2,
                   logical_coords_face_centered);
    CHECK_ITERABLE_APPROX(gsl::at(reconstructed_upper_side_of_face_vars, dim),
                          expected_volume_vars);
  }
}
}  // namespace TestHelpers::fd::reconstruction
