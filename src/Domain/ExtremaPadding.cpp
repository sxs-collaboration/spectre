// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ExtremaPadding.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace domain {
namespace {
// Safety factor applied to the terms involving second derivatives, because
// these are only estimated from the values at the grid points.
constexpr double safety_factor = 2.0;

bool is_i1_basis(const Spectral::Basis basis) {
  return basis == Spectral::Basis::Legendre or
         basis == Spectral::Basis::Chebyshev;
}

bool is_s2_colatitude(const Mesh<3>& mesh, const size_t d) {
  return mesh.basis(d) == Spectral::Basis::SphericalHarmonic and
         mesh.quadrature(d) == Spectral::Quadrature::Gauss;
}

bool is_s2_longitude(const Mesh<3>& mesh, const size_t d) {
  return mesh.basis(d) == Spectral::Basis::SphericalHarmonic and
         mesh.quadrature(d) == Spectral::Quadrature::Equiangular;
}

// Padding from the variation of the function with `grid_point_values` along
// logical dimension `d`. Between neighboring grid points, the function
// exceeds the linear interpolant between them by at most h^2 max|f''| / 8,
// where h is the spacing in the logical coordinate and f'' is the second
// derivative with respect to it. The second derivative is estimated by second
// divided differences
double padding_along_dimension(const Mesh<3>& mesh,
                               const DataVector& grid_point_values,
                               const size_t d, const size_t stride) {
  const size_t extent = mesh.extents(d);
  if (extent < 2) {
    return 0.0;
  }
  const DataVector logical_coords =
      get<0>(logical_coordinates(mesh.slice_through(d)));

  const auto spacing = [&logical_coords](const size_t j) {
    return logical_coords[j + 1] - logical_coords[j];
  };

  // Largest first and second derivatives of the function with respect to the
  // logical coordinate, estimated by first and second divided differences. The
  // first derivative is only needed for the distance to the boundary below.
  double max_first_derivative = 0.0;
  double max_second_derivative = 0.0;
  for (size_t s = 0; s < grid_point_values.size(); ++s) {
    const size_t j = (s / stride) % extent;
    if (j + 1 == extent) {
      continue;
    }
    const double upper_slope =
        (grid_point_values[s + stride] - grid_point_values[s]) / spacing(j);
    max_first_derivative = std::max(max_first_derivative, abs(upper_slope));
    if (j > 0) {
      const double lower_slope =
          (grid_point_values[s] - grid_point_values[s - stride]) /
          spacing(j - 1);
      max_second_derivative = std::max(max_second_derivative,
                                       abs(2.0 * (upper_slope - lower_slope) /
                                           (spacing(j) + spacing(j - 1))));
    }
  }

  double max_spacing = 0.0;
  for (size_t j = 0; j + 1 < extent; ++j) {
    max_spacing = std::max(max_spacing, spacing(j));
  }
  double result =
      0.125 * safety_factor * max_second_derivative * square(max_spacing);
  if (is_s2_colatitude(mesh, d)) {
    const double pole_gap =
        2.0 * std::max(logical_coords[0],
                       std::numbers::pi - logical_coords[extent - 1]);
    result = std::max(result, 0.125 * safety_factor * max_second_derivative *
                                  square(pole_gap));
  } else if (not is_s2_longitude(mesh, d)) {
    // Logical distance from the outermost grid points to the element boundary
    // at -1 and 1. This term only contributes for quadratures (e.g. Gauss)
    // whose grid points do not include the endpoints. Over this distance the
    // function can exceed its value at the outermost grid point by at most
    // |f'| h + |f''| h^2 / 2
    const double distance_to_boundary =
        std::max(logical_coords[0] + 1.0, 1.0 - logical_coords[extent - 1]);
    result += max_first_derivative * distance_to_boundary +
              0.5 * safety_factor * max_second_derivative *
                  square(distance_to_boundary);
  }
  return result;
}

// Conservative padding for meshes not handled above: the sum over the
// logical dimensions of the largest change of `grid_point_values` between
// neighboring grid points.
double neighbor_difference_padding(const Mesh<3>& mesh,
                                   const DataVector& grid_point_values) {
  double result = 0.0;
  size_t stride = 1;
  for (size_t d = 0; d < 3; ++d) {
    const size_t extent = mesh.extents(d);
    double max_difference = 0.0;
    for (size_t s = 0; s < grid_point_values.size(); ++s) {
      if ((s / stride) % extent + 1 < extent) {
        max_difference =
            std::max(max_difference,
                     abs(grid_point_values[s + stride] - grid_point_values[s]));
      }
    }
    result += max_difference;
    stride *= extent;
  }
  return result;
}
}  // namespace

double extrema_padding(const Mesh<3>& mesh,
                       const DataVector& grid_point_values) {
  ASSERT(grid_point_values.size() == mesh.number_of_grid_points(),
         "The mesh has " << mesh.number_of_grid_points()
                         << " grid points but there are "
                         << grid_point_values.size() << " grid point values.");
  const bool all_i1 = is_i1_basis(mesh.basis(0)) and
                      is_i1_basis(mesh.basis(1)) and is_i1_basis(mesh.basis(2));
  const bool s2_shell = is_i1_basis(mesh.basis(0)) and
                        is_s2_colatitude(mesh, 1) and is_s2_longitude(mesh, 2);
  if (not(all_i1 or s2_shell)) {
    return neighbor_difference_padding(mesh, grid_point_values);
  }
  double result = 0.0;
  size_t stride = 1;
  for (size_t d = 0; d < 3; ++d) {
    result += padding_along_dimension(mesh, grid_point_values, d, stride);
    stride *= mesh.extents(d);
  }
  return result;
}
}  // namespace domain
