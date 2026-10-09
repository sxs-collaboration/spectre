// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace intrp {
namespace InterpolateOnSphericalShell_detail {
// Target points whose radial logical coordinates differ by at most this much
// from the first point of a group are interpolated together. This absorbs the
// roundoff in the logical coordinates of points at the same radius.
constexpr double radial_coordinate_tolerance = 1.0e-12;
}  // namespace InterpolateOnSphericalShell_detail

/*!
 * \brief Interpolates `vars` on a spherical-harmonic shell `mesh` to the
 * `target_points`, first in the radial dimension and then in the angular
 * dimensions.
 *
 * Consecutive target points whose radial logical coordinates agree with that of
 * the first point of the group to within a tolerance share the
 * radial interpolation, which leaves an interpolation on the sphere with
 * `intrp::Irregular<2>`. This is much cheaper than `intrp::Irregular<3>` when
 * the target points lie on a few spheres of constant radial logical coordinate.
 */
template <typename TagsList>
Variables<TagsList> interpolate_on_spherical_shell(
    const Variables<TagsList>& vars, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& target_points) {
  ASSERT((mesh.basis(0) == Spectral::Basis::Legendre or
          mesh.basis(0) == Spectral::Basis::Chebyshev) and
             mesh.basis(1) == Spectral::Basis::SphericalHarmonic and
             mesh.basis(2) == Spectral::Basis::SphericalHarmonic,
         "Expected a spherical-harmonic shell, but the mesh is " << mesh);
  ASSERT(vars.number_of_grid_points() == mesh.number_of_grid_points(),
         "The mesh has " << mesh.number_of_grid_points()
                         << " grid points but the variables have "
                         << vars.number_of_grid_points() << ".");
  const size_t number_of_target_points = get<0>(target_points).size();
  Variables<TagsList> result(number_of_target_points);
  const Mesh<1> radial_mesh = mesh.slice_through(0);
  const Mesh<2> angular_mesh = mesh.slice_away(0);
  size_t begin = 0;
  while (begin < number_of_target_points) {
    const double radial_coord = get<0>(target_points)[begin];
    size_t end = begin + 1;
    while (
        end < number_of_target_points and
        std::abs(get<0>(target_points)[end] - radial_coord) <=
            InterpolateOnSphericalShell_detail::radial_coordinate_tolerance) {
      ++end;
    }
    const size_t group_size = end - begin;
    // Interpolate radially to the sphere of the group
    const Variables<TagsList> vars_on_sphere = apply_matrices(
        std::array<Matrix, 3>{{Spectral::interpolation_matrix(
                                   radial_mesh, DataVector{1, radial_coord}),
                               Matrix{}, Matrix{}}},
        vars, mesh.extents());
    // Then on the sphere to the angular coordinates of the group
    tnsr::I<DataVector, 2, Frame::ElementLogical> angular_points{group_size};
    for (size_t d = 0; d < 2; ++d) {
      std::copy_n(target_points.get(d + 1).data() + begin, group_size,
                  angular_points.get(d).data());
    }
    const Variables<TagsList> result_on_sphere =
        intrp::Irregular<2>(angular_mesh, angular_points)
            .interpolate(vars_on_sphere);
    for (size_t component = 0;
         component < result.number_of_independent_components; ++component) {
      std::copy_n(result_on_sphere.data() + component * group_size, group_size,
                  result.data() + component * number_of_target_points + begin);
    }
    begin = end;
  }
  return result;
}
}  // namespace intrp
