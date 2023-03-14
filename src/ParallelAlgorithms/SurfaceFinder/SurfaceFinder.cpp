// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/SurfaceFinder/SurfaceFinder.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace SurfaceFinder {
namespace {
// Wrapping for the interpolator for Toms748 rootfind.
struct RayInterpolant {
 public:
  double operator()(double x) const {
    std::array<Matrix, 1> interpolation_matrices{
        Spectral::interpolation_matrix(mesh, x)};
    return apply_matrices(interpolation_matrices, values, mesh.extents())[0];
  }

  const DataVector& values{};
  const Mesh<1>& mesh{};
};
}  // namespace

std::vector<std::optional<double>> find_radial_surface(
    const Scalar<DataVector>& data, const double target, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& angular_coords,
    const double relative_tolerance, const double absolute_tolerance) {
  const size_t num_rays = angular_coords[0].size();
  const DataVector subtracted_data = get(data) - target;
  // The third Matrix in this array is not initialised so it is skipped by
  // apply_matrices()
  std::array<Matrix, 3> interpolation_matrices;
  std::vector<std::optional<double>> result(num_rays, std::nullopt);
  const double ray_size = mesh.extents(2);
  DataVector interpolated_data(ray_size);
  const auto xi_mesh = mesh.slice_through(0);
  const auto eta_mesh = mesh.slice_through(1);

  for (size_t i = 0; i < num_rays; i++) {
    // Potential speed-up: Currently, the interpolation is done one ray at a
    // time (i.e. the interpolation matrices are built to interpolate to one
    // ray, and then applied to the data sequentially). Instead one could use
    // something like irregular interpolant to do this all at once.
    interpolation_matrices[0] =
        Spectral::interpolation_matrix(xi_mesh, get<0>(angular_coords)[i]);
    interpolation_matrices[1] =
        Spectral::interpolation_matrix(eta_mesh, get<1>(angular_coords)[i]);

    // Interpolate data onto the ray.
    apply_matrices(make_not_null(&interpolated_data), interpolation_matrices,
                   subtracted_data, mesh.extents());
    const RayInterpolant data_interpolator{interpolated_data,
                                           mesh.slice_through(2)};

    // Perform root-find only if the element brackets a root.
    const double lower_radial_bound =
        mesh.quadrature(2) == Spectral::Quadrature::GaussLobatto
            ? interpolated_data[0]
            : data_interpolator(-1.);
    const double upper_radial_bound =
        mesh.quadrature(2) == Spectral::Quadrature::GaussLobatto
            ? interpolated_data[mesh.extents(2) - 1]
            : data_interpolator(1.);
    if (std::signbit(lower_radial_bound) != std::signbit(upper_radial_bound)) {
      result[i] = RootFinder::toms748(data_interpolator, -1., 1.,
                                      lower_radial_bound, upper_radial_bound,
                                      relative_tolerance, absolute_tolerance);
    }
  }
  return result;
}
}  // namespace SurfaceFinder
