// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/SurfaceFinder/SurfaceFinder.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace SurfaceFinder {
namespace {
// Wrapping for the interpolator for Toms748 rootfind.
struct RayInterpolant {
 public:
  double operator()(const double x) const {
    const intrp::Irregular<1> interpolant{
        mesh,
        tnsr::I<DataVector, 1, Frame::ElementLogical>{DataVector{1_st, x}}};
    return interpolant.interpolate(values)[0];
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const DataVector& values;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Mesh<1>& mesh;
};
}  // namespace

std::vector<std::optional<double>> find_radial_surface(
    const Scalar<DataVector>& data, const double target, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& angular_coords,
    const double relative_tolerance, const double absolute_tolerance) {
  const size_t num_rays = angular_coords[0].size();
  std::vector<std::optional<double>> result(num_rays, std::nullopt);

  Variables<tmpl::list<::Tags::TempI<0, 3, Frame::ElementLogical>,
                       ::Tags::TempScalar<0>>>
      temp_vars_on_rays{mesh.extents(2) * num_rays};
  DataVector& data_on_rays = get(get<::Tags::TempScalar<0>>(temp_vars_on_rays));
  {
    const DataVector& zeta_logical_coords =
        Spectral::collocation_points(mesh.slice_through(2));
    tnsr::I<DataVector, 3, Frame::ElementLogical>& target_coords =
        get<::Tags::TempI<0, 3, Frame::ElementLogical>>(temp_vars_on_rays);
    for (size_t i = 0; i < num_rays; i++) {
      DataVector view{&get<0>(target_coords)[i * mesh.extents(2)],
                      mesh.extents(2)};
      view = get<0>(angular_coords)[i];

      view.set_data_ref(&get<1>(target_coords)[i * mesh.extents(2)],
                        mesh.extents(2));
      view = get<1>(angular_coords)[i];

      view.set_data_ref(&get<2>(target_coords)[i * mesh.extents(2)],
                        mesh.extents(2));
      view = zeta_logical_coords;
    }
    // We root find the function `data - target`, so interpolate that to the
    // rays.
    intrp::Irregular<3>{mesh, target_coords}.interpolate(&data_on_rays,
                                                         get(data) - target);
  }

  for (size_t i = 0; i < num_rays; i++) {
    const DataVector data_on_ray{&data_on_rays[i * mesh.extents(2)],
                                 mesh.extents(2)};
    const RayInterpolant data_interpolator{data_on_ray, mesh.slice_through(2)};

    // Perform root-find only if the element brackets a root.
    const double lower_radial_bound =
        mesh.quadrature(2) == Spectral::Quadrature::GaussLobatto
            ? data_on_ray[0]
            : data_interpolator(-1.);
    const double upper_radial_bound =
        mesh.quadrature(2) == Spectral::Quadrature::GaussLobatto
            ? data_on_ray[mesh.extents(2) - 1]
            : data_interpolator(1.);
    if (std::signbit(lower_radial_bound) != std::signbit(upper_radial_bound)) {
      result[i] = RootFinder::toms748(data_interpolator, -1., 1.,
                                      lower_radial_bound, upper_radial_bound,
                                      absolute_tolerance, relative_tolerance);
    }
  }
  return result;
}
}  // namespace SurfaceFinder
