// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/CellCrossingTime.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace Particles::MonteCarlo {

void cell_light_crossing_time(
    gsl::not_null<Scalar<DataVector>*> cell_light_crossing_time,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  const Index<3>& extents = mesh.extents();
  const size_t n_pts = mesh.number_of_grid_points();
  const std::array<size_t, 3> step{1, extents[0], extents[0] * extents[1]};
  const std::array<double, 3> dx_inertial{
      inertial_coordinates.get(0)[step[0]] - inertial_coordinates.get(0)[0],
      inertial_coordinates.get(1)[step[1]] - inertial_coordinates.get(1)[0],
      inertial_coordinates.get(2)[step[2]] - inertial_coordinates.get(2)[0]};

  // Estimate light-crossing time in the cell.
  for (size_t i = 0; i < n_pts; i++) {
    double& min_crossing_time = get(*cell_light_crossing_time)[i];

    min_crossing_time = dx_inertial[0] /
                        (fabs(shift.get(0)[i]) +
                         sqrt(inv_spatial_metric.get(0, 0)[i]) * get(lapse)[i]);
    for (size_t d = 1; d < 3; d++) {
      const double dim_crossing_time =
        gsl::at(dx_inertial,d) /
          (fabs(shift.get(d)[i]) +
           sqrt(inv_spatial_metric.get(d, d)[i]) * get(lapse)[i]);
      min_crossing_time = std::min(min_crossing_time, dim_crossing_time);
    }
  }
}

}  // namespace Particles::MonteCarlo
