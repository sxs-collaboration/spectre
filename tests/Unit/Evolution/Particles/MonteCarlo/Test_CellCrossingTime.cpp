// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CellCrossingTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Particles.CellCrossingTime",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);

  const size_t dv_size_1d = 3;
  const Mesh<3> mesh(dv_size_1d, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const size_t dv_size = cube(dv_size_1d);
  DataVector zero_dv(dv_size, 0.0);
  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), zero_dv);
  const auto shift =
      TestHelpers::gr::random_shift<3>(make_not_null(&generator), zero_dv);
  const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
      make_not_null(&generator), zero_dv);
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;

  // Inertial coordinates (taken to be the same as the logical coordinates here,
  // as the coordinate transformation plays no role in the calculation).
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  for (size_t iz = 0; iz < dv_size_1d; iz++) {
    const double z_coord = -1.0 + (0.5 + static_cast<double>(iz)) /
                                      static_cast<double>(dv_size_1d) * 2.0;
    for (size_t iy = 0; iy < dv_size_1d; iy++) {
      const double y_coord = -1.0 + (0.5 + static_cast<double>(iy)) /
                                        static_cast<double>(dv_size_1d) * 2.0;
      for (size_t ix = 0; ix < dv_size_1d; ix++) {
        const double x_coord = -1.0 + (0.5 + static_cast<double>(ix)) /
                                          static_cast<double>(dv_size_1d) * 2.0;
        const size_t idx = ix + dv_size_1d * (iy + iz * dv_size_1d);
        inertial_coordinates.get(0)[idx] = x_coord;
        inertial_coordinates.get(1)[idx] = y_coord;
        inertial_coordinates.get(2)[idx] = z_coord;
      }
    }
  }
  Scalar<DataVector> cell_crossing_time =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Particles::MonteCarlo::cell_light_crossing_time(&cell_crossing_time, mesh,
                                                  inertial_coordinates, lapse,
                                                  shift, inv_spatial_metric);

  Scalar<DataVector> expected_cell_light_crossing_time =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);

  for (size_t i = 0; i < dv_size; i++) {
    for (size_t d = 0; d < 3; d++) {
      const double max_light_speed_in_direction =
          fabs(shift.get(d)[i]) +
          sqrt(inv_spatial_metric.get(d, d)[i]) * get(lapse)[i];
      const double min_crossing_time_in_direction =
          2.0 / static_cast<double>(dv_size_1d) / max_light_speed_in_direction;
      if (d == 0 || min_crossing_time_in_direction <
                        get(expected_cell_light_crossing_time)[i]) {
        get(expected_cell_light_crossing_time)[i] =
            min_crossing_time_in_direction;
      }
    }
  }
  const double epsilon_approx = 1.e-13;
  CHECK_ITERABLE_CUSTOM_APPROX(
      expected_cell_light_crossing_time, cell_crossing_time,
      Approx::custom().epsilon(epsilon_approx).scale(1.0));
}
