// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/External/InterpolateFromCocal.hpp"

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

extern "C" {
void coc2cac_ir(const int& npoints, const double* x, const double* y,
                const double* z, double* lapse, double* shift_x,
                double* shift_y, double* shift_z, double* spatial_metric_xx,
                double* spatial_metric_xy, double* spatial_metric_xz,
                double* spatial_metric_yy, double* spatial_metric_yz,
                double* spatial_metric_zz, double* extrinsic_curvature_xx,
                double* extrinsic_curvature_xy, double* extrinsic_curvature_xz,
                double* extrinsic_curvature_yy, double* extrinsic_curvature_yz,
                double* extrinsic_curvature_zz, double* rest_mass_density,
                double* spatial_velocity_x, double* spatial_velocity_y,
                double* spatial_velocity_z, double* pressure,
                double* specific_internal_energy);
}

namespace io {

// namespace {
// DataVector to_datavector(const double* data, size_t size) {
//   DataVector result(size);
//   std::copy(data, data + size, result.begin());
//   return result;
// }

// }  // namespace
namespace {
DataVector to_datavector(std::vector<double> vec) {
  DataVector result(vec.size());
  std::copy(vec.begin(), vec.end(), result.begin());
  return result;
}
}  // namespace

tuples::tagged_tuple_from_typelist<cocal_tags> interpolate_from_cocal(
    const gsl::not_null<std::mutex*> cocal_lock,
    /*const std::string& data_directory,*/
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  tuples::tagged_tuple_from_typelist<cocal_tags> result{};
  const std::lock_guard lock{*cocal_lock};
  const size_t num_points = get<0>(x).size();
  std::vector<double> x_coords(num_points);
  std::vector<double> y_coords(num_points);
  std::vector<double> z_coords(num_points);
  for (size_t i = 0; i < num_points; i++) {
    x_coords[i] = x.get(0)[i];
    y_coords[i] = x.get(1)[i];
    z_coords[i] = x.get(2)[i];
  }

  // std::cout << "Coordinates prepared " << std::endl;

  std::vector<double> lapse(num_points);
  std::vector<double> shift_x(num_points);
  std::vector<double> shift_y(num_points);
  std::vector<double> shift_z(num_points);
  std::vector<double> spatial_metric_xx(num_points);
  std::vector<double> spatial_metric_xy(num_points);
  std::vector<double> spatial_metric_xz(num_points);
  std::vector<double> spatial_metric_yy(num_points);
  std::vector<double> spatial_metric_yz(num_points);
  std::vector<double> spatial_metric_zz(num_points);
  std::vector<double> extrinsic_curvature_xx(num_points);
  std::vector<double> extrinsic_curvature_xy(num_points);
  std::vector<double> extrinsic_curvature_xz(num_points);
  std::vector<double> extrinsic_curvature_yy(num_points);
  std::vector<double> extrinsic_curvature_yz(num_points);
  std::vector<double> extrinsic_curvature_zz(num_points);
  std::vector<double> rest_mass_density(num_points);
  std::vector<double> spatial_velocity_x(num_points);
  std::vector<double> spatial_velocity_y(num_points);
  std::vector<double> spatial_velocity_z(num_points);
  std::vector<double> pressure(num_points);
  std::vector<double> specific_internal_energy(num_points);

  // std::cout << "Calling coc2cac_ir " /* << num_points << " points." */ <<
  // std::endl;

  coc2cac_ir(num_points, x_coords.data(), y_coords.data(), z_coords.data(),
             lapse.data(), shift_x.data(), shift_y.data(), shift_z.data(),
             spatial_metric_xx.data(), spatial_metric_xy.data(),
             spatial_metric_xz.data(), spatial_metric_yy.data(),
             spatial_metric_yz.data(), spatial_metric_zz.data(),
             extrinsic_curvature_xx.data(), extrinsic_curvature_xy.data(),
             extrinsic_curvature_xz.data(), extrinsic_curvature_yy.data(),
             extrinsic_curvature_yz.data(), extrinsic_curvature_zz.data(),
             rest_mass_density.data(), spatial_velocity_x.data(),
             spatial_velocity_y.data(), spatial_velocity_z.data(),
             pressure.data(), specific_internal_energy.data());

  get(get<gr::Tags::Lapse<DataVector>>(result)) = DataVector(num_points);
  get<gr::Tags::Shift<DataVector, 3>>(result).get(0) = DataVector(num_points);
  get<gr::Tags::Shift<DataVector, 3>>(result).get(1) = DataVector(num_points);
  get<gr::Tags::Shift<DataVector, 3>>(result).get(2) = DataVector(num_points);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(0, 0) =
      DataVector(num_points);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(0, 1) =
      DataVector(num_points);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(0, 2) =
      DataVector(num_points);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(1, 1) =
      DataVector(num_points);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(1, 2) =
      DataVector(num_points);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(2, 2) =
      DataVector(num_points);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(0, 0) =
      DataVector(num_points);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(0, 1) =
      DataVector(num_points);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(0, 2) =
      DataVector(num_points);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(1, 1) =
      DataVector(num_points);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(1, 2) =
      DataVector(num_points);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(2, 2) =
      DataVector(num_points);
  get(get<hydro::Tags::RestMassDensity<DataVector>>(result)) =
      DataVector(num_points);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result).get(0) =
      DataVector(num_points);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result).get(1) =
      DataVector(num_points);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result).get(2) =
      DataVector(num_points);
  get(get<hydro::Tags::SpecificInternalEnergy<DataVector>>(result)) =
      DataVector(num_points);
  get(get<hydro::Tags::Pressure<DataVector>>(result)) = DataVector(num_points);

  get(get<gr::Tags::Lapse<DataVector>>(result)) = to_datavector(lapse);
  get<gr::Tags::Shift<DataVector, 3>>(result).get(0) = to_datavector(shift_x);
  get<gr::Tags::Shift<DataVector, 3>>(result).get(1) = to_datavector(shift_y);
  get<gr::Tags::Shift<DataVector, 3>>(result).get(2) = to_datavector(shift_z);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(0, 0) =
      to_datavector(spatial_metric_xx);

  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(0, 1) =
      to_datavector(spatial_metric_xy);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(0, 2) =
      to_datavector(spatial_metric_xz);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(1, 1) =
      to_datavector(spatial_metric_yy);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(1, 2) =
      to_datavector(spatial_metric_yz);
  get<gr::Tags::SpatialMetric<DataVector, 3>>(result).get(2, 2) =
      to_datavector(spatial_metric_zz);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(0, 0) =
      to_datavector(extrinsic_curvature_xx);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(0, 1) =
      to_datavector(extrinsic_curvature_xy);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(0, 2) =
      to_datavector(extrinsic_curvature_xz);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(1, 1) =
      to_datavector(extrinsic_curvature_yy);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(1, 2) =
      to_datavector(extrinsic_curvature_yz);
  get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result).get(2, 2) =
      to_datavector(extrinsic_curvature_zz);
  get(get<hydro::Tags::RestMassDensity<DataVector>>(result)) =
      to_datavector(rest_mass_density);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result).get(0) =
      to_datavector(spatial_velocity_x);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result).get(1) =
      to_datavector(spatial_velocity_y);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result).get(2) =
      to_datavector(spatial_velocity_z);
  get(get<hydro::Tags::Pressure<DataVector>>(result)) = to_datavector(pressure);
  get(get<hydro::Tags::SpecificInternalEnergy<DataVector>>(result)) =
      to_datavector(specific_internal_energy);



  return result;
}

}  // namespace io
