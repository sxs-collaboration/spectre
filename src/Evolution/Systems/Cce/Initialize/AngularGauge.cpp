// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/AngularGauge.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

AngularGauge::AngularGauge(CkMigrateMessage* msg) : InitializeJ<false>(msg) {}

AngularGauge::AngularGauge(std::string input_filename,
                           std::string input_subfile_name_coord,
                           double start_time)
    : input_filename_{std::move(input_filename)},
      input_subfile_name_coord_{std::move(input_subfile_name_coord)},
      start_time_{std::move(start_time)} {}

std::unique_ptr<InitializeJ<false>> AngularGauge::get_clone() const {
  return std::make_unique<AngularGauge>(*this);
}

void AngularGauge::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<tmpl::list<::Tags::TempSpinWeightedScalar<0, 2>,
                       ::Tags::TempSpinWeightedScalar<1, 2>,
                       ::Tags::TempSpinWeightedScalar<2, 0>,
                       ::Tags::TempSpinWeightedScalar<3, 2>,
                       ::Tags::TempSpinWeightedScalar<4, 2>,
                       ::Tags::TempSpinWeightedScalar<5, 2>,
                       ::Tags::TempSpinWeightedScalar<6, 0>,
                       ::Tags::TempSpinWeightedScalar<7, 0>>>
      buffers{number_of_angular_points};
  auto& surface_j_buffer = get<::Tags::TempSpinWeightedScalar<0, 2>>(buffers);
  auto& surface_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<1, 2>>(buffers);
  auto& surface_r_buffer = get<::Tags::TempSpinWeightedScalar<2, 0>>(buffers);

  auto& one_minus_y_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<3, 2>>(buffers));
  auto& one_minus_y_cubed_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<4, 2>>(buffers));

  auto& gauge_c = get<::Tags::TempSpinWeightedScalar<5, 2>>(buffers);
  auto& gauge_d = get<::Tags::TempSpinWeightedScalar<6, 0>>(buffers);

  auto& gauge_omega = get<::Tags::TempSpinWeightedScalar<7, 0>>(buffers);

  // Read angular coordinates from Cce Volume file (Tag::CauchyCartesianCoords)
  h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename_};
  auto& data_coord =
      cce_data_file.get<h5::VolumeData>(input_subfile_name_coord_);
  if (data_coord.list_observation_ids().size() == 0) {
    ERROR("The observation IDs list is empty");
  }
  size_t target_obs_id_coord = data_coord.find_observation_id(start_time_);

  const auto& coord_tensor_component_x = data_coord.get_tensor_component(
      target_obs_id_coord, "CauchyCartesianCoords_x");

  const auto& coord_tensor_component_y = data_coord.get_tensor_component(
      target_obs_id_coord, "CauchyCartesianCoords_y");

  const auto& coord_tensor_component_z = data_coord.get_tensor_component(
      target_obs_id_coord, "CauchyCartesianCoords_z");

  if ((not std::holds_alternative<DataVector>(coord_tensor_component_x.data)) or
      (not std::holds_alternative<DataVector>(coord_tensor_component_y.data)) or
      (not std::holds_alternative<DataVector>(coord_tensor_component_z.data))) {
    ERROR("CCE initial coord must be a DataVector");
  }

  get<0>(*cartesian_cauchy_coordinates) =
      std::get<DataVector>(coord_tensor_component_x.data);
  get<1>(*cartesian_cauchy_coordinates) =
      std::get<DataVector>(coord_tensor_component_y.data);
  get<2>(*cartesian_cauchy_coordinates) =
      std::get<DataVector>(coord_tensor_component_z.data);

  // This function transforms the unit vector in cartesian coordinates to
  // spherical coordinates
  GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                          cartesian_cauchy_coordinates);

  Spectral::Swsh::SwshInterpolator interpolator;
  // This function creates the interpolator for the worldtube quantities in the
  // new coordinates from the input cauchy cartesian coordinates
  interpolator = Spectral::Swsh::SwshInterpolator{
      get<0>(*angular_cauchy_coordinates), get<1>(*angular_cauchy_coordinates),
      l_max};

  // This function computes the jacobian factors from the angular coordinate
  // transformation provided
  GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                          make_not_null(&gauge_d),
                                          angular_cauchy_coordinates,
                                          *cartesian_cauchy_coordinates, l_max);

  // This code generates J, dr_J and R in the new angular gauge
  get(gauge_omega).data() =
      0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                 get(gauge_c).data() * conj(get(gauge_c).data()));
  GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>::apply(
      make_not_null(&surface_dr_j_buffer), boundary_dr_j, boundary_j, gauge_c,
      gauge_d, gauge_omega, interpolator, l_max);
  GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
      make_not_null(&surface_j_buffer), boundary_j, gauge_c, gauge_d,
      gauge_omega, interpolator);
  GaugeAdjustedBoundaryValue<Tags::BondiR>::apply(
      make_not_null(&surface_r_buffer), r, gauge_omega, interpolator);

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  one_minus_y_coefficient =
      0.25 * (3.0 * get(surface_j_buffer) +
              get(surface_r_buffer) * get(surface_dr_j_buffer));
  one_minus_y_cubed_coefficient =
      -0.0625 * (get(surface_j_buffer) +
                 get(surface_r_buffer) * get(surface_dr_j_buffer));
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    angular_view_j =
        one_minus_y_collocation[i] * one_minus_y_coefficient.data() +
        pow<3>(one_minus_y_collocation[i]) *
            one_minus_y_cubed_coefficient.data();
  }
}

void AngularGauge::pup(PUP::er& p) {
  p | input_filename_;
  p | input_subfile_name_coord_;
  p | start_time_;
}

PUP::able::PUP_ID AngularGauge::my_PUP_ID = 0;

}  // namespace Cce::InitializeJ
