// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/CcsnCollapse.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::AnalyticData {
namespace detail {
ProgenitorProfile::ProgenitorProfile(const std::string& filename) {
  if (not file_system::check_if_file_exists(filename)) {
    ERROR("Data file not found " << filename);
  }

  std::ifstream prog_file(filename);
  std::string header_line;
  // Loads header lines (including license information) into header_line.
  for (int header_index = 0; header_index < 4; header_index++) {
    std::getline(prog_file, header_line);
  }

  // Read as integer from file.
  double num_radial_points = -1.0;
  double zero_dummy = 0.0;

  // Zero dummy is needed b/c of extra columns in
  // 1st line of txt profile.
  prog_file >> num_radial_points >> zero_dummy >> zero_dummy >> zero_dummy >>
      zero_dummy >> zero_dummy >> zero_dummy;

  ASSERT(num_radial_points > 0.0, "Must have radial points > 0, not "
                                      << num_radial_points
                                      << ".  Check proper file readin. \n");

  num_radial_points_ = static_cast<size_t>(num_radial_points);
  num_angular_points_ = 1;
  num_grid_points_ = num_radial_points_;

  radius_.destructive_resize(num_grid_points_);
  rest_mass_density_.destructive_resize(num_grid_points_);
  temperature_.destructive_resize(num_grid_points_);
  fluid_velocity_.destructive_resize(num_grid_points_);
  electron_fraction_.destructive_resize(num_grid_points_);
  chi_.destructive_resize(num_grid_points_);
  metric_potential_.destructive_resize(num_grid_points_);

  // Loop over data stored in text file.
  for (size_t i = 0; i < num_radial_points_; i++) {
    prog_file >> radius_[i] >> rest_mass_density_[i] >> temperature_[i] >>
        fluid_velocity_[i] >> electron_fraction_[i] >> chi_[i] >>
        metric_potential_[i];
  }
  radius_ *= c2g_length_;
  rest_mass_density_ *= c2g_dens_;
  temperature_ *= c2g_temp_;
  fluid_velocity_ /= speed_of_light_cgs_;
  maximum_radius_ = max(radius_);
}

void ProgenitorProfile::pup(PUP::er& p) {
  p | maximum_radius_;
  p | max_density_ratio_for_linear_interpolation_;

  p | num_radial_points_;
  p | num_angular_points_;
  p | num_grid_points_;

  p | radius_;
  p | rest_mass_density_;
  p | temperature_;
  p | fluid_velocity_;
  p | electron_fraction_;
  p | chi_;
  p | metric_potential_;
}

namespace {
template <size_t StencilSize>
void hydro_interpolation(const gsl::not_null<double*> target_var,
                         const double max_var_ratio_for_linear_interpolation,
                         const double target_coord,
                         const gsl::span<const double>& var_stencil,
                         const gsl::span<const double>& coords) {
  double error_y = 0.0;
  if (const auto min_max_iters =
          std::minmax_element(var_stencil.begin(), var_stencil.end());
      *min_max_iters.second >
      max_var_ratio_for_linear_interpolation * *min_max_iters.first) {
    std::array<double, 2> coord_linear{
        {std::numeric_limits<double>::signaling_NaN(),
         std::numeric_limits<double>::signaling_NaN()}};
    std::array<double, 2> var_linear{
        {std::numeric_limits<double>::signaling_NaN(),
         std::numeric_limits<double>::signaling_NaN()}};
    for (size_t i = 0; i < StencilSize - 1; ++i) {
      // Grab nearest neighbor coordinates and associated variable.
      if (coords[i] <= target_coord and target_coord <= coords[i + 1]) {
        coord_linear[0] = coords[i];
        coord_linear[1] = coords[i + 1];
        var_linear[0] = gsl::at(var_stencil, i);
        var_linear[1] = gsl::at(var_stencil, i + 1);
        break;
      }
    }
    intrp::polynomial_interpolation<1>(
        target_var, make_not_null(&error_y), target_coord,
        gsl::make_span(var_linear.data(), var_linear.size()),
        gsl::make_span(coord_linear.data(), coord_linear.size()));
  } else {
    intrp::polynomial_interpolation<StencilSize - 1>(
        target_var, make_not_null(&error_y), target_coord, var_stencil, coords);
  }
}
}  // namespace

// Input are SpECTRE coords (target_radius & target_cos_theta).
std::array<double, 6> ProgenitorProfile::interpolate(
    const double target_radius, const double target_cos_theta,
    const bool interpolate_hydro_vars) const {
  constexpr size_t stencil_size = 4;
  ASSERT(target_radius >= 0.0,
         "The target radius must be positive, not " << target_radius);
  if (UNLIKELY(target_radius > maximum_radius_)) {
    ERROR("Requested radius " << target_radius
                              << " is greater than what the  "
                                 "progenitor input file contains, "
                              << maximum_radius_);
  }
  using std::abs;

  auto radius_index =
      static_cast<size_t>(num_radial_points_ * target_radius / maximum_radius_);

  if (const size_t grid_index = num_angular_points_ * radius_index;
      UNLIKELY(grid_index > num_grid_points_)) {
    ERROR("grid_index " << grid_index
                        << " is larger than the number of grid points "
                        << num_grid_points_
                        << "\nnumber of angular points: " << num_angular_points_
                        << "\nnumber of radial points: " << num_radial_points_
                        << "\nradius index: " << radius_index);
  }

  if (UNLIKELY(radius_index >= num_radial_points_)) {
    ERROR(
        "radius_index " << radius_index
                        << " is larger than the number of radial grid points: "
                        << num_radial_points_);
  }

  // The radial index at which the stencil lies on the progenitor grid.
  const size_t radial_stencil_index = static_cast<size_t>(std::clamp(
      static_cast<int>(radius_index) - static_cast<int>(stencil_size) / 2, 0,
      static_cast<int>(num_radial_points_ - stencil_size)));

  const size_t radial_index = radial_stencil_index;
  // Radial interpolation
  double error_y = 0.0;
  // Variables to be output after radial interpolation
  double target_rest_mass_density{std::numeric_limits<double>::signaling_NaN()};
  double target_temperature{std::numeric_limits<double>::signaling_NaN()};
  double target_fluid_velocity{std::numeric_limits<double>::signaling_NaN()};
  double target_electron_fraction{std::numeric_limits<double>::signaling_NaN()};
  double target_chi{std::numeric_limits<double>::signaling_NaN()};
  double target_metric_potential{std::numeric_limits<double>::signaling_NaN()};

  const auto radius_span = gsl::make_span(&radius_[radial_index], stencil_size);

  // Interpolate along the radial stencil.
  if (interpolate_hydro_vars) {
    hydro_interpolation<stencil_size>(
        make_not_null(&target_rest_mass_density),
        max_density_ratio_for_linear_interpolation_, target_radius,
        gsl::make_span(&rest_mass_density_[radial_index], stencil_size),
        radius_span);
    hydro_interpolation<stencil_size>(
        make_not_null(&target_temperature),
        // Note: we use max density ratio for temperature.
        max_density_ratio_for_linear_interpolation_, target_radius,
        gsl::make_span(&temperature_[radial_index], stencil_size), radius_span);
    hydro_interpolation<stencil_size>(
        make_not_null(&target_fluid_velocity),
        // Note: we use max density ratio for velocity
        max_density_ratio_for_linear_interpolation_, target_radius,
        gsl::make_span(&fluid_velocity_[radial_index], stencil_size),
        radius_span);
    hydro_interpolation<stencil_size>(
        make_not_null(&target_electron_fraction),
        // Note: we use max density ratio for Ye.
        max_density_ratio_for_linear_interpolation_, target_radius,
        gsl::make_span(&electron_fraction_[radial_index], stencil_size),
        radius_span);
  }

  // Metric variables
  intrp::polynomial_interpolation<stencil_size - 1>(
      make_not_null(&target_chi), make_not_null(&error_y), target_radius,
      gsl::make_span(&chi_[radial_index], stencil_size), radius_span);
  intrp::polynomial_interpolation<stencil_size - 1>(
      make_not_null(&target_metric_potential), make_not_null(&error_y),
      target_radius,
      gsl::make_span(&metric_potential_[radial_index], stencil_size),
      radius_span);

  if (interpolate_hydro_vars) {
    if (UNLIKELY(target_rest_mass_density < 0.0)) {
      ERROR(
          "Failed to interpolate to (r, cos(theta)) = ("
          << target_radius << ',' << target_cos_theta
          << ") because the resulting density is negative: "
          << target_rest_mass_density
          << ". The interpolation should select linear interpolation to avoid "
             "generating negative densities. Please file an issue so this bug "
             "can get fixed.");
    }
  }

  return {
      {interpolate_hydro_vars ? target_rest_mass_density
                              : std::numeric_limits<double>::signaling_NaN(),
       interpolate_hydro_vars ? target_temperature
                              : std::numeric_limits<double>::signaling_NaN(),
       interpolate_hydro_vars ? target_fluid_velocity
                              : std::numeric_limits<double>::signaling_NaN(),
       interpolate_hydro_vars ? target_electron_fraction
                              : std::numeric_limits<double>::signaling_NaN(),
       target_chi, target_metric_potential}};
}

namespace {
template <typename DataType>
void compute_angular_coordinates(
    const gsl::not_null<DataType*> radius,
    const gsl::not_null<DataType*> cos_theta,
    const gsl::not_null<DataType*> sin_theta,
    const gsl::not_null<DataType*> phi,
    const tnsr::I<DataType, 3, Frame::Inertial>& coords) {
  const size_t num_points = get_size(get<0>(coords));
  *radius = get(magnitude(coords));
  if constexpr (not std::is_fundamental_v<DataType>) {
    cos_theta->destructive_resize(num_points);
    sin_theta->destructive_resize(num_points);
    phi->destructive_resize(num_points);
  }
  for (size_t i = 0; i < num_points; ++i) {
    if (get_element(*radius, i) < 1.e-12) {
      // Need to pick an angle when r~0
      get_element(*cos_theta, i) = 0.5;
      get_element(*phi, i) = 0.0;
    } else {
      using std::atan2;
      get_element(*cos_theta, i) =
          get_element(get<2>(coords), i) / get_element(*radius, i);
      get_element(*phi, i) =
          atan2(get_element(get<1>(coords), i), get_element(get<0>(coords), i));
    }
  }
  *sin_theta = sqrt(1.0 - *cos_theta * *cos_theta);
}
}  // namespace
}  // namespace detail

CcsnCollapse::CcsnCollapse(std::string progenitor_filename,
                           double polytropic_constant, double adiabatic_index,
                           double central_angular_velocity,
                           double diff_rot_parameter,
                           double max_dens_ratio_interp)
    : progenitor_filename_(std::move(progenitor_filename)),
      prog_data_{progenitor_filename_},
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(adiabatic_index),
      equation_of_state_(polytropic_constant_, polytropic_exponent_),
      central_angular_velocity_(central_angular_velocity),
      inv_diff_rot_parameter_(1.0 / diff_rot_parameter) {
  CcsnCollapse::prog_data_.set_dens_ratio(max_dens_ratio_interp);
}

CcsnCollapse::CcsnCollapse(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> CcsnCollapse::get_clone()
    const {
  return std::make_unique<CcsnCollapse>(*this);
}

void CcsnCollapse::pup(PUP::er& p) {
  InitialData::pup(p);
  p | progenitor_filename_;
  p | prog_data_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | central_angular_velocity_;
  p | inv_diff_rot_parameter_;
}

// Computes trig quantities based on SpECTRE grid data.
template <typename DataType>
CcsnCollapse::IntermediateVariables<DataType>::IntermediateVariables(
    const tnsr::I<DataType, 3, Frame::Inertial>& in_coords,
    const double in_delta_r)
    : coords(in_coords),
      radius(get_size(get<0>(coords))),
      phi(get_size(radius)),
      cos_theta(get_size(radius)),
      sin_theta(get_size(radius)),
      delta_r(in_delta_r) {
  detail::compute_angular_coordinates(
      make_not_null(&radius), make_not_null(&cos_theta),
      make_not_null(&sin_theta), make_not_null(&phi), coords);
}

template <typename DataType>
CcsnCollapse::IntermediateVariables<DataType>::IntermediateVariables::
    MetricData::MetricData(const size_t num_points)
    : chi(make_with_value<DataType>(
          num_points, std::numeric_limits<double>::signaling_NaN())),
      metric_potential(make_with_value<DataType>(
          num_points, std::numeric_limits<double>::signaling_NaN())) {}

template <typename DataType>
void CcsnCollapse::interpolate_vars_if_necessary(
    const gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  if (vars->rest_mass_density.has_value()) {
    return;
  }
  const size_t num_points = get_size(vars->radius);

  vars->rest_mass_density = make_with_value<DataType>(num_points, 0.0);
  vars->temperature = make_with_value<DataType>(num_points, 0.0);
  vars->fluid_velocity = make_with_value<DataType>(num_points, 0.0);
  vars->electron_fraction = make_with_value<DataType>(num_points, 0.0);
  vars->metric_data =
      typename IntermediateVariables<DataType>::MetricData(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const std::array<double, 6> interpolated_data = prog_data_.interpolate(
        get_element(vars->radius, i), get_element(vars->cos_theta, i), true);

    // Assign SpECTRE data based on interpolated data (right)
    get_element(vars->rest_mass_density.value(), i) = interpolated_data[0];
    get_element(vars->temperature.value(), i) = interpolated_data[1];
    get_element(vars->fluid_velocity.value(), i) = interpolated_data[2];
    get_element(vars->electron_fraction.value(), i) = interpolated_data[3];
    get_element(vars->metric_data->chi, i) = interpolated_data[4];
    get_element(vars->metric_data->metric_potential, i) = interpolated_data[5];
  }
}

template <typename DataType>
void CcsnCollapse::interpolate_deriv_vars_if_necessary(
    const gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  if (vars->metric_data_upper.has_value()) {
    return;
  }
  vars->metric_data_upper =
      std::array<typename IntermediateVariables<DataType>::MetricData, 3>{};
  vars->metric_data_lower =
      std::array<typename IntermediateVariables<DataType>::MetricData, 3>{};
  const double dr = vars->delta_r;
  const size_t num_points = get_size(get<0>(vars->coords));

  for (size_t d = 0; d < 3; ++d) {
    auto coord_upper = vars->coords;
    coord_upper.get(d) += dr;
    auto coord_lower = vars->coords;
    coord_lower.get(d) -= dr;

    detail::compute_angular_coordinates(
        make_not_null(&gsl::at(vars->radius_upper, d)),
        make_not_null(&gsl::at(vars->cos_theta_upper, d)),
        make_not_null(&gsl::at(vars->sin_theta_upper, d)),
        make_not_null(&gsl::at(vars->phi_upper, d)), coord_upper);
    detail::compute_angular_coordinates(
        make_not_null(&gsl::at(vars->radius_lower, d)),
        make_not_null(&gsl::at(vars->cos_theta_lower, d)),
        make_not_null(&gsl::at(vars->sin_theta_lower, d)),
        make_not_null(&gsl::at(vars->phi_lower, d)), coord_lower);

    gsl::at(vars->metric_data_upper.value(), d) =
        typename IntermediateVariables<DataType>::MetricData(num_points);
    gsl::at(vars->metric_data_lower.value(), d) =
        typename IntermediateVariables<DataType>::MetricData(num_points);

    for (size_t i = 0; i < num_points; ++i) {
      const std::array<double, 6> interpolated_data = prog_data_.interpolate(
          get_element(gsl::at(vars->radius_upper, d), i),
          get_element(gsl::at(vars->cos_theta_upper, d), i), false);
      get_element(gsl::at(vars->metric_data_upper.value(), d).chi, i) =

          interpolated_data[4];
      get_element(gsl::at(vars->metric_data_upper.value(), d).metric_potential,
                  i) = interpolated_data[5];
    }
    for (size_t i = 0; i < num_points; ++i) {
      const std::array<double, 6> interpolated_data = prog_data_.interpolate(
          get_element(gsl::at(vars->radius_lower, d), i),
          get_element(gsl::at(vars->cos_theta_lower, d), i), false);
      get_element(gsl::at(vars->metric_data_lower.value(), d).chi, i) =
          interpolated_data[4];
      get_element(gsl::at(vars->metric_data_lower.value(), d).metric_potential,
                  i) = interpolated_data[5];
    }
  }
}

// Lapse = exp(metric_potential)
template <typename DataType>
Scalar<DataType> CcsnCollapse::lapse(const DataType& metric_potential) const {
  return Scalar<DataType>{exp(metric_potential)};
}

// Shift = 0 when using Schwarzschild coordinates
template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial> CcsnCollapse::shift(
    const DataType& radius) const {
  auto result =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(radius, 0.0);
  return result;
}

// Spatial metric
template <typename DataType>
tnsr::ii<DataType, 3, Frame::Inertial> CcsnCollapse::spatial_metric(
    const DataType& chi, const DataType& cos_theta, const DataType& sin_theta,
    const DataType& phi) const {
  auto result =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(chi, 0.0);
  for (size_t i = 0; i < get_size(chi); ++i) {
    // chi2 = grr in RGPS
    const double chisqrd = square(get_element(chi, i));
    // cos(theta)
    const double ct = get_element(cos_theta, i);
    // sin(theta)
    const double st = get_element(sin_theta, i);
    // cos(phi)
    const double cp = cos(get_element(phi, i));
    // sin(phi)
    const double sp = sin(get_element(phi, i));

    // Remapped quantities from M.A. Pajkos thesis, Appendix A.2
    // xx
    get_element(get<0, 0>(result), i) =
        square(sp) + square(cp) * (chisqrd * square(st) + square(ct));
    // xy
    get_element(get<0, 1>(result), i) =
        cp * sp * (-1.0 + chisqrd * square(st) + square(ct));
    // xz
    get_element(get<0, 2>(result), i) = cp * ct * st * (chisqrd - 1.0);
    // yy
    get_element(get<1, 1>(result), i) =
        square(cp) + square(sp) * (chisqrd * square(st) + square(ct));
    // yz
    get_element(get<1, 2>(result), i) = sp * ct * st * (chisqrd - 1.0);
    // zz
    get_element(get<2, 2>(result), i) = chisqrd * square(ct) + square(st);
  }
  return result;
}

// Begin hydro variables.
// Rest mass density
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return {Scalar<DataType>{vars->rest_mass_density.value()}};
}

// Specific enthalpy
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(vars, x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}));
  using std::max;
  return {hydro::relativistic_specific_enthalpy(
      Scalar<DataType>{DataType{max(1.0e-300, get(rest_mass_density))}},
      equation_of_state_.specific_internal_energy_from_density(
          rest_mass_density),
      equation_of_state_.pressure_from_density(rest_mass_density))};
}

// Pressure
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(vars, x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}));
  return {equation_of_state_.pressure_from_density(rest_mass_density)};
}

// Specific internal energy
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(vars, x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}));

  // Eventually, below will call for piecewise polytrope or tabulated EOS
  //  auto temperature = vars->temperature;
  //  auto electron_fraction = vars->electron_fraction;
  // return{equation_of_state_.
  // specific_internal_energy_from_density_and_temperature_and_ye_impl(
  // rest_mass_density, temperature, electron_fraction)};

  return {equation_of_state_.specific_internal_energy_from_density(
      rest_mass_density)};
}

// Electron fraction
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return {Scalar<DataType>{vars->electron_fraction.value()}};
}

// Once temperature is added to hydro tags for the grmhd system,
// temperature will be stored similar to below
// // Temperature
// template <typename DataType>
// tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>
// CcsnCollapse::variables(
//     const gsl::not_null<IntermediateVariables<DataType>*> vars,
//     const tnsr::I<DataType, 3>& /*x*/,
//     tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const {
//   return {Scalar<DataType>{vars->temperature.value()}};
// }

// Velocity (typical spherical coord system rotating around z axis)
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  // x and y components will have a radial velocity (from file),
  // and toroidal velocity (from rotation law) component.

  using std::cos;
  using std::sin;
  // Omega = omega0/(1 + (r/A)^2)
  // vphi = distance_from_rotation_axis * Omega
  // Temporarily store rotational velocity in z velocity component;
  // will overwrite at the end.
  get<2>(spatial_velocity) =
      vars->radius * vars->sin_theta * central_angular_velocity_ /
      (1.0 + square(vars->radius * inv_diff_rot_parameter_));

  // speed of light check
  ASSERT(max(get<2>(spatial_velocity)) <= 1.0,
         "Spatial velocity " << max(get<2>(spatial_velocity))
                             << " is greater than the speed of light.  Central"
                                " angular velocity is too large. \n");

  // vx = vr*sin(theta)*cos(phi) - vphi*sin(phi)
  get<0>(spatial_velocity) =
      vars->fluid_velocity.value() * vars->sin_theta * cos(vars->phi) -
      get<2>(spatial_velocity) * sin(vars->phi);

  // vy = vr*sin(theta)*sin(phi) + vphi*cos(phi)
  get<1>(spatial_velocity) =
      vars->fluid_velocity.value() * vars->sin_theta * sin(vars->phi) +
      get<2>(spatial_velocity) * cos(vars->phi);

  // vz = vr*cos(theta)
  get<2>(spatial_velocity) =
      vars->fluid_velocity.value() * sqrt(1.0 - square(vars->sin_theta));

  return {std::move(spatial_velocity)};
}

// Lorentz
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);

  auto lorentz_factor = make_with_value<Scalar<DataType>>(get<0>(x), 0.0);
  // Compute spatial proper velocity, then Lorentz factor.
  const auto spatial_metric = get<gr::Tags::SpatialMetric<DataType, 3>>(
      variables(vars, x, tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{}));

  const auto spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataType, 3>>(variables(
          vars, x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}));

  // Spatial velocity squared
  get(lorentz_factor) =
      get(dot_product(spatial_velocity, spatial_velocity, spatial_metric));

  get(lorentz_factor) = 1.0 / sqrt(1.0 - get(lorentz_factor));
  return {std::move(lorentz_factor)};
}

// B field
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> /*vars*/,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

// Div cleaning field
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> /*vars*/,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

// Lapse
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Lapse<DataType>> CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return lapse(vars->metric_data->metric_potential);
}

// Shift
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<DataType, 3>> CcsnCollapse::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::Shift<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  // 0 in Schwarzschild coords
  return shift(vars->radius);
}

// Spatial metric
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, 3>>
CcsnCollapse::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::SpatialMetric<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return spatial_metric(vars->metric_data->chi, vars->cos_theta,
                        vars->sin_theta, vars->phi);
}

// Sqrt_det_spatial_metric
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>
CcsnCollapse::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  // Analytically, in the remapped coordinates, the square root of
  // the determinant of the spatial metric reduces to Chi
  return {Scalar<DataType>{vars->metric_data->chi}};
}

// Inverse spatial metric
template <typename DataType>
tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<DataType, 3>>
CcsnCollapse::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);

  const auto spatial_metric = get<gr::Tags::SpatialMetric<DataType, 3>>(
      variables(vars, x, tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{}));

  // Determinant is returned first, inverse is returned second
  return determinant_and_inverse(spatial_metric).second;
}

// Lapse spatial derivative
template <typename DataType>
auto CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  interpolate_deriv_vars_if_necessary(vars);
  // Do 2nd-order FD (upper=upper index, lower = lower index)
  auto deriv_lapse =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  for (size_t d = 0; d < 3; ++d) {
    const auto lapse_upper =
        lapse(gsl::at(vars->metric_data_upper.value(), d).metric_potential);

    const auto lapse_lower =
        lapse(gsl::at(vars->metric_data_lower.value(), d).metric_potential);
    deriv_lapse.get(d) =
        (0.5 / vars->delta_r) * (get(lapse_upper) - get(lapse_lower));
  }
  return deriv_lapse;
}

// Shift derivative
template <typename DataType>
auto CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> /*vars*/,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<DerivShift<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  auto deriv_shift =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  return deriv_shift;
}

// Spatial metric derivative (finite difference)
template <typename DataType>
auto CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivSpatialMetric<DataType>> {
  interpolate_deriv_vars_if_necessary(vars);
  // Do 2nd-order FD. This isn't super accurate at r=0, but it's fine
  // everywhere else.
  auto deriv_spatial_metric =
      make_with_value<tnsr::ijj<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  for (size_t d = 0; d < 3; ++d) {
    const auto spatial_metric_upper = spatial_metric(
        gsl::at(vars->metric_data_upper.value(), d).chi,
        gsl::at(vars->cos_theta_upper, d), gsl::at(vars->sin_theta_upper, d),
        gsl::at(vars->phi_upper, d));
    const auto spatial_metric_lower = spatial_metric(
        gsl::at(vars->metric_data_lower.value(), d).chi,
        gsl::at(vars->cos_theta_lower, d), gsl::at(vars->sin_theta_lower, d),
        gsl::at(vars->phi_lower, d));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        deriv_spatial_metric.get(d, i, j) =
            (0.5 / vars->delta_r) *
            (spatial_metric_upper.get(i, j) - spatial_metric_lower.get(i, j));
      }
    }
  }
  return deriv_spatial_metric;
}

// Extrinsic curvature
template <typename DataType>
auto CcsnCollapse::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, 3>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, 3>> {
  // 0 in constant spacetime, RGPS
  return gr::extrinsic_curvature(
      get<gr::Tags::Lapse<DataType>>(
          variables(vars, x, tmpl::list<gr::Tags::Lapse<DataType>>{})),
      get<gr::Tags::Shift<DataType, 3>>(
          variables(vars, x, tmpl::list<gr::Tags::Shift<DataType, 3>>{})),
      get<DerivShift<DataType>>(
          variables(vars, x, tmpl::list<DerivShift<DataType>>{})),
      get<gr::Tags::SpatialMetric<DataType, 3>>(variables(
          vars, x, tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{})),
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.0),
      get<DerivSpatialMetric<DataType>>(
          variables(vars, x, tmpl::list<DerivSpatialMetric<DataType>>{})));
}

PUP::able::PUP_ID CcsnCollapse::my_PUP_ID = 0;

bool operator==(const CcsnCollapse& lhs, const CcsnCollapse& rhs) {
  // The equation of state and progenitor solution aren't explicitly checked.
  // However, if progenitor_filename_ and the polytropic_exponent_ and
  // polytropic_constant_ are the same, then the solution and
  // EOS should be too.
  return lhs.progenitor_filename_ == rhs.progenitor_filename_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_ and
         lhs.central_angular_velocity_ == rhs.central_angular_velocity_ and
         lhs.inv_diff_rot_parameter_ == rhs.inv_diff_rot_parameter_;
}

bool operator!=(const CcsnCollapse& lhs, const CcsnCollapse& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template class CcsnCollapse::IntermediateVariables<DTYPE(data)>;          \
  template Scalar<DTYPE(data)> CcsnCollapse::lapse(const DTYPE(data) &      \
                                                   metric_potential) const; \
  template tnsr::I<DTYPE(data), 3, Frame::Inertial> CcsnCollapse::shift(    \
      const DTYPE(data) & radius) const;                                    \
  template tnsr::ii<DTYPE(data), 3, Frame::Inertial>                        \
  CcsnCollapse::spatial_metric(                                             \
      const DTYPE(data) & chi, const DTYPE(data) & cos_theta,               \
      const DTYPE(data) & sin_theta, const DTYPE(data) & phi) const;        \
  template void CcsnCollapse::interpolate_vars_if_necessary(                \
      const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars) const; \
  template void CcsnCollapse::interpolate_deriv_vars_if_necessary(          \
      const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE

#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                     \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data) >>              \
      CcsnCollapse::variables(                                           \
          const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars, \
          const tnsr::I<DTYPE(data), 3>& x,                              \
          tmpl::list < TAG(data) < DTYPE(data) >>                        \
          /*meta*/) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::LorentzFactor,
     hydro::Tags::SpecificEnthalpy, gr::Tags::Lapse,
     gr::Tags::SqrtDetSpatialMetric))

#undef INSTANTIATE_SCALARS

#define INSTANTIATE_MHD_VECTORS(_, data)                                     \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data), 3,                 \
      Frame::Inertial >>                                                     \
          CcsnCollapse::variables(                                           \
              const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars, \
              const tnsr::I<DTYPE(data), 3>& x,                              \
              tmpl::list < TAG(data) < DTYPE(data), 3, Frame::Inertial >>    \
              /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_MHD_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef INSTANTIATE_MHD_VECTORS

#define INSTANTIATE_METRIC_TENSORS(_, data)                                   \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data),                     \
      3 >> CcsnCollapse::variables(                                           \
               const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars, \
               const tnsr::I<DTYPE(data), 3>& x,                              \
               tmpl::list < TAG(data) < DTYPE(data), 3 >>                     \
               /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_METRIC_TENSORS, (double, DataVector),
                        (gr::Tags::Shift, gr::Tags::SpatialMetric,
                         gr::Tags::InverseSpatialMetric,
                         gr::Tags::ExtrinsicCurvature))

#undef INSTANTIATE_METRIC_TENSORS

#define INSTANTIATE_METRIC_DERIVS(_, data)                              \
  template auto CcsnCollapse::variables(                                \
      const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars,    \
      const tnsr::I<DTYPE(data), 3>& x,                                 \
      tmpl::list < TAG(data) < DTYPE(data) >>                           \
      /*meta*/) const->tuples::TaggedTuple < TAG(data) < DTYPE(data) >> \
      ;

GENERATE_INSTANTIATIONS(INSTANTIATE_METRIC_DERIVS, (double, DataVector),
                        (DerivLapse, DerivShift, DerivSpatialMetric))

#undef INSTANTIATE_METRIC_DERIVS

#undef TAG
#undef DTYPE
}  // namespace grmhd::AnalyticData
