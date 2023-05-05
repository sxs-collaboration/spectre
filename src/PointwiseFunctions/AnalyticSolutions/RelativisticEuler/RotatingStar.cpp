// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
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
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace RelativisticEuler::Solutions {
namespace detail {
CstSolution::CstSolution(const std::string& filename,
                         const double equilibrium_kappa) {
  std::ifstream cst_file(filename);
  std::string header_line;
  std::getline(cst_file, header_line);
  // Read as integers from file to be compatible with SpEC. Might not be
  // necessary.
  int num_radial_points = -1;
  int num_angular_points = -1;
  cst_file >> num_radial_points >> num_angular_points;
  num_radial_points_ = static_cast<size_t>(num_radial_points);
  // The RotNS code seems to write out more grid points than it claims. It's
  // unclear why, but SpEC does the `+ 1`.
  num_angular_points_ = static_cast<size_t>(num_angular_points) + 1;
  num_grid_points_ = num_radial_points_ * num_angular_points_;
  cst_file >> equatorial_radius_ >> polytropic_index_ >>
      central_angular_speed_ >> rotation_profile_;

  radius_.destructive_resize(num_grid_points_);
  cos_theta_.destructive_resize(num_grid_points_);
  rest_mass_density_.destructive_resize(num_grid_points_);
  fluid_velocity_.destructive_resize(num_grid_points_);
  alpha_.destructive_resize(num_grid_points_);
  rho_.destructive_resize(num_grid_points_);
  gamma_.destructive_resize(num_grid_points_);
  omega_.destructive_resize(num_grid_points_);

  for (size_t i = 0; i < num_radial_points_; i++) {
    for (size_t j = 0; j < num_angular_points_; j++) {
      // The data is stored in cos(theta) varies fastest
      //
      // Note that cos(theta) is between 0 and 1.
      const size_t index = num_angular_points_ * i + j;
      cst_file >> radius_[index] >> cos_theta_[index] >>
          rest_mass_density_[index] >> alpha_[index] >> rho_[index] >>
          gamma_[index] >> omega_[index] >> fluid_velocity_[index];
      radius_[index] *= pow(equilibrium_kappa, polytropic_index_ * 0.5);
      rest_mass_density_[index] *= pow(equilibrium_kappa, -polytropic_index_);
    }
  }
  maximum_radius_ = max(radius_);
  equatorial_radius_ *= pow(equilibrium_kappa, polytropic_index_ * 0.5);
}

void CstSolution::pup(PUP::er& p) {
  p | maximum_radius_;
  p | max_density_ratio_for_linear_interpolation_;

  p | equatorial_radius_;
  p | polytropic_index_;
  p | central_angular_speed_;
  p | rotation_profile_;
  p | num_radial_points_;
  p | num_angular_points_;
  p | num_grid_points_;

  p | radius_;
  p | cos_theta_;
  p | rest_mass_density_;
  p | fluid_velocity_;
  p | alpha_;
  p | rho_;
  p | gamma_;
  p | omega_;
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

std::array<double, 6> CstSolution::interpolate(
    const double target_radius, const double target_cos_theta,
    const bool interpolate_hydro_vars) const {
  constexpr size_t stencil_size = 4;
  ASSERT(target_radius >= 0.0,
         "The target radius must be positive, not " << target_radius);
  if (UNLIKELY(target_radius > maximum_radius_)) {
    ERROR("Requested radius " << target_radius
                              << " is greater than what the Cook, Shapiro, "
                                 "Teukolsky input file contains, "
                              << maximum_radius_);
  }
  using std::abs;
  const double target_abs_cos_theta = abs(target_cos_theta);
  const auto angular_index = static_cast<size_t>(
      target_abs_cos_theta * static_cast<double>(num_angular_points_ - 1));
  // radius_index differs from SpEC by +1 to get the stencils to be centered.
  size_t radius_index =
      static_cast<size_t>(num_radial_points_ * target_radius /
                          (target_radius + equatorial_radius_)) +
      1;
  if (const size_t grid_index =
          num_angular_points_ * radius_index + angular_index;
      UNLIKELY(grid_index > num_grid_points_)) {
    ERROR("grid_index " << grid_index
                        << " is larger than the number of grid points "
                        << num_grid_points_
                        << "\nnumber of angular points: " << num_angular_points_
                        << "\nnumber of radial points: " << num_radial_points_
                        << "\nradius index: " << radius_index
                        << "\nangular_index: " << angular_index);
  }
  // if (UNLIKELY(radius_[num_angular_points_ * radius_index + angular_index] >
  //              target_radius)) {
  //   // This sanity check and adjustment is in SpEC, but Nils Deppe isn't sure
  //   // under what circumstances it is necessary. Best guess is that because
  //   // the radius_index is an estimate it could be that you compute an index
  //   // that's slightly an over-estimate. We could do a binary search to find
  //   // the exact index if the estimate fails.
  //   radius_index--;
  // }
  if (UNLIKELY(radius_index >= num_radial_points_)) {
    ERROR(
        "radius_index " << radius_index
                        << " is larger than the number of radial grid points: "
                        << num_radial_points_);
  }

  const size_t radial_stencil_index = static_cast<size_t>(std::clamp(
      static_cast<int>(radius_index) - static_cast<int>(stencil_size) / 2, 0,
      static_cast<int>(num_radial_points_ - stencil_size)));
  const size_t angular_stencil_index = static_cast<size_t>(std::clamp(
      static_cast<int>(angular_index) - static_cast<int>(stencil_size) / 2, 0,
      static_cast<int>(num_angular_points_ - stencil_size - 1)));

  // We first interpolate in cos(theta) and then in radius.
  std::array<double, stencil_size> rest_mass_density_rad_stencil{};
  std::array<double, stencil_size> fluid_velocity_rad_stencil{};
  std::array<double, stencil_size> alpha_rad_stencil{};
  std::array<double, stencil_size> rho_rad_stencil{};
  std::array<double, stencil_size> gamma_rad_stencil{};
  std::array<double, stencil_size> omega_rad_stencil{};
  // Since the radius is not contiguous, we need to copy the radius into a
  // contiguous buffer. radius_for_stencil is that buffer.
  std::array<double, stencil_size> radius_for_stencil{};
  for (size_t stencil_rad_index = 0; stencil_rad_index < stencil_size;
       ++stencil_rad_index) {
    const size_t radial_index =
        (stencil_rad_index + radial_stencil_index) * num_angular_points_;
    gsl::at(radius_for_stencil, stencil_rad_index) =
        radius_[radial_index + angular_stencil_index];

    double error_y = 0.0;
    const auto cos_theta_span = gsl::make_span(
        &cos_theta_[radial_index + angular_stencil_index], stencil_size);
    // At the surface we want to do linear interpolation to avoid unphysical
    // oscillations that result in negative densities. We could do some sort of
    // WENO-type approach to do a one-sided interpolation, but that's alot more
    // involved.
    if (interpolate_hydro_vars) {
      const auto density_span = gsl::make_span(
          &rest_mass_density_[radial_index + angular_stencil_index],
          stencil_size);
      hydro_interpolation<stencil_size>(
          make_not_null(
              &gsl::at(rest_mass_density_rad_stencil, stencil_rad_index)),
          max_density_ratio_for_linear_interpolation_, target_abs_cos_theta,
          density_span, cos_theta_span);
      hydro_interpolation<stencil_size>(
          make_not_null(
              &gsl::at(fluid_velocity_rad_stencil, stencil_rad_index)),
          // Note: we use max density ratio for velocity
          max_density_ratio_for_linear_interpolation_, target_abs_cos_theta,
          gsl::make_span(&fluid_velocity_[radial_index + angular_stencil_index],
                         stencil_size),
          cos_theta_span);
    }
    intrp::polynomial_interpolation<stencil_size - 1>(
        make_not_null(&gsl::at(alpha_rad_stencil, stencil_rad_index)),
        make_not_null(&error_y), target_abs_cos_theta,
        gsl::make_span(&alpha_[radial_index + angular_stencil_index],
                       stencil_size),
        cos_theta_span);
    intrp::polynomial_interpolation<stencil_size - 1>(
        make_not_null(&gsl::at(rho_rad_stencil, stencil_rad_index)),
        make_not_null(&error_y), target_abs_cos_theta,
        gsl::make_span(&rho_[radial_index + angular_stencil_index],
                       stencil_size),
        cos_theta_span);
    intrp::polynomial_interpolation<stencil_size - 1>(
        make_not_null(&gsl::at(gamma_rad_stencil, stencil_rad_index)),
        make_not_null(&error_y), target_abs_cos_theta,
        gsl::make_span(&gamma_[radial_index + angular_stencil_index],
                       stencil_size),
        cos_theta_span);
    intrp::polynomial_interpolation<stencil_size - 1>(
        make_not_null(&gsl::at(omega_rad_stencil, stencil_rad_index)),
        make_not_null(&error_y), target_abs_cos_theta,
        gsl::make_span(&omega_[radial_index + angular_stencil_index],
                       stencil_size),
        cos_theta_span);
  }

  // Do radial interpolation
  double error_y = 0.0;
  double target_rest_mass_density{std::numeric_limits<double>::signaling_NaN()};
  double target_fluid_velocity{std::numeric_limits<double>::signaling_NaN()};
  double target_alpha{std::numeric_limits<double>::signaling_NaN()};
  double target_rho{std::numeric_limits<double>::signaling_NaN()};
  double target_gamma{std::numeric_limits<double>::signaling_NaN()};
  double target_omega{std::numeric_limits<double>::signaling_NaN()};
  const auto radius_span = gsl::make_span(&radius_for_stencil[0], stencil_size);
  if (interpolate_hydro_vars) {
    hydro_interpolation<stencil_size>(
        make_not_null(&target_rest_mass_density),
        max_density_ratio_for_linear_interpolation_, target_radius,
        gsl::make_span(&rest_mass_density_rad_stencil[0], stencil_size),
        radius_span);
    hydro_interpolation<stencil_size>(
        make_not_null(&target_fluid_velocity),
        // Note: we use max density ratio for velocity
        max_density_ratio_for_linear_interpolation_, target_radius,
        gsl::make_span(&fluid_velocity_rad_stencil[0], stencil_size),
        radius_span);
  }
  intrp::polynomial_interpolation<stencil_size - 1>(
      make_not_null(&target_alpha), make_not_null(&error_y), target_radius,
      gsl::make_span(&alpha_rad_stencil[0], stencil_size), radius_span);
  intrp::polynomial_interpolation<stencil_size - 1>(
      make_not_null(&target_rho), make_not_null(&error_y), target_radius,
      gsl::make_span(&rho_rad_stencil[0], stencil_size), radius_span);
  intrp::polynomial_interpolation<stencil_size - 1>(
      make_not_null(&target_gamma), make_not_null(&error_y), target_radius,
      gsl::make_span(&gamma_rad_stencil[0], stencil_size), radius_span);
  intrp::polynomial_interpolation<stencil_size - 1>(
      make_not_null(&target_omega), make_not_null(&error_y), target_radius,
      gsl::make_span(&omega_rad_stencil[0], stencil_size), radius_span);

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

    target_fluid_velocity /= equatorial_radius_;
  }
  target_omega /= equatorial_radius_;

  return {
      {interpolate_hydro_vars ? target_rest_mass_density
                              : std::numeric_limits<double>::signaling_NaN(),
       interpolate_hydro_vars ? target_fluid_velocity
                              : std::numeric_limits<double>::signaling_NaN(),
       target_alpha, target_rho, target_gamma, target_omega}};
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
      // cos_theta=0.5 is what SpEC uses. Use that to match.
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

RotatingStar::RotatingStar(std::string rot_ns_filename,
                           double polytropic_constant)
    : rot_ns_filename_(std::move(rot_ns_filename)),
      cst_solution_{rot_ns_filename_, polytropic_constant},
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_{1.0 + 1.0 / cst_solution_.polytropic_index()},
      equation_of_state_(polytropic_constant_, polytropic_exponent_) {}

RotatingStar::RotatingStar(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> RotatingStar::get_clone()
    const {
  return std::make_unique<RotatingStar>(*this);
}

void RotatingStar::pup(PUP::er& p) {
  InitialData::pup(p);
  p | rot_ns_filename_;
  p | cst_solution_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
}

template <typename DataType>
RotatingStar::IntermediateVariables<DataType>::IntermediateVariables(
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
RotatingStar::IntermediateVariables<DataType>::IntermediateVariables::
    MetricData::MetricData(const size_t num_points)
    : alpha(make_with_value<DataType>(
          num_points, std::numeric_limits<double>::signaling_NaN())),
      rho(make_with_value<DataType>(
          num_points, std::numeric_limits<double>::signaling_NaN())),
      gamma(make_with_value<DataType>(
          num_points, std::numeric_limits<double>::signaling_NaN())),
      omega(make_with_value<DataType>(
          num_points, std::numeric_limits<double>::signaling_NaN())) {}

template <typename DataType>
void RotatingStar::interpolate_vars_if_necessary(
    const gsl::not_null<IntermediateVariables<DataType>*> vars) const {
  if (vars->rest_mass_density.has_value()) {
    return;
  }
  const size_t num_points = get_size(vars->radius);
  vars->rest_mass_density = make_with_value<DataType>(num_points, 0.0);
  vars->fluid_velocity = make_with_value<DataType>(num_points, 0.0);
  vars->metric_data =
      typename IntermediateVariables<DataType>::MetricData(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const std::array<double, 6> interpolated_data = cst_solution_.interpolate(
        get_element(vars->radius, i), get_element(vars->cos_theta, i), true);
    get_element(vars->rest_mass_density.value(), i) = interpolated_data[0];
    get_element(vars->fluid_velocity.value(), i) = interpolated_data[1];
    get_element(vars->metric_data->alpha, i) = interpolated_data[2];
    get_element(vars->metric_data->rho, i) = interpolated_data[3];
    get_element(vars->metric_data->gamma, i) = interpolated_data[4];
    get_element(vars->metric_data->omega, i) = interpolated_data[5];
  }
}

template <typename DataType>
void RotatingStar::interpolate_deriv_vars_if_necessary(
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
      const std::array<double, 6> interpolated_data = cst_solution_.interpolate(
          get_element(gsl::at(vars->radius_upper, d), i),
          get_element(gsl::at(vars->cos_theta_upper, d), i), false);
      get_element(gsl::at(vars->metric_data_upper.value(), d).alpha, i) =
          interpolated_data[2];
      get_element(gsl::at(vars->metric_data_upper.value(), d).rho, i) =
          interpolated_data[3];
      get_element(gsl::at(vars->metric_data_upper.value(), d).gamma, i) =
          interpolated_data[4];
      get_element(gsl::at(vars->metric_data_upper.value(), d).omega, i) =
          interpolated_data[5];
    }
    for (size_t i = 0; i < num_points; ++i) {
      const std::array<double, 6> interpolated_data = cst_solution_.interpolate(
          get_element(gsl::at(vars->radius_lower, d), i),
          get_element(gsl::at(vars->cos_theta_lower, d), i), false);
      get_element(gsl::at(vars->metric_data_lower.value(), d).alpha, i) =
          interpolated_data[2];
      get_element(gsl::at(vars->metric_data_lower.value(), d).rho, i) =
          interpolated_data[3];
      get_element(gsl::at(vars->metric_data_lower.value(), d).gamma, i) =
          interpolated_data[4];
      get_element(gsl::at(vars->metric_data_lower.value(), d).omega, i) =
          interpolated_data[5];
    }
  }
}

template <typename DataType>
Scalar<DataType> RotatingStar::lapse(const DataType& gamma,
                                     const DataType& rho) const {
  return Scalar<DataType>{exp(0.5 * (gamma + rho))};
}

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial> RotatingStar::shift(
    const DataType& omega, const DataType& phi, const DataType& radius,
    const DataType& sin_theta) const {
  auto result =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(radius, 0.0);
  get<0>(result) = sin(phi) * omega * radius * sin_theta;
  get<1>(result) = -cos(phi) * omega * radius * sin_theta;
  return result;
}

template <typename DataType>
tnsr::ii<DataType, 3, Frame::Inertial> RotatingStar::spatial_metric(
    const DataType& gamma, const DataType& rho, const DataType& alpha,
    const DataType& phi) const {
  auto result =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(gamma, 0.0);
  for (size_t i = 0; i < get_size(gamma); ++i) {
    const double g_rr = exp(2.0 * get_element(alpha, i));
    const double g_phi_phi = exp(get_element(gamma, i) - get_element(rho, i));
    const double cos_phi = cos(get_element(phi, i));
    const double sin_phi = sin(get_element(phi, i));
    // Note: We analytically simplify the expressions using trig identities but
    // have left in the full expressions relating to the jacobian (L) for
    // comparison with SpEC.
    //
    // const double g_theta_theta = g_rr;
    // const double Lrx = get_element(sin_theta, i) * cos(get_element(phi, i));
    // const double Lry = get_element(sin_theta, i) * sin(get_element(phi, i));
    // const double Lrz = get_element(cos_theta, i);
    // const double Ltx = get_element(cos_theta, i) * cos(get_element(phi, i));
    // const double Lty = get_element(cos_theta, i) * sin(get_element(phi, i));
    // const double Ltz = -get_element(sin_theta, i);
    // const double Lpx = -sin(get_element(phi, i));
    // const double Lpy = cos(get_element(phi, i));
    // const double Lpz = 0.0;

    // xx
    get_element(get<0, 0>(result), i) =
        square(cos_phi) * g_rr + square(sin_phi) * g_phi_phi;
    // Lrx * Lrx * g_rr + Ltx * Ltx * g_theta_theta + Lpx * Lpx * g_phi_phi;

    // xy
    get_element(get<0, 1>(result), i) = cos_phi * sin_phi * (g_rr - g_phi_phi);
    // Lrx * Lry * g_rr + Ltx * Lty * g_theta_theta + Lpx * Lpy * g_phi_phi;

    // xz
    // gamma_{xz} = 0
    // get_element(get<0, 2>(result), i) =
    //     Lrx * Lrz * g_rr + Ltx * Ltz * g_theta_theta + Lpx * Lpz * g_phi_phi;

    // yy
    get_element(get<1, 1>(result), i) =
        square(sin_phi) * g_rr + square(cos_phi) * g_phi_phi;
    // Lry * Lry * g_rr + Lty * Lty * g_theta_theta + Lpy * Lpy * g_phi_phi;

    // yz
    // gamma_{yz} = 0
    // get_element(get<1, 2>(result), i) =
    //     Lrz * Lry * g_rr + Ltz * Lty * g_theta_theta + Lpz * Lpy * g_phi_phi;

    // zz
    get_element(get<2, 2>(result), i) = g_rr;
    // Lrz * Lrz * g_rr + Ltz * Ltz * g_theta_theta + Lpz * Lpz * g_phi_phi;
  }
  return result;
}

template <typename DataType>
tnsr::II<DataType, 3, Frame::Inertial> RotatingStar::inverse_spatial_metric(
    const DataType& gamma, const DataType& rho, const DataType& alpha,
    const DataType& phi) const {
  auto result =
      make_with_value<tnsr::II<DataType, 3, Frame::Inertial>>(gamma, 0.0);
  for (size_t i = 0; i < get_size(gamma); ++i) {
    const double one_over_g_rr = exp(-2.0 * get_element(alpha, i));
    const double one_over_g_phi_phi =
        exp(get_element(rho, i) - get_element(gamma, i));
    const double cos_phi = cos(get_element(phi, i));
    const double sin_phi = sin(get_element(phi, i));
    get_element(get<0, 0>(result), i) =
        square(sin_phi) * one_over_g_phi_phi + square(cos_phi) * one_over_g_rr;
    get_element(get<0, 1>(result), i) =
        cos_phi * sin_phi * (one_over_g_rr - one_over_g_phi_phi);
    get_element(get<1, 1>(result), i) =
        square(cos_phi) * one_over_g_phi_phi + square(sin_phi) * one_over_g_rr;
    get_element(get<2, 2>(result), i) = one_over_g_rr;
  }
  return result;
}

template <typename DataType>
Scalar<DataType> RotatingStar::sqrt_det_spatial_metric(
    const DataType& gamma, const DataType& rho, const DataType& alpha) const {
  return Scalar<DataType>{exp(2.0 * alpha + 0.5 * (gamma - rho))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return {Scalar<DataType>{vars->rest_mass_density.value()}};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
RotatingStar::variables(
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

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>> RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(vars, x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}));
  return {equation_of_state_.pressure_from_density(rest_mass_density)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  const auto rest_mass_density = get<hydro::Tags::RestMassDensity<DataType>>(
      variables(vars, x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{}));
  return {equation_of_state_.specific_internal_energy_from_density(
      rest_mass_density)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> /* vars */,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {

  auto ye = make_with_value<Scalar<DataType>>(x, 0.1);

  return {std::move(ye)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(x, 0.0);
  // temp compute v=(Omega-omega)r\sin(\theta) e^{-\rho}
  get<0>(spatial_velocity) =
      (vars->fluid_velocity.value() - vars->metric_data->omega) * vars->radius *
      vars->sin_theta * exp(-vars->metric_data->rho);
  get<1>(spatial_velocity) =
      get<0>(spatial_velocity)  // temp for v
      * cos(vars->phi) *
      exp(0.5 * (vars->metric_data->rho - vars->metric_data->gamma));
  get<0>(spatial_velocity) *=
      -sin(vars->phi) *
      exp(0.5 * (vars->metric_data->rho - vars->metric_data->gamma));
  return {std::move(spatial_velocity)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  auto lorentz_factor = make_with_value<Scalar<DataType>>(get<0>(x), 0.0);
  // Compute spatial proper velocity, then Lorentz factor.
  get(lorentz_factor) =
      (vars->fluid_velocity.value() - vars->metric_data->omega) * vars->radius *
      vars->sin_theta * exp(-vars->metric_data->rho);
  get(lorentz_factor) = 1.0 / sqrt(1.0 - square(get(lorentz_factor)));
  return {std::move(lorentz_factor)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> /*vars*/,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> /*vars*/,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::Lapse<DataType>> RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return lapse(vars->metric_data->gamma, vars->metric_data->rho);
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<DataType, 3>> RotatingStar::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::Shift<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return shift(vars->metric_data->omega, vars->phi, vars->radius,
               vars->sin_theta);
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, 3>>
RotatingStar::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::SpatialMetric<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return spatial_metric(vars->metric_data->gamma, vars->metric_data->rho,
                        vars->metric_data->alpha, vars->phi);
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>
RotatingStar::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return sqrt_det_spatial_metric(vars->metric_data->gamma,
                                 vars->metric_data->rho,
                                 vars->metric_data->alpha);
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<DataType, 3>>
RotatingStar::variables(
    gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataType, 3>> /*meta*/) const {
  interpolate_vars_if_necessary(vars);
  return inverse_spatial_metric(vars->metric_data->gamma,
                                vars->metric_data->rho,
                                vars->metric_data->alpha, vars->phi);
}

template <typename DataType>
auto RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  interpolate_deriv_vars_if_necessary(vars);
  // Do 2nd-order FD
  auto deriv_lapse =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  for (size_t d = 0; d < 3; ++d) {
    const auto lapse_upper =
        lapse(gsl::at(vars->metric_data_upper.value(), d).gamma,
              gsl::at(vars->metric_data_upper.value(), d).rho);
    const auto lapse_lower =
        lapse(gsl::at(vars->metric_data_lower.value(), d).gamma,
              gsl::at(vars->metric_data_lower.value(), d).rho);
    deriv_lapse.get(d) =
        (0.5 / vars->delta_r) * (get(lapse_upper) - get(lapse_lower));
  }
  return deriv_lapse;
}

template <typename DataType>
auto RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<DerivShift<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  interpolate_deriv_vars_if_necessary(vars);
  // Do 2nd-order FD
  auto deriv_shift =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  for (size_t d = 0; d < 3; ++d) {
    const auto shift_upper =
        shift(gsl::at(vars->metric_data_upper.value(), d).omega,
              gsl::at(vars->phi_upper, d), gsl::at(vars->radius_upper, d),
              gsl::at(vars->sin_theta_upper, d));
    const auto shift_lower =
        shift(gsl::at(vars->metric_data_lower.value(), d).omega,
              gsl::at(vars->phi_lower, d), gsl::at(vars->radius_lower, d),
              gsl::at(vars->sin_theta_lower, d));
    for (size_t i = 0; i < 3; ++i) {
      deriv_shift.get(d, i) =
          (0.5 / vars->delta_r) * (shift_upper.get(i) - shift_lower.get(i));
    }
  }
  return deriv_shift;
}

template <typename DataType>
auto RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivSpatialMetric<DataType>> {
  interpolate_deriv_vars_if_necessary(vars);
  // Do 2nd-order FD. This isn't super accurate at r=0, but it's fine everywhere
  // else.
  auto deriv_spatial_metric =
      make_with_value<tnsr::ijj<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  for (size_t d = 0; d < 3; ++d) {
    const auto spatial_metric_upper =
        spatial_metric(gsl::at(vars->metric_data_upper.value(), d).gamma,
                       gsl::at(vars->metric_data_upper.value(), d).rho,
                       gsl::at(vars->metric_data_upper.value(), d).alpha,
                       gsl::at(vars->phi_upper, d));
    const auto spatial_metric_lower =
        spatial_metric(gsl::at(vars->metric_data_lower.value(), d).gamma,
                       gsl::at(vars->metric_data_lower.value(), d).rho,
                       gsl::at(vars->metric_data_lower.value(), d).alpha,
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

template <typename DataType>
auto RotatingStar::variables(
    const gsl::not_null<IntermediateVariables<DataType>*> vars,
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, 3>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, 3>> {
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

PUP::able::PUP_ID RotatingStar::my_PUP_ID = 0;

bool operator==(const RotatingStar& lhs, const RotatingStar& rhs) {
  // The equation of state and CST solution aren't explicitly checked. However,
  // if rot_ns_filename_ and the polytropic_exponent_ and polytropic_constant_
  // are the same, then the solution and EOS should be too.
  return lhs.rot_ns_filename_ == rhs.rot_ns_filename_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const RotatingStar& lhs, const RotatingStar& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template class RotatingStar::IntermediateVariables<DTYPE(data)>;          \
  template Scalar<DTYPE(data)> RotatingStar::lapse(                         \
      const DTYPE(data) & gamma, const DTYPE(data) & rho) const;            \
  template tnsr::I<DTYPE(data), 3, Frame::Inertial> RotatingStar::shift(    \
      const DTYPE(data) & omega, const DTYPE(data) & phi,                   \
      const DTYPE(data) & radius, const DTYPE(data) & sin_theta) const;     \
  template tnsr::ii<DTYPE(data), 3, Frame::Inertial>                        \
  RotatingStar::spatial_metric(                                             \
      const DTYPE(data) & gamma, const DTYPE(data) & rho,                   \
      const DTYPE(data) & alpha, const DTYPE(data) & phi) const;            \
  template tnsr::II<DTYPE(data), 3, Frame::Inertial>                        \
  RotatingStar::inverse_spatial_metric(                                     \
      const DTYPE(data) & gamma, const DTYPE(data) & rho,                   \
      const DTYPE(data) & alpha, const DTYPE(data) & phi) const;            \
  template Scalar<DTYPE(data)> RotatingStar::sqrt_det_spatial_metric(       \
      const DTYPE(data) & gamma, const DTYPE(data) & rho,                   \
      const DTYPE(data) & alpha) const;                                     \
  template void RotatingStar::interpolate_vars_if_necessary(                \
      const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars) const; \
  template void RotatingStar::interpolate_deriv_vars_if_necessary(          \
      const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE

#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                     \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>                 \
      RotatingStar::variables(                                           \
          const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars, \
          const tnsr::I<DTYPE(data), 3>& x,                              \
          tmpl::list<TAG(data) < DTYPE(data)>> /*meta*/) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::LorentzFactor,
     hydro::Tags::SpecificEnthalpy, gr::Tags::Lapse,
     gr::Tags::SqrtDetSpatialMetric))

#undef INSTANTIATE_SCALARS

#define INSTANTIATE_MHD_VECTORS(_, data)                                     \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3, Frame::Inertial>> \
      RotatingStar::variables(                                               \
          const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars,     \
          const tnsr::I<DTYPE(data), 3>& x,                                  \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial>> /*meta*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE_MHD_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef INSTANTIATE_MHD_VECTORS

#define INSTANTIATE_METRIC_TENSORS(_, data)                                   \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data),                     \
      3 >> RotatingStar::variables(                                           \
               const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars, \
               const tnsr::I<DTYPE(data), 3>& x,                              \
               tmpl::list < TAG(data) < DTYPE(data), 3 >>                     \
               /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_METRIC_TENSORS, (double, DataVector),
                        (gr::Tags::Shift, gr::Tags::SpatialMetric,
                         gr::Tags::InverseSpatialMetric,
                         gr::Tags::ExtrinsicCurvature))

#undef INSTANTIATE_METRIC_TENSORS

#define INSTANTIATE_METRIC_DERIVS(_, data)                                   \
  template auto RotatingStar::variables(                                     \
      const gsl::not_null<IntermediateVariables<DTYPE(data)>*> vars,         \
      const tnsr::I<DTYPE(data), 3>& x, tmpl::list<TAG(data) < DTYPE(data)>> \
      /*meta*/) const->tuples::TaggedTuple<TAG(data) < DTYPE(data)>> ;

GENERATE_INSTANTIATIONS(INSTANTIATE_METRIC_DERIVS, (double, DataVector),
                        (DerivLapse, DerivShift, DerivSpatialMetric))

#undef INSTANTIATE_METRIC_DERIVS

#undef TAG
#undef DTYPE
}  // namespace RelativisticEuler::Solutions
