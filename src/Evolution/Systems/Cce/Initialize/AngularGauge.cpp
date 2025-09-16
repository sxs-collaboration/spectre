// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshFiltering.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, const size_t l_max,
    const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<tmpl::list<::Tags::TempSpinWeightedScalar<0, 2>,
                       ::Tags::TempSpinWeightedScalar<1, 2>,
                       ::Tags::TempSpinWeightedScalar<2, 0>,
                       ::Tags::TempSpinWeightedScalar<3, 2>,
                       ::Tags::TempSpinWeightedScalar<4, 2>,
                       ::Tags::TempSpinWeightedScalar<5, 2>>>
      buffers{number_of_angular_points};
  auto& surface_j_buffer = get<::Tags::TempSpinWeightedScalar<0, 2>>(buffers);
  auto& surface_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<1, 2>>(buffers);
  auto& surface_r_buffer = get<::Tags::TempSpinWeightedScalar<2, 0>>(buffers);

  auto& one_minus_y_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<3, 2>>(buffers));
  auto& one_minus_y_cubed_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<4, 2>>(buffers));
  auto& one_minus_y_fourth_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<5, 2>>(buffers));

  void (*iteration_heuristic_function)(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 2>&,
      const SpinWeighted<ComplexDataVector, 0>&, size_t) = nullptr;
  if (iteration_heuristic_ ==
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation) {
    iteration_heuristic_function = &spin_weight_1_coord_perturbation_heuristic;
  } else if (iteration_heuristic_ ==
             ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
                 OnlyVaryGaugeD) {
    iteration_heuristic_function = &only_vary_gauge_d_heuristic;
  } else {  // LCOV_EXCL_LINE
    // LCOV_EXCL_START
    ERROR("Unknown ConformalFactorIterationHeuristic");
    // LCOV_EXCL_STOP
  }

  auto iteration_function =
      [&iteration_heuristic_function, &filtered_gauge_omega, &gauge_omega,
       &target_omega, &interpolated_target_gauge_omega,
       &gauge_omega_transform_buffer, &l_max, &surface_r_buffer,
       &input_j_buffer, &r,
       this](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                 gauge_c_step,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                 gauge_d_step,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
             const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        iteration_interpolator.interpolate(
            make_not_null(&interpolated_target_gauge_omega), target_omega);
        if (use_input_modes_ and use_beta_integral_estimate_) {
          // when using input modes, the `input_j_buffer` stores the
          // 1/r part of J in the evolution gauge
          iteration_interpolator.interpolate(
              make_not_null(&get(surface_r_buffer)), get(r));
          get(surface_r_buffer).data() *= get(gauge_omega).data();
          interpolated_target_gauge_omega.data() /= pow(
              1.0 + real(input_j_buffer.data() * conj(input_j_buffer.data()) /
                         (square(get(surface_r_buffer).data()))),
              0.125);
        }
        filtered_gauge_omega = get(gauge_omega);
        if (not optimize_l_0_mode_) {
          Spectral::Swsh::filter_swsh_boundary_quantity(
              make_not_null(&filtered_gauge_omega), l_max, 1_st, l_max,
              make_not_null(&gauge_omega_transform_buffer));
          Spectral::Swsh::filter_swsh_boundary_quantity(
              make_not_null(&interpolated_target_gauge_omega), l_max, 1_st,
              l_max, make_not_null(&gauge_omega_transform_buffer));
        }
        double max_error = max(abs(filtered_gauge_omega.data() -
                                   interpolated_target_gauge_omega.data()));
        iteration_heuristic_function(make_not_null(&get(*gauge_c_step)),
                                     make_not_null(&get(*gauge_d_step)),
                                     get(gauge_omega), filtered_gauge_omega,
                                     interpolated_target_gauge_omega,
                                     get(gauge_c), get(gauge_d), l_max);
        return max_error;
      };

  auto finalize_function =
      [&gauge_omega, &l_max, &surface_dr_j_buffer, &boundary_dr_j, &boundary_j,
       &surface_j_buffer, &surface_r_buffer,
       &r](const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
           const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
           const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
           /*angular_cauchy_coordinates*/,
           const Spectral::Swsh::SwshInterpolator& interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>::apply(
            make_not_null(&surface_dr_j_buffer), boundary_dr_j, boundary_j,
            gauge_c, gauge_d, gauge_omega, interpolator, l_max);
        GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
            make_not_null(&surface_j_buffer), boundary_j, gauge_c, gauge_d,
            gauge_omega, interpolator);
        GaugeAdjustedBoundaryValue<Tags::BondiR>::apply(
            make_not_null(&surface_r_buffer), r, gauge_omega, interpolator);
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_, finalize_function);

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

void ConformalFactor::pup(PUP::er& p) {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | require_convergence_;
  p | optimize_l_0_mode_;
  p | use_beta_integral_estimate_;
  p | iteration_heuristic_;
  p | use_input_modes_;
  p | input_modes_;
  p | input_mode_filename_;
}

PUP::able::PUP_ID ConformalFactor::my_PUP_ID = 0;
std::ostream& operator<<(
    std::ostream& os,
    const Cce::InitializeJ::ConformalFactorIterationHeuristic& heuristic_type) {
  switch (heuristic_type) {
    case Cce::InitializeJ::ConformalFactorIterationHeuristic::
        SpinWeight1CoordPerturbation:
      return os << "SpinWeight1CoordPerturbation";
    case Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD:
      return os << "OnlyVaryGaugeD";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Unknown ConformalFactorIterationHeuristic");
      // LCOV_EXCL_STOP
  }
}
}  // namespace Cce::InitializeJ

template <>
Cce::InitializeJ::ConformalFactorIterationHeuristic
Options::create_from_yaml<Cce::InitializeJ::ConformalFactorIterationHeuristic>::
    create<void>(const Options::Option& options) {
  const auto heuristic_read = options.parse_as<std::string>();
  if ("SpinWeight1CoordPerturbation" == heuristic_read) {
    return Cce::InitializeJ::ConformalFactorIterationHeuristic::
        SpinWeight1CoordPerturbation;
  } else if ("OnlyVaryGaugeD" == heuristic_read) {
    return Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD;
  }
  // LCOV_EXCL_START
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << heuristic_read
          << "\" to Cce::InitializeJ::ConformalFactorIterationHeuristic. "
             "Must be one of SpinWeight1CoordPerturbation, OnlyVaryGaugeD.");
  // LCOV_EXCL_STOP
}
