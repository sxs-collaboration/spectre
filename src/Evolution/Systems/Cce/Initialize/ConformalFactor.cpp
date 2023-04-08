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
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {
namespace {
void read_modes_from_input_file(
    const gsl::not_null<ComplexModalVector*> input_modes,
    const std::string& input_filename) {
  Matrix initial_j_modes;
  {
    h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename};
    auto& dat_file = cce_data_file.get<h5::Dat>("/InitialJ");
    initial_j_modes = dat_file.get_data();
    cce_data_file.close_current_object();
  }
  for (size_t i = 0; i < input_modes->size(); ++i) {
    (*input_modes)[i] = std::complex<double>(initial_j_modes(0, 2 * i),
                                             initial_j_modes(0, 2 * i + 1));
  }
}

// This is a choice of heuristic for generating the angular
// coordinates based on assuming the spin-weighted jacobians are
// approximately \eth and \ethbar of a common spin-weight-1 quantity.
// This holds if the perturbation \delta x^A q_A happens to be representable
// as a spin-weight 1 quantity, which is an acceptable choice of
// perturbation, but is not generic to all possible coordinate
// transformations
void spin_weight_1_coord_perturbation_heuristic(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> gauge_c_step,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> gauge_d_step,
    const SpinWeighted<ComplexDataVector, 0>& full_omega,
    const SpinWeighted<ComplexDataVector, 0>& omega_filtered,
    const SpinWeighted<ComplexDataVector, 0>& target_omega,
    const SpinWeighted<ComplexDataVector, 2>& /*gauge_c*/,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d, const size_t l_max) {
  SpinWeighted<ComplexDataVector, 1> jacobian_supplement_f{full_omega.size()};

  // The alteration in each of the spin-weighted Jacobian factors determined
  // by linearizing the system in small \Delta \omega
  gauge_d_step->data() = full_omega.data() *
                         (target_omega.data() - omega_filtered.data()) /
                         gauge_d.data();
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::InverseEthbar>>(
      l_max, 1, make_not_null(&jacobian_supplement_f), *gauge_d_step);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, gauge_c_step, jacobian_supplement_f);
}

void only_vary_gauge_d_heuristic(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> gauge_c_step,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> gauge_d_step,
    const SpinWeighted<ComplexDataVector, 0>& full_omega,
    const SpinWeighted<ComplexDataVector, 0>& omega_filtered,
    const SpinWeighted<ComplexDataVector, 0>& target_omega,
    const SpinWeighted<ComplexDataVector, 2>& /*gauge_c*/,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d,
    const size_t /*l_max*/) {
  // The alteration in each of the spin-weighted Jacobian factors determined
  // by linearizing the system in small \Delta \omega
  gauge_d_step->data() = full_omega.data() *
                         (target_omega.data() - omega_filtered.data()) /
                         gauge_d.data();
  gauge_c_step->data() = 0;
}
}  // namespace

ConformalFactor::ConformalFactor(CkMigrateMessage* msg)
    : InitializeJ<false>(msg) {}

ConformalFactor::ConformalFactor(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const bool optimize_l_0_mode,
    const bool use_beta_integral_estimate,
    const ::Cce::InitializeJ::ConformalFactorIterationHeuristic
        iteration_heuristic,
    const bool use_input_modes, std::string input_mode_filename)
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      require_convergence_{require_convergence},
      optimize_l_0_mode_{optimize_l_0_mode},
      use_beta_integral_estimate_{use_beta_integral_estimate},
      iteration_heuristic_{iteration_heuristic},
      use_input_modes_{use_input_modes},
      input_mode_filename_{std::move(input_mode_filename)} {}

ConformalFactor::ConformalFactor(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const bool optimize_l_0_mode,
    const bool use_beta_integral_estimate,
    const ::Cce::InitializeJ::ConformalFactorIterationHeuristic
        iteration_heuristic,
    const bool use_input_modes,
    std::vector<std::complex<double>> input_modes)
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      require_convergence_{require_convergence},
      optimize_l_0_mode_{optimize_l_0_mode},
      use_beta_integral_estimate_{use_beta_integral_estimate},
      iteration_heuristic_{iteration_heuristic},
      use_input_modes_{use_input_modes},
      input_modes_{std::move(input_modes)} {}

std::unique_ptr<InitializeJ<false>> ConformalFactor::get_clone() const {
  return std::make_unique<ConformalFactor>(*this);
}

void ConformalFactor::operator()(
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
                       ::Tags::TempSpinWeightedScalar<2, 2>,
                       ::Tags::TempSpinWeightedScalar<3, 0>,
                       ::Tags::TempSpinWeightedScalar<4, 0>,
                       ::Tags::TempSpinWeightedScalar<5, 0>,
                       ::Tags::TempSpinWeightedScalar<6, 0>,
                       ::Tags::TempSpinWeightedScalar<7, 0>,
                       ::Tags::TempSpinWeightedScalar<8, 0>,
                       ::Tags::TempSpinWeightedScalar<9, 2>,
                       ::Tags::TempSpinWeightedScalar<10, 2>,
                       ::Tags::TempSpinWeightedScalar<11, 2>>>
      buffers{number_of_angular_points};
  auto& surface_j_buffer = get<::Tags::TempSpinWeightedScalar<0, 2>>(buffers);
  auto& surface_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<1, 2>>(buffers);
  auto& input_j_buffer =
      get(get<::Tags::TempSpinWeightedScalar<2, 2>>(buffers));
  auto& gauge_omega = get<::Tags::TempSpinWeightedScalar<4, 0>>(buffers);
  auto& filtered_gauge_omega =
      get(get<::Tags::TempSpinWeightedScalar<5, 0>>(buffers));
  auto& target_omega = get(get<::Tags::TempSpinWeightedScalar<6, 0>>(buffers));
  auto& interpolated_target_gauge_omega =
      get(get<::Tags::TempSpinWeightedScalar<7, 0>>(buffers));
  auto& surface_r_buffer = get<::Tags::TempSpinWeightedScalar<8, 0>>(buffers);

  auto& one_minus_y_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<9, 2>>(buffers));
  auto& one_minus_y_cubed_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<10, 2>>(buffers));
  auto& one_minus_y_fourth_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<11, 2>>(buffers));

  Variables<tmpl::list<::Tags::ModalTempSpinWeightedScalar<0, 2>,
                       ::Tags::ModalTempSpinWeightedScalar<1, 0>>>
      modal_buffers{Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  auto& input_j_libsharp_modes =
      get(get<::Tags::ModalTempSpinWeightedScalar<0, 2>>(modal_buffers));
  auto& gauge_omega_transform_buffer =
      get(get<::Tags::ModalTempSpinWeightedScalar<1, 0>>(modal_buffers));

  SpinWeighted<ComplexModalVector, 2> goldberg_modes{square(l_max + 1)};
  if (use_input_modes_) {
    if (input_mode_filename_.has_value()) {
      const std::lock_guard hold_lock(*hdf5_lock);
      read_modes_from_input_file(make_not_null(&(goldberg_modes.data())),
                                 input_mode_filename_.value());
    } else {
      ASSERT(input_modes_.size() <= goldberg_modes.size(),
             "The size of the input modes is too large. Specify at most  "
             "(l_max + 1)^2 modes in the input file.");
      std::fill(goldberg_modes.data().begin(), goldberg_modes.data().end(),
                0.0);
      std::copy(input_modes_.begin(), input_modes_.end(),
                goldberg_modes.data().begin());
    }
    Spectral::Swsh::goldberg_to_libsharp_modes(
        make_not_null(&input_j_libsharp_modes), goldberg_modes, l_max);
    Spectral::Swsh::inverse_swsh_transform(
        l_max, 1_st, make_not_null(&input_j_buffer), input_j_libsharp_modes);
    // input modes fix the  j^(1) / r = (j^(1) / (2 R)) (1 - y)
  }

  // The asymptotic value of beta is beta|_scri = beta|_\Gamma + \int_-1^1 dy
  // \partial_y \beta, and this estimates the second term.
  // The coordinate transform acts to set e^(2 \hat \beta) = e^(2 \beta) /
  // \omega. We organize the iteration to try to find the coordinate transform
  // that fixes \omega = e^(2 \beta - 2\hat \beta) a simple approximation is to
  // choose the coordinate transform such that at the boundary, \hat \beta = 0,
  // so we just seek a value set by the original boundary beta Alternatively, we
  // can try to choose values such that the asymptotic value of beta is zeroed.
  // The equation of motion for \beta is the same before and after the
  // coordinate transformation, but in the two cases, it is easier to compute
  // the contributions from the integral to \beta for inverse cubic or to \hat
  // \beta for input file specification.

  // So, in either case we want to fix
  // e^(2 \hat \beta|_scri+) = 1
  //
  // in the first case, we write that as
  // e^(2 \beta|_\scri^+) / \omega = 1
  // => e^(2(\int dy \partial_y \beta + \beta|_\Gamma))|_{x(\hat x)} = \omega
  // so in this case, we should compute the estimated addition to beta as a
  // one-time addition as an input to the algorithm.
  //
  // in the second case we want to write it as
  // e^(2 \hat \beta|_\Gamma + 2 \int dy \partial_y \hat \beta) = 1
  // => e^(2 \beta|_\Gamma)|_{x(\hat x)} *
  //    e^(2\int dy \partial_y \hat \beta) = \omega
  // so in this case, we can still compute the integral up front to determine
  // the estimated addition, but it should be multiplied into the target _after_
  // interpolation
  //
  // Either way, it just represents a small alteration to the method by which
  // the target is computed on each iteration, so can largely be absorbed into
  // the main algorithm.

  // we use the estimate:
  // \int dy \partial_y \beta \approx -1/16 ln(1 + 4.0 * j jbar)

  target_omega.data() = exp(2.0 * get(beta).data());
  if (not use_input_modes_ and use_beta_integral_estimate_) {
    // use buffer for the (1-y) coefficient that we'd generate in the original
    // gauge.
    get(surface_j_buffer) = 0.25 * (3.0 * get(boundary_j).data() +
                                    get(r).data() * get(boundary_dr_j).data());
    target_omega.data() /= pow(1.0 + 4.0 * get(surface_j_buffer).data() *
                                         conj(get(surface_j_buffer).data()),
                               0.125);
  }

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
  if (not use_input_modes_) {
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
  } else {
    // chosen so that:
    // - asymptotic 1/r part matches input modes
    // - matches j and dr_j on the worldtube
    one_minus_y_coefficient = 0.5 * input_j_buffer / get(surface_r_buffer);
    one_minus_y_cubed_coefficient =
        -0.75 * one_minus_y_coefficient + 0.5 * get(surface_j_buffer) +
        0.125 * get(surface_r_buffer) * get(surface_dr_j_buffer);
    one_minus_y_fourth_coefficient =
        0.25 * one_minus_y_coefficient - 0.1875 * get(surface_j_buffer) -
        0.0625 * get(surface_r_buffer) * get(surface_dr_j_buffer);
    for (size_t i = 0; i < number_of_radial_points; i++) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + get(boundary_j).size() * i,
          get(boundary_j).size()};
      angular_view_j =
          one_minus_y_collocation[i] * one_minus_y_coefficient.data() +
          pow<3>(one_minus_y_collocation[i]) *
              one_minus_y_cubed_coefficient.data() +
          pow<4>(one_minus_y_collocation[i]) *
              one_minus_y_fourth_coefficient.data();
    }
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
