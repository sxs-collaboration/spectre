// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

namespace {
// `max(abs(...))` can drop a NaN that is not the first element, so check every
// element explicitly.
bool all_finite(const ComplexDataVector& values) {
  return std::ranges::all_of(values, [](const std::complex<double>& value) {
    return std::isfinite(value.real()) and std::isfinite(value.imag());
  });
}
}  // namespace

namespace CauchySecondOrder_detail {
void radial_ansatz_coefficients(const gsl::not_null<ComplexDataVector*> j0,
                                const gsl::not_null<ComplexDataVector*> j1,
                                const gsl::not_null<ComplexDataVector*> j3,
                                const ComplexDataVector& j2,
                                const ComplexDataVector& boundary_j,
                                const ComplexDataVector& boundary_dr_j,
                                const ComplexDataVector& boundary_dy_dy_j,
                                const ComplexDataVector& boundary_r) {
  *j0 = boundary_j + boundary_r * boundary_dr_j +
        (4.0 / 3.0) * boundary_dy_dy_j + j0_x_coefficient * j2;
  *j1 = -(0.5 * boundary_r * boundary_dr_j + boundary_dy_dy_j) +
        j1_x_coefficient * j2;
  *j3 = (1.0 / 12.0) * boundary_dy_dy_j + j3_x_coefficient * j2;
}

size_t solve_asymptotic_j2(const gsl::not_null<ComplexDataVector*> j2,
                           const gsl::not_null<double*> final_step,
                           const ComplexDataVector& j0_at_zero,
                           const ComplexDataVector& j1_at_zero,
                           const double tolerance,
                           const size_t max_iterations) {
  *j2 = ComplexDataVector{j0_at_zero.size(), 0.0};
  *final_step = std::numeric_limits<double>::infinity();
  ComplexDataVector j0 = j0_at_zero;
  ComplexDataVector j1 = j1_at_zero;
  size_t iterations = 0;
  // Written as `not (a < b)` so that a NaN step continues rather than ends the
  // iteration as if converged.
  while (iterations < max_iterations and not(*final_step < tolerance)) {
    const DataVector k1 = real(j0 * conj(j1)) / sqrt(1.0 + real(j0 * conj(j0)));
    const ComplexDataVector next_j2 =
        0.5 * j0 * (real(j1 * conj(j1)) - k1 * k1);
    *final_step = max(abs(next_j2 - *j2));
    *j2 = next_j2;
    ++iterations;
    if (not all_finite(*j2)) {
      *final_step = std::numeric_limits<double>::quiet_NaN();
      break;
    }
    j0 = j0_at_zero + j0_x_coefficient * (*j2);
    j1 = j1_at_zero + j1_x_coefficient * (*j2);
  }
  return iterations;
}
}  // namespace CauchySecondOrder_detail

CauchySecondOrder::CauchySecondOrder(
    const double j0_tolerance, const size_t j0_max_iterations,
    const double max_cauchy_j0, const double j2_tolerance,
    const size_t j2_max_iterations, const double max_partially_flat_j2,
    std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator)
    : j0_tolerance_{j0_tolerance},
      j0_max_iterations_{j0_max_iterations},
      max_cauchy_j0_{max_cauchy_j0},
      j2_tolerance_{j2_tolerance},
      j2_max_iterations_{j2_max_iterations},
      max_partially_flat_j2_{max_partially_flat_j2},
      du_dr_j_interpolator_{std::move(du_dr_j_interpolator)} {}

std::unique_ptr<InitializeJ<false>> CauchySecondOrder::get_clone() const {
  return std::make_unique<CauchySecondOrder>(
      j0_tolerance_, j0_max_iterations_, max_cauchy_j0_, j2_tolerance_,
      j2_max_iterations_, max_partially_flat_j2_, du_dr_j_interpolator());
}

std::unique_ptr<intrp::SpanInterpolator>
CauchySecondOrder::du_dr_j_interpolator() const {
  return du_dr_j_interpolator_ == nullptr ? nullptr
                                          : du_dr_j_interpolator_->get_clone();
}

void CauchySecondOrder::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_q,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_du_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Solve the H hypersurface equation at y = -1 for dy^2 J using the worldtube
  // boundary data. The volume J is then the cubic polynomial in (1 - y) that
  // matches J, dr_j, and dy^2 J at the worldtube and satisfies the partially
  // flat gauge condition on its second asymptotic coefficient.
  Scalar<SpinWeighted<ComplexDataVector, 2>> boundary_dy2_j{
      number_of_angular_points};
  CauchySecondOrder_detail::compute_dy_dy_j(
      make_not_null(&boundary_dy2_j), boundary_j, boundary_u, boundary_w,
      boundary_beta, boundary_q, boundary_du_j, boundary_dr_j, boundary_du_dr_j,
      boundary_du_r, r, l_max);

  // The three worldtube values matched above supply only three of the four
  // conditions on the coefficients of the ansatz
  //
  //   J = J^(0) + J^(1) (1 - y) + x (1 - y)^2 + J^(3) (1 - y)^3 .
  //
  // The fourth is the partially flat gauge condition on the second asymptotic
  // coefficient, which constrains x = J^(2) nonlinearly (see below). Because
  // the matching conditions are linear, every other coefficient depends on x
  // affinely, J^(n)(x) = J^(n)(0) + alpha^(n) x (see
  // `CauchySecondOrder_detail::j0_x_coefficient` and friends). Hold on to the
  // x-independent parts, which are fixed by the worldtube data alone.
  ComplexDataVector j0_at_zero{number_of_angular_points};
  ComplexDataVector j1_at_zero{number_of_angular_points};
  ComplexDataVector j3{number_of_angular_points};
  CauchySecondOrder_detail::radial_ansatz_coefficients(
      make_not_null(&j0_at_zero), make_not_null(&j1_at_zero),
      make_not_null(&j3), ComplexDataVector{number_of_angular_points, 0.0},
      get(boundary_j).data(), get(boundary_dr_j).data(),
      get(boundary_dy2_j).data(), get(r).data());

  // The angular solve can only eliminate J at scri+ through a well-behaved
  // alteration of the spherical mesh, so it tolerates only a small asymptotic
  // J, which here is J^(0). Guard against a large one before going any
  // further: the fixed-point solve below contracts only for small data too, so
  // it would otherwise fail first and less informatively.
  const double max_asymptotic_j = max(abs(j0_at_zero));
  if (max_asymptotic_j > max_cauchy_j0_) {
    ERROR(
        "The asymptotic value of the initial J in Cauchy coordinates has "
        "magnitude "
        << max_asymptotic_j << ", which exceeds the threshold "
        << max_cauchy_j0_
        << " set by the MaxCauchyJ0 option, so the angular-coordinate solve "
           "is unlikely to converge. This usually means the worldtube is too "
           "close to the strong-field region for the worldtube data to be "
           "resolved at their angular resolution. Use a worldtube at a larger "
           "extraction radius or worldtube data with a higher angular "
           "resolution. Raising MaxCauchyJ0 only lets the angular solve "
           "start; it does not make it converge.");
  }

  // Solve the gauge constraint on x by fixed-point iteration from x = 0; see
  // `CauchySecondOrder_detail::solve_asymptotic_j2`. A zero iteration budget
  // keeps x = 0.
  ComplexDataVector j2{number_of_angular_points, 0.0};
  double j2_step = std::numeric_limits<double>::infinity();
  const size_t j2_iterations = CauchySecondOrder_detail::solve_asymptotic_j2(
      make_not_null(&j2), make_not_null(&j2_step), j0_at_zero, j1_at_zero,
      j2_tolerance_, j2_max_iterations_);
  if (j2_max_iterations_ > 0) {
    if (std::isnan(j2_step)) {
      ERROR(
          "The initial J^(2) fixed-point solve produced a non-finite J^(2) "
          "after "
          << j2_iterations
          << " iterations. Either the worldtube data contain non-finite "
             "values, or they are far too large for the iteration to "
             "contract. Check the worldtube data.");
    }
    // Written as `not (a < b)` to match the loop condition of the solve.
    if (not(j2_step < j2_tolerance_)) {
      ERROR(
          "The initial J^(2) fixed-point solve did not reach target "
          "tolerance "
          << j2_tolerance_ << " after " << j2_iterations
          << " iterations (last step " << j2_step
          << "). Increase J2MaxIterations or loosen J2Tolerance. If the step "
             "is not shrinking, the worldtube data are too large for the "
             "iteration to contract.");
    }
  }
  ComplexDataVector j0{number_of_angular_points};
  ComplexDataVector j1{number_of_angular_points};
  CauchySecondOrder_detail::radial_ansatz_coefficients(
      make_not_null(&j0), make_not_null(&j1), make_not_null(&j3), j2,
      get(boundary_j).data(), get(boundary_dr_j).data(),
      get(boundary_dy2_j).data(), get(r).data());

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
        get(boundary_j).size()};
    angular_view_j = j0 + one_minus_y_collocation[i] * j1 +
                     square(one_minus_y_collocation[i]) * j2 +
                     pow<3>(one_minus_y_collocation[i]) * j3;
  }

  // Iteratively adjust the angular coordinates so that J vanishes at scri+
  // (identical to the procedure in NoIncomingRadiation).
  const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
  make_const_view(make_not_null(&j_at_scri_view), get(*j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>>>
      iteration_buffers{number_of_angular_points};

  auto& evolution_gauge_surface_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          iteration_buffers));
  auto& interpolated_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          iteration_buffers));
  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          iteration_buffers);

  auto iteration_function =
      [&interpolated_k, &gauge_omega, &evolution_gauge_surface_j,
       &j_at_scri_view](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              gauge_c_step,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              gauge_d_step,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        iteration_interpolator.interpolate(
            make_not_null(&evolution_gauge_surface_j), j_at_scri_view);
        interpolated_k.data() =
            sqrt(1.0 + evolution_gauge_surface_j.data() *
                           conj(evolution_gauge_surface_j.data()));
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        evolution_gauge_surface_j.data() =
            0.25 *
            (square(conj(get(gauge_d).data())) *
                 evolution_gauge_surface_j.data() +
             square(get(gauge_c).data()) *
                 conj(evolution_gauge_surface_j.data()) +
             2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
                 interpolated_k.data()) /
            square(get(gauge_omega).data());

        const double max_error = max(abs(evolution_gauge_surface_j.data()));
        get(*gauge_c_step).data() =
            -0.5 * evolution_gauge_surface_j.data() *
            square(get(gauge_omega).data()) /
            (get(gauge_d).data() * interpolated_k.data());
        get(*gauge_d_step).data() = get(*gauge_c_step).data() *
                                    conj(get(gauge_c).data()) /
                                    conj(get(gauge_d).data());
        return max_error;
      };

  // Set by `finalize_function`, and checked once the angular solve has
  // succeeded so that a failure to converge is reported first.
  double max_scri_j2 = std::numeric_limits<double>::signaling_NaN();
  auto finalize_function =
      [&j, &gauge_omega, &l_max, &max_scri_j2, number_of_radial_points,
       number_of_angular_points](
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
              local_angular_cauchy_coordinates,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        GaugeAdjustInitialJ::apply(j, gauge_c, gauge_d, gauge_omega,
                                   local_angular_cauchy_coordinates,
                                   interpolator, l_max);

        // Measure the partially flat constraints after the gauge
        // transformation.
        Scalar<SpinWeighted<ComplexDataVector, 2>> dy_j{
            number_of_angular_points * number_of_radial_points};
        Scalar<SpinWeighted<ComplexDataVector, 2>> dy_dy_j{
            number_of_angular_points * number_of_radial_points};
        Tags::DyCompute<Tags::BondiJ>::function(make_not_null(&dy_j), *j,
                                                l_max);
        Tags::DyCompute<Tags::Dy<Tags::BondiJ>>::function(
            make_not_null(&dy_dy_j), dy_j, l_max);
        const SpinWeighted<ComplexDataVector, 2> scri_dy_dy_j;
        make_const_view(
            make_not_null(&scri_dy_dy_j), get(dy_dy_j),
            (number_of_radial_points - 1) * number_of_angular_points,
            number_of_angular_points);
        const SpinWeighted<ComplexDataVector, 2> scri_j;
        make_const_view(
            make_not_null(&scri_j), get(*j),
            (number_of_radial_points - 1) * number_of_angular_points,
            number_of_angular_points);
        max_scri_j2 = 0.5 * max(abs(scri_dy_dy_j.data()));
        Parallel::printf(
            "CauchySecondOrder partially flat constraint norms at scri+: "
            "max|J0| = %.16e, max|J2| = %.16e (J2 = 0.5 * Dy^2 J)\n",
            max(abs(scri_j.data())), max_scri_j2);
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      j0_tolerance_, j0_max_iterations_, max_cauchy_j0_, iteration_function,
      /*require_convergence=*/true, finalize_function);

  // Written as `not (a <= b)` so that a NaN also aborts.
  if (not(max_scri_j2 <= max_partially_flat_j2_)) {
    ERROR(
        "After the gauge transformation the initial J has max|J2| = "
        "max|(1/2) Dy^2 J| at scri+ of "
        << max_scri_j2 << ", which exceeds the threshold "
        << max_partially_flat_j2_
        << " set by the MaxPartiallyFlatJ2 option. The partially flat gauge "
           "condition on the second asymptotic coefficient holds only up to "
           "discretization error, so this usually means the angular or "
           "radial resolution is too low for the worldtube data. Increase "
           "LMax or NumberOfRadialPoints, or raise MaxPartiallyFlatJ2.");
  }
}

void CauchySecondOrder::pup(PUP::er& p) {
  p | j0_tolerance_;
  p | j0_max_iterations_;
  p | max_cauchy_j0_;
  p | j2_tolerance_;
  p | j2_max_iterations_;
  p | max_partially_flat_j2_;
  p | du_dr_j_interpolator_;
}

PUP::able::PUP_ID CauchySecondOrder::my_PUP_ID = 0;  // NOLINT
}  // namespace Cce::InitializeJ
