// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Options/String.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ {
struct LinearizedBondiSachs;
}  // namespace Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ
/// \endcond

namespace Cce {
/// Contains utilities and \ref DataBoxGroup mutators for generating data for
/// \f$J\f$ on the initial CCE hypersurface.
namespace InitializeJ {

namespace detail {
// used to provide a default for the finalize functor in
// `iteratively_adapt_angular_coordinates`
struct NoOpFinalize {
  void operator()(
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*gauge_c*/,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*gauge_d*/,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
      /*angular_cauchy_coordinates*/,
      const Spectral::Swsh::SwshInterpolator& /*interpolator*/) const {}
};

// eth of the Cartesian coordinates x^i on the identity grid. The angular solves
// interpolate these onto the current map at each pass, rather than
// differentiating the displaced coordinates, which aliases and raises the
// residual floor.
void unit_sphere_eth_cartesian_coordinates(
    gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> eth_x,
    gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> eth_y,
    gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> eth_z, size_t l_max);

// Normalize the Cartesian coordinates of the current map, and update its
// angular coordinates, interpolator and Jacobians to match.
void update_angular_coordinates_and_jacobians(
    gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    gsl::not_null<Spectral::Swsh::SwshInterpolator*> interpolator,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> gauge_c,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> gauge_d,
    size_t l_max);

// What an angular solve does when it stops above its tolerance.
enum class UnconvergedAngularSolve : uint8_t { Error, Warn, Silent };

// The residual of the map an angular solve hands back, and the number of
// coordinate updates it performed.
struct AngularSolveResult {
  double max_error = std::numeric_limits<double>::signaling_NaN();
  size_t number_of_coordinate_updates = 0;
};

// Abort if `max_error` exceeds `error_threshold` or is not a number.
void check_angular_solve_error_threshold(double max_error,
                                         double error_threshold);

// Report an angular solve that stopped above its tolerance as an error, a
// warning or not at all. The messages describe `tolerance` as a target for
// `max_error`, the role it has in the default convergence test.
void report_unconverged_angular_solve(UnconvergedAngularSolve if_unconverged,
                                      double tolerance,
                                      size_t number_of_iterations,
                                      double max_error);

// used to provide a default for the convergence test in
// `iteratively_adapt_angular_coordinates`
struct ResidualBelowTolerance {
  bool operator()(const double max_error, const double tolerance) const {
    return max_error < tolerance;
  }
};

// Records the running minimum of the residuals of an angular solve, and tells
// when it has improved by less than a relative `tolerance` over the last
// `window` passes. Unlike the change between passes, this is not held open by
// round-off noise on a slow descent.
class ResidualPlateau {
 public:
  explicit ResidualPlateau(size_t window);

  void record(double residual);

  // False until `window + 1` passes are recorded.
  bool stopped_improving(double tolerance) const;

 private:
  size_t window_;
  std::vector<double> smallest_residual_;
};

// perform an iterative solve for the set of angular coordinates. The iteration
// callable `iteration_function` must have function signature:
//
// double iteration_function(
//     const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
//         gauge_c_step,
//     const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
//         gauge_d_step,
//     const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
//     const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
//     const Spectral::Swsh::SwshInterpolator& iteration_interpolator);
//
// but need not be a function pointer -- a callable class or lambda will also
// suffice.
// For each step specified by the iteration function, the coordinates are
// updated via \hat \eth \delta x^i = \delta c \eth x^i|_{x^i=\hat x^i}
//                        + \delta \bar d (\eth x^i)|_{x^i=\hat x^i}
// This coordinate update is exact, and comes from expanding the chain rule to
// determine Jacobian factors. However, the result is not guaranteed to
// produce the desired Jacobian c and d, because \delta c and \delta d are
// not necessarily consistent with the underlying coordinates.
// We then update the x^i by inverting \hat \eth, which is also exact, but
// assumes a no l=0 contribution to the transformation.
// Depending on the choice of approximations used to specify
// `iteration_function`, though, the method can be slow to converge.

// However, the iterations are typically fast, and the computation is for
// initial data that needs to be computed only once during a simulation, so it
// is not currently an optimization priority. If this function becomes a
// bottleneck, the numerical procedure of the iterative method or the choice of
// approximation used for `iteration_function` should be revisited.
//
// The solve stops once `has_converged(max_error, tolerance)` holds (by default
// `max_error < tolerance`) or after `max_steps` steps, and hands back the best
// map it evaluated. `has_converged` is called once after each evaluation of
// `iteration_function` in the loop, but not after the final re-evaluation of
// the best map. A caller with another convergence test passes
// `UnconvergedAngularSolve::Silent` and reports a failure itself, since the
// other modes describe `tolerance` as a target for `max_error`. With
// `initialize_coordinates = false` it continues from the map the caller
// supplies.
template <typename IterationFunctor, typename FinalizeFunctor = NoOpFinalize,
          typename ConvergenceTest = ResidualBelowTolerance>
AngularSolveResult iteratively_adapt_angular_coordinates(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const size_t l_max, const double tolerance, const size_t max_steps,
    const double error_threshold, const IterationFunctor& iteration_function,
    const UnconvergedAngularSolve if_unconverged,
    const FinalizeFunctor finalize_function = NoOpFinalize{},
    const bool initialize_coordinates = true,
    const ConvergenceTest& has_converged = ResidualBelowTolerance{}) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  if (initialize_coordinates) {
    Spectral::Swsh::create_angular_and_cartesian_coordinates(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
  }

  Variables<tmpl::list<
      // eth of cartesian coordinates
      ::Tags::TempSpinWeightedScalar<3, 1>,
      ::Tags::TempSpinWeightedScalar<4, 1>,
      ::Tags::TempSpinWeightedScalar<5, 1>,
      // eth of gauge-transformed cartesian coordinates
      ::Tags::TempSpinWeightedScalar<6, 1>,
      ::Tags::TempSpinWeightedScalar<7, 1>,
      ::Tags::TempSpinWeightedScalar<8, 1>,
      // gauge Jacobians
      ::Tags::TempSpinWeightedScalar<9, 2>,
      ::Tags::TempSpinWeightedScalar<10, 0>,
      // gauge Jacobians on next iteration
      ::Tags::TempSpinWeightedScalar<11, 2>,
      ::Tags::TempSpinWeightedScalar<12, 0>,
      // cartesian coordinates steps
      ::Tags::TempSpinWeightedScalar<13, 0>,
      ::Tags::TempSpinWeightedScalar<14, 0>,
      ::Tags::TempSpinWeightedScalar<15, 0>>>
      computation_buffers{number_of_angular_points};

  auto& eth_x =
      get(get<::Tags::TempSpinWeightedScalar<3, 1>>(computation_buffers));
  auto& eth_y =
      get(get<::Tags::TempSpinWeightedScalar<4, 1>>(computation_buffers));
  auto& eth_z =
      get(get<::Tags::TempSpinWeightedScalar<5, 1>>(computation_buffers));
  unit_sphere_eth_cartesian_coordinates(make_not_null(&eth_x),
                                        make_not_null(&eth_y),
                                        make_not_null(&eth_z), l_max);

  auto& evolution_gauge_eth_x_step =
      get(get<::Tags::TempSpinWeightedScalar<6, 1>>(computation_buffers));
  auto& evolution_gauge_eth_y_step =
      get(get<::Tags::TempSpinWeightedScalar<7, 1>>(computation_buffers));
  auto& evolution_gauge_eth_z_step =
      get(get<::Tags::TempSpinWeightedScalar<8, 1>>(computation_buffers));

  auto& gauge_c =
      get<::Tags::TempSpinWeightedScalar<9, 2>>(computation_buffers);
  auto& gauge_d =
      get<::Tags::TempSpinWeightedScalar<10, 0>>(computation_buffers);

  auto& gauge_c_step =
      get<::Tags::TempSpinWeightedScalar<11, 2>>(computation_buffers);
  auto& gauge_d_step =
      get<::Tags::TempSpinWeightedScalar<12, 0>>(computation_buffers);

  auto& x_step =
      get(get<::Tags::TempSpinWeightedScalar<13, 0>>(computation_buffers));
  auto& y_step =
      get(get<::Tags::TempSpinWeightedScalar<14, 0>>(computation_buffers));
  auto& z_step =
      get(get<::Tags::TempSpinWeightedScalar<15, 0>>(computation_buffers));

  double max_error = 1.0;
  size_t number_of_steps = 0;
  bool converged = false;
  Spectral::Swsh::SwshInterpolator iteration_interpolator;

  const auto evaluate_current_map = [&]() {
    update_angular_coordinates_and_jacobians(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates,
        make_not_null(&iteration_interpolator), make_not_null(&gauge_c),
        make_not_null(&gauge_d), l_max);
    return iteration_function(make_not_null(&gauge_c_step),
                              make_not_null(&gauge_d_step), gauge_c, gauge_d,
                              iteration_interpolator);
  };

  // The best map is copied before `evaluate_current_map` normalizes it, so that
  // evaluating it again on the rewind reproduces its residual bit for bit.
  auto unevaluated_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
  auto best_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
  double best_max_error = std::numeric_limits<double>::max();

  while (true) {
    unevaluated_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
    max_error = evaluate_current_map();
    if (max_error < best_max_error) {
      best_max_error = max_error;
      best_cartesian_cauchy_coordinates =
          unevaluated_cartesian_cauchy_coordinates;
    }

    check_angular_solve_error_threshold(max_error, error_threshold);
    ++number_of_steps;
    converged = has_converged(max_error, tolerance);
    if (converged or number_of_steps > max_steps) {
      break;
    }
    // using the evolution_gauge_.._step as temporary buffers for the
    // interpolation results
    iteration_interpolator.interpolate(
        make_not_null(&evolution_gauge_eth_x_step), eth_x);
    iteration_interpolator.interpolate(
        make_not_null(&evolution_gauge_eth_y_step), eth_y);
    iteration_interpolator.interpolate(
        make_not_null(&evolution_gauge_eth_z_step), eth_z);

    evolution_gauge_eth_x_step =
        0.5 * ((get(gauge_c_step)) * conj(evolution_gauge_eth_x_step) +
               conj((get(gauge_d_step))) * evolution_gauge_eth_x_step);
    evolution_gauge_eth_y_step =
        0.5 * ((get(gauge_c_step)) * conj(evolution_gauge_eth_y_step) +
               conj((get(gauge_d_step))) * evolution_gauge_eth_y_step);
    evolution_gauge_eth_z_step =
        0.5 * ((get(gauge_c_step)) * conj(evolution_gauge_eth_z_step) +
               conj((get(gauge_d_step))) * evolution_gauge_eth_z_step);

    Spectral::Swsh::angular_derivatives<tmpl::list<
        Spectral::Swsh::Tags::InverseEth, Spectral::Swsh::Tags::InverseEth,
        Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&x_step), make_not_null(&y_step),
        make_not_null(&z_step), evolution_gauge_eth_x_step,
        evolution_gauge_eth_y_step, evolution_gauge_eth_z_step);

    get<0>(*cartesian_cauchy_coordinates) += real(x_step.data());
    get<1>(*cartesian_cauchy_coordinates) += real(y_step.data());
    get<2>(*cartesian_cauchy_coordinates) += real(z_step.data());
  }

  if (best_max_error < max_error) {
    *cartesian_cauchy_coordinates = best_cartesian_cauchy_coordinates;
    max_error = evaluate_current_map();
  }

  finalize_function(gauge_c, gauge_d, *angular_cauchy_coordinates,
                    iteration_interpolator);

  // The loop evaluates once more than it updates.
  const size_t number_of_coordinate_updates = number_of_steps - 1;
  if (not converged) {
    report_unconverged_angular_solve(if_unconverged, tolerance,
                                     number_of_coordinate_updates, max_error);
  }
  return {.max_error = max_error,
          .number_of_coordinate_updates = number_of_coordinate_updates};
}

// Solve for the partially flat angular coordinates with a displacement
// generated by a single spin-weight-0 potential \zeta,
//
//     \eta = \eth \zeta,   \delta \hat x^i = Re(\eta conj(\eth x^i)|_{\hat x}),
//
// where \eth x^i, the eth of the Cartesian coordinates on the identity grid, is
// evaluated at the current map \hat x.
// `iteratively_adapt_angular_coordinates` prescribes both Jacobian variations,
// four real functions for a map that has two, so only part of each requested
// step is realized. Here the variation of \hat c at the identity is \eth \eta,
// so the step to the target Jacobian \hat c_* is one application of \eth^{-1},
//
//     \eta = \eth^{-1}(\hat c_* - \hat c),
//
// and \hat d follows from the map. \eth^{-1} leaves the l = 1 part of \eta at
// zero: \eth annihilates it, and it is the residual conformal freedom of the
// problem. Since \hat c_* is the exact root of the gauge condition, a few
// passes do the work of many linearized sweeps.
//
// `target_function` must have the signature
//
// double target_function(
//     const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
//         gauge_c_target,
//     const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
//     const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
//     const Spectral::Swsh::SwshInterpolator& iteration_interpolator);
//
// setting `gauge_c_target` to \hat c_* and returning the current residual.
//
// The residual first falls by orders of magnitude and then creeps back up, so
// the solve stops at the first pass from the third on with
// e_n > 0.9 e_{n-1}, and hands back the best map it evaluated. The number of
// coordinate updates it reports includes the ones the rewind discarded, so that
// a caller can charge them to an iteration budget shared with another solve.
template <typename TargetFunctor, typename FinalizeFunctor = NoOpFinalize>
AngularSolveResult adapt_angular_coordinates_via_potential(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const size_t l_max, const double tolerance, const size_t max_steps,
    const double error_threshold, const TargetFunctor& target_function,
    const UnconvergedAngularSolve if_unconverged,
    const FinalizeFunctor finalize_function = NoOpFinalize{},
    const bool initialize_coordinates = true) {
  constexpr double plateau_factor = 0.9;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  if (initialize_coordinates) {
    Spectral::Swsh::create_angular_and_cartesian_coordinates(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
  }

  Variables<tmpl::list<
      // eth of the cartesian coordinates
      ::Tags::TempSpinWeightedScalar<3, 1>,
      ::Tags::TempSpinWeightedScalar<4, 1>,
      ::Tags::TempSpinWeightedScalar<5, 1>,
      // ... interpolated onto the current map
      ::Tags::TempSpinWeightedScalar<10, 1>,
      ::Tags::TempSpinWeightedScalar<11, 1>,
      ::Tags::TempSpinWeightedScalar<12, 1>,
      // gauge Jacobians
      ::Tags::TempSpinWeightedScalar<6, 2>,
      ::Tags::TempSpinWeightedScalar<7, 0>,
      // the target Jacobian, then the step to it
      ::Tags::TempSpinWeightedScalar<8, 2>,
      // eta = eth zeta
      ::Tags::TempSpinWeightedScalar<9, 1>>>
      computation_buffers{number_of_angular_points};

  auto& eth_x =
      get(get<::Tags::TempSpinWeightedScalar<3, 1>>(computation_buffers));
  auto& eth_y =
      get(get<::Tags::TempSpinWeightedScalar<4, 1>>(computation_buffers));
  auto& eth_z =
      get(get<::Tags::TempSpinWeightedScalar<5, 1>>(computation_buffers));
  auto& gauge_c =
      get<::Tags::TempSpinWeightedScalar<6, 2>>(computation_buffers);
  auto& gauge_d =
      get<::Tags::TempSpinWeightedScalar<7, 0>>(computation_buffers);
  auto& gauge_c_target =
      get<::Tags::TempSpinWeightedScalar<8, 2>>(computation_buffers);
  auto& eta =
      get(get<::Tags::TempSpinWeightedScalar<9, 1>>(computation_buffers));
  auto& interpolated_eth_x =
      get(get<::Tags::TempSpinWeightedScalar<10, 1>>(computation_buffers));
  auto& interpolated_eth_y =
      get(get<::Tags::TempSpinWeightedScalar<11, 1>>(computation_buffers));
  auto& interpolated_eth_z =
      get(get<::Tags::TempSpinWeightedScalar<12, 1>>(computation_buffers));
  unit_sphere_eth_cartesian_coordinates(make_not_null(&eth_x),
                                        make_not_null(&eth_y),
                                        make_not_null(&eth_z), l_max);

  double max_error = 1.0;
  double previous_max_error = std::numeric_limits<double>::max();
  size_t number_of_steps = 0;
  Spectral::Swsh::SwshInterpolator iteration_interpolator;

  const auto evaluate_current_map = [&]() {
    update_angular_coordinates_and_jacobians(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates,
        make_not_null(&iteration_interpolator), make_not_null(&gauge_c),
        make_not_null(&gauge_d), l_max);
    return target_function(make_not_null(&gauge_c_target), gauge_c, gauge_d,
                           iteration_interpolator);
  };

  // As in `iteratively_adapt_angular_coordinates`, the best map is copied
  // before it is normalized, so that the rewind reproduces it bit for bit.
  auto unevaluated_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
  auto best_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
  double best_max_error = std::numeric_limits<double>::max();

  while (true) {
    unevaluated_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
    max_error = evaluate_current_map();
    if (max_error < best_max_error) {
      best_max_error = max_error;
      best_cartesian_cauchy_coordinates =
          unevaluated_cartesian_cauchy_coordinates;
    }

    check_angular_solve_error_threshold(max_error, error_threshold);
    ++number_of_steps;
    if (max_error < tolerance or number_of_steps > max_steps or
        (number_of_steps > 2 and
         max_error > plateau_factor * previous_max_error)) {
      break;
    }
    previous_max_error = max_error;

    // the step to the target Jacobian, then the potential that generates it
    get(gauge_c_target).data() -= get(gauge_c).data();
    Spectral::Swsh::angular_derivatives<
        tmpl::list<Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&eta), get(gauge_c_target));

    iteration_interpolator.interpolate(make_not_null(&interpolated_eth_x),
                                       eth_x);
    iteration_interpolator.interpolate(make_not_null(&interpolated_eth_y),
                                       eth_y);
    iteration_interpolator.interpolate(make_not_null(&interpolated_eth_z),
                                       eth_z);

    // dx^i = Re(eta conj(eth x^i)) = v^A \partial_A x^i is tangent to the
    // sphere, since x^i eth x^i = 0; the radial part of its eth is removed when
    // the next pass normalizes the coordinates.
    get<0>(*cartesian_cauchy_coordinates) +=
        real(eta.data() * conj(interpolated_eth_x.data()));
    get<1>(*cartesian_cauchy_coordinates) +=
        real(eta.data() * conj(interpolated_eth_y.data()));
    get<2>(*cartesian_cauchy_coordinates) +=
        real(eta.data() * conj(interpolated_eth_z.data()));
  }

  if (best_max_error < max_error) {
    *cartesian_cauchy_coordinates = best_cartesian_cauchy_coordinates;
    max_error = evaluate_current_map();
  }

  finalize_function(gauge_c, gauge_d, *angular_cauchy_coordinates,
                    iteration_interpolator);

  // The loop evaluates once more than it updates.
  const size_t number_of_coordinate_updates = number_of_steps - 1;
  // Written as `not (a < b)` so that a NaN counts as not converged.
  if (not(max_error < tolerance)) {
    report_unconverged_angular_solve(if_unconverged, tolerance,
                                     number_of_coordinate_updates, max_error);
  }
  return {.max_error = max_error,
          .number_of_coordinate_updates = number_of_coordinate_updates};
}

double adjust_angular_coordinates_for_j(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const SpinWeighted<ComplexDataVector, 2>& surface_j, size_t l_max,
    double tolerance, size_t max_steps, bool adjust_volume_gauge);
}  // namespace detail

/*!
 * \brief Apply a radius-independent angular gauge transformation to a volume
 * \f$J\f$, for use with initial data generation.
 *
 * \details Performs the gauge transformation to \f$\hat J\f$,
 *
 * \f{align*}{
 * \hat J = \frac{1}{4 \hat{\omega}^2} \left( \bar{\hat d}^2  J(\hat x^{\hat A})
 *  + \hat c^2 \bar J(\hat x^{\hat A})
 *  + 2 \hat c \bar{\hat d} K(\hat x^{\hat A}) \right).
 * \f}
 *
 * Where \f$\hat c\f$ and \f$\hat d\f$ are the spin-weighted angular Jacobian
 * factors computed by `GaugeUpdateJacobianFromCoords`, and \f$\hat \omega\f$ is
 * the conformal factor associated with the angular coordinate transformation.
 * Note that the right-hand sides with explicit \f$\hat x^{\hat A}\f$ dependence
 * must be interpolated and that \f$K = \sqrt{1 + J \bar J}\f$.
 */
struct GaugeAdjustInitialJ {
  using boundary_tags =
      tmpl::list<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                 Tags::PartiallyFlatGaugeOmega, Tags::CauchyAngularCoords,
                 Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::BondiJ>;
  using argument_tags = tmpl::append<boundary_tags>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
          cauchy_angular_coordinates,
      const Spectral::Swsh::SwshInterpolator& interpolator, size_t l_max);
};

/// \cond
struct NoIncomingRadiation;
struct ZeroNonSmooth;
template <bool evolve_ccm>
struct InverseCubic;
template <bool evolve_ccm>
struct InitializeJ;
struct ConformalFactor;
struct CauchySecondOrder;
/// \endcond

/*!
 * \brief Abstract base class for an initial hypersurface data generator for
 * Cce, when the partially flat Bondi-like coordinates are evolved.
 *
 * \details The algorithm is same as `InitializeJ<false>`, but with an
 * additional initialization for the partially flat Bondi-like coordinates. The
 * functions that are required to be overriden in the derived classes are:
 * - `InitializeJ::get_clone()`: should return a
 * `std::unique_ptr<InitializeJ<true>>` with cloned state.
 * - `InitializeJ::operator() const`: should take as arguments, first a
 * set of `gsl::not_null` pointers represented by `mutate_tags`, followed by a
 * set of `const` references to quantities represented by `argument_tags`. \note
 * The `InitializeJ::operator()` should be const, and therefore not alter
 * the internal state of the generator. This is compatible with all known
 * use-cases and permits the `InitializeJ` generator to be placed in the
 * `GlobalCache`.
 */
template <>
struct InitializeJ<true> : public PUP::able {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>,
                                   Tags::BoundaryValue<Tags::BondiBeta>>;

  using mutate_tags =
      tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                 Tags::CauchyAngularCoords, Tags::PartiallyFlatCartesianCoords,
                 Tags::PartiallyFlatAngularCoords>;
  using return_tags = mutate_tags;
  using argument_tags =
      tmpl::push_back<boundary_tags, Tags::LMax, Tags::NumberOfRadialPoints>;

  // The evolution of inertial coordinates are allowed only when InverseCubic is
  // used
  using creatable_classes = tmpl::list<InverseCubic<true>>;

  InitializeJ() = default;
  explicit InitializeJ(CkMigrateMessage* /*msg*/) {}

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ<true>> get_clone() const = 0;

  /// \brief The interpolator this generator wants used to build the worldtube
  /// boundary value of \f$\partial_u \partial_r J\f$, or `nullptr` to use the
  /// evolution's `H5Interpolator`.
  ///
  /// \details That boundary value is consumed by the initial data alone, so a
  /// generator is free to choose its own time-interpolation order for it
  /// without affecting the evolution.
  virtual std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator()
      const {
    return nullptr;
  }

  // Each derived class declares its own `return_tags` and `argument_tags` and
  // implements a non-virtual `operator()`; the dispatch below picks the
  // correct dynamic type and forwards through `db::mutate_apply`.
  template <typename DbTags>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
    call_with_dynamic_type<void, creatable_classes>(
        this, [&](auto* const derived) {
          db::mutate_apply(*derived, box, hdf5_lock);
        });
  }
};

/*!
 * \brief Abstract base class for an initial hypersurface data generator for
 * Cce, when the partially flat Bondi-like coordinates are not evolved.
 *
 * \details The functions that are required to be overriden in the derived
 * classes are:
 * - `InitializeJ::get_clone()`: should return a
 * `std::unique_ptr<InitializeJ<false>>` with cloned state.
 * - `InitializeJ::operator() const`: should take as arguments, first a
 * set of `gsl::not_null` pointers represented by `mutate_tags`, followed by a
 * set of `const` references to quantities represented by `argument_tags`. \note
 * The `InitializeJ::operator()` should be const, and therefore not alter
 * the internal state of the generator. This is compatible with all known
 * use-cases and permits the `InitializeJ` generator to be placed in the
 * `GlobalCache`.
 */
template <>
struct InitializeJ<false> : public PUP::able {
  // Default boundary/mutate/argument tags used by the simple derived classes
  // (ConformalFactor, InverseCubic<false>, NoIncomingRadiation, ZeroNonSmooth).
  // A derived class can shadow these with its own list (see CauchySecondOrder),
  // in which case the per-class list is used during dispatch.
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>,
                                   Tags::BoundaryValue<Tags::BondiBeta>>;

  using mutate_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using return_tags = mutate_tags;
  using argument_tags =
      tmpl::push_back<boundary_tags, Tags::LMax, Tags::NumberOfRadialPoints>;

  using creatable_classes =
      tmpl::list<ConformalFactor, InverseCubic<false>, NoIncomingRadiation,
                 ZeroNonSmooth, CauchySecondOrder,
                 ::Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ::
                     LinearizedBondiSachs>;

  InitializeJ() = default;
  explicit InitializeJ(CkMigrateMessage* /*msg*/) {}

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ<false>> get_clone() const = 0;

  /// \brief The interpolator this generator wants used to build the worldtube
  /// boundary value of \f$\partial_u \partial_r J\f$, or `nullptr` to use the
  /// evolution's `H5Interpolator`.
  ///
  /// \details That boundary value is consumed by the initial data alone, so a
  /// generator is free to choose its own time-interpolation order for it
  /// without affecting the evolution.
  virtual std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator()
      const {
    return nullptr;
  }

  // Each derived class declares its own `return_tags` and `argument_tags` and
  // implements a non-virtual `operator()` whose signature matches those tags.
  // The dispatch below picks the correct dynamic type and forwards through
  // `db::mutate_apply` so the per-class tag lists are honored.
  template <typename DbTags>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
    call_with_dynamic_type<void, creatable_classes>(
        this, [&](auto* const derived) {
          db::mutate_apply(*derived, box, hdf5_lock);
        });
  }
};
}  // namespace InitializeJ
}  // namespace Cce

#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachsInitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
