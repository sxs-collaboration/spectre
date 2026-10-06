// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce::InitializeJ {

namespace CauchySecondOrder_detail {

/// \brief The dependence of the coefficients of the Cauchy-gauge ansatz on
/// the free parameter \f$x = \tilde J^{(2)}\f$.
///
/// \details The matching conditions at the worldtube are linear, so each
/// coefficient \f$\tilde J^{(n)}\f$ is an affine function of \f$x\f$ with slope
/// \f$\alpha^{(n)} = (-1)^n 2^{3-n} / (n! (3-n)!)\f$.
/// @{
constexpr double j0_x_coefficient = 4.0 / 3.0;
constexpr double j1_x_coefficient = -2.0;
constexpr double j3_x_coefficient = -1.0 / 6.0;
/// @}

/*!
 * \brief Compute the coefficients \f$\tilde J^{(0)}\f$, \f$\tilde J^{(1)}\f$
 * and \f$\tilde J^{(3)}\f$ of the Cauchy-gauge ansatz for a given
 * \f$x = \tilde J^{(2)}\f$.
 *
 * \details The ansatz is
 *
 * \f{align*}{
 *   \tilde J(\tilde y; x) = \tilde J^{(0)}(x)
 *     + \tilde J^{(1)}(x)\,(1 - \tilde y) + x\,(1 - \tilde y)^2
 *     + \tilde J^{(3)}(x)\,(1 - \tilde y)^3 ,
 * \f}
 *
 * and its coefficients are fixed by matching the worldtube
 * (\f$\tilde y = -1\f$) values \f$\tilde J|_\Gamma\f$,
 * \f$\partial_{\tilde y} \tilde J|_\Gamma = (R/2)\, \partial_r J|_\Gamma\f$ and
 * \f$\partial_{\tilde y}^2 \tilde J|_\Gamma\f$, where \f$R\f$ is the worldtube
 * radius. They are affine in \f$x\f$, with slopes `j0_x_coefficient`,
 * `j1_x_coefficient` and `j3_x_coefficient`.
 */
void radial_ansatz_coefficients(gsl::not_null<ComplexDataVector*> j0,
                                gsl::not_null<ComplexDataVector*> j1,
                                gsl::not_null<ComplexDataVector*> j3,
                                const ComplexDataVector& j2,
                                const ComplexDataVector& boundary_j,
                                const ComplexDataVector& boundary_dr_j,
                                const ComplexDataVector& boundary_dy_dy_j,
                                const ComplexDataVector& boundary_r);

/*!
 * \brief Determine \f$x = \tilde J^{(2)}\f$ of the Cauchy-gauge ansatz by
 * fixed-point iteration.
 *
 * \details Matching the worldtube value of \f$\tilde J\f$ and its first two
 * radial derivatives leaves \f$x\f$ to be determined by the remaining
 * partially flat condition \f$\breve J^{(2)} = 0\f$. Written in Cauchy-gauge
 * quantities, it is the nonlinear equation
 *
 * \f{align*}{
 *   x = \Phi(x) \equiv \frac{1}{2} \tilde J^{(0)}
 *     \left[\big|\tilde J^{(1)}\big|^2 - \big(\tilde K^{(1)}\big)^2\right],
 *   \quad
 *   \tilde K^{(1)} = \frac{\Re\big(\tilde J^{(0)} \bar{\tilde J}^{(1)}\big)}
 *     {\sqrt{1 + |\tilde J^{(0)}|^2}},
 * \f}
 *
 * where \f$\tilde J^{(0)}\f$ and \f$\tilde J^{(1)}\f$ are the affine functions
 * of \f$x\f$ fixed by the matching conditions, whose \f$x\f$-independent parts
 * `j0_at_zero` and `j1_at_zero` are supplied by the caller. With
 *
 * \f{align*}{
 *   \varrho = \max\left(\big\lVert \tilde J^{(0)}(0) \big\rVert_\infty,
 *     \big\lVert \tilde J^{(1)}(0) \big\rVert_\infty\right),
 * \f}
 *
 * \f$\Phi\f$ maps the disc \f$|x| \le \varrho\f$ into itself and is a
 * contraction there whenever \f$\varrho \le 7\times 10^{-2}\f$, so the
 * iteration from \f$x = 0\f$ converges to the unique root in the disc.
 * Worldtube data in the radiation zone satisfy this bound comfortably, and one
 * or two iterations suffice for them.
 *
 * Iteration stops once the largest change of \f$x\f$ over the angular
 * collocation points falls below `tolerance`, or after `max_iterations`
 * iterations. `max_iterations` of zero performs no iterations and leaves
 * \f$x = 0\f$.
 *
 * \returns the number of iterations performed; `final_step` is set to the
 * largest change of \f$x\f$ on the last of them (infinite if none were taken).
 * If \f$x\f$ becomes non-finite, because the inputs contain non-finite values
 * or the iteration diverges, the iteration stops and `final_step` is set to
 * NaN.
 */
size_t solve_asymptotic_j2(gsl::not_null<ComplexDataVector*> j2,
                           gsl::not_null<double*> final_step,
                           const ComplexDataVector& j0_at_zero,
                           const ComplexDataVector& j1_at_zero,
                           double tolerance, size_t max_iterations);

/*!
 * \brief The sentence of the `MaxPartiallyFlatJ0` error that says whether
 * raising `J0MaxIterations` can help.
 *
 * \details Empty if no linearized sweeps ran. Otherwise it says whether the
 * sweeps stopped because \f$\max|\breve J^{(0)}|\f$ had stopped improving
 * (`stopped_improving`), or because the angular solve reached
 * `j0_max_iterations` iterations, either while \f$\max|\breve J^{(0)}|\f$
 * was still
 * improving or before the `CauchySecondOrder::j0_plateau_sweeps` sweeps that
 * telling the two apart takes. If `j0_max_iterations` is already at the upper
 * bound of the `J0MaxIterations` option, it says so instead of suggesting a
 * larger value.
 */
std::string j0_max_iterations_hint(size_t number_of_linearized_sweeps,
                                   bool stopped_improving,
                                   size_t j0_max_iterations);
}  // namespace CauchySecondOrder_detail

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface by second-order Cauchy
 * matching at the worldtube.
 *
 * \details Here \f$\tilde J\f$ denotes \f$J\f$ in the Cauchy coordinates and
 * \f$\breve J\f$ its counterpart in the partially flat coordinates, and
 * \f$\tilde J^{(n)}\f$ and \f$\breve J^{(n)}\f$ are their coefficients of
 * \f$(1 - \tilde y)^n\f$, where \f$\tilde y = 1 - 2R/r\f$ is the compactified
 * radial coordinate and \f$R\f$ the worldtube radius. The initial data must
 * satisfy the partially flat conditions \f$\breve J^{(0)} = 0\f$ and
 * \f$\breve J^{(2)} = 0\f$. They are constructed in three steps:
 *
 * 1. Match the initial \f$\tilde J\f$ to the worldtube value and its first two
 *    radial derivatives in the Cauchy gauge, using the ansatz of
 *    `CauchySecondOrder_detail::radial_ansatz_coefficients`, and solve the
 *    fixed-point problem that the remaining condition \f$\breve J^{(2)} = 0\f$
 *    poses for \f$x = \tilde J^{(2)}\f$
 *    (`CauchySecondOrder_detail::solve_asymptotic_j2`). The worldtube
 *    \f$\partial_{\tilde y}^2 \tilde J|_\Gamma\f$ is obtained by inverting the
 *    \f$H\f$-hypersurface equation at the worldtube
 *    (`CauchySecondOrder_detail::compute_dy_dy_j`).
 * 2. Use the resulting \f$\tilde J^{(0)}\f$ to solve the Beltrami problem
 *    \f$\breve J^{(0)} = 0\f$ for the initial angular map, as described below.
 * 3. Transform the whole profile into the partially flat gauge with this map
 *    (`GaugeAdjustInitialJ`).
 *
 * In step 2, \f$\breve J^{(0)} = 0\f$ specifies the target angular Jacobian
 *
 * \f{align*}{
 *   \hat a_\star = -\frac{\bar{\hat b}_\star\, \tilde J^{(0)}}
 *     {1 + \tilde K^{(0)}}, \qquad
 *   \tilde K^{(0)} = \sqrt{1 + \big|\tilde J^{(0)}\big|^2}
 * \f}
 *
 * at every point of the sphere, where \f$\hat a\f$ and \f$\hat b\f$ are the
 * spin-weighted Jacobian factors of the angular map (\f$\hat c\f$ and
 * \f$\hat d\f$ in `GaugeUpdateJacobianFromCoordinates`). The map is recovered
 * iteratively, in up to two stages that together take at most
 * `J0MaxIterations` iterations. A potential-based solve
 * (`detail::adapt_angular_coordinates_via_potential`) generates each
 * displacement of the Cartesian coordinates \f$\hat x^i\f$ of the map from a
 * single spin-weight-1 potential \f$\eta\f$,
 *
 * \f{align*}{
 *   \delta \hat x^i = \mathrm{Re}\big(\eta\, \overline{\eth \hat x^i}\big),
 *   \qquad \eta = \eth^{-1}\big(\hat a_\star - \hat a\big),
 * \f}
 *
 * and runs to its minimum residual, which takes a few iterations. If that
 * brings \f$\max|\breve J^{(0)}|\f$ within `MaxPartiallyFlatJ0`, its map is the
 * solution. Otherwise linearized sweeps
 * (`detail::iteratively_adapt_angular_coordinates`) continue from it until the
 * smallest \f$\max|\breve J^{(0)}|\f$ reached has improved by less than 1% over
 * the last 50 sweeps, or until the two stages together have taken
 * `J0MaxIterations` iterations. Both stages keep the best map they evaluated.
 *
 * Initialization aborts if \f$\max|\tilde J^{(0)}(0)|\f$, or
 * \f$\max|\breve J^{(0)}|\f$ at any iteration of step 2, exceeds
 * `MaxCauchyJ0`, or if the fixed-point iteration of step 1 does not reach
 * `J2Tolerance` within `J2MaxIterations` iterations. After step 3 it prints a
 * summary of steps 1 and 2 and of \f$\max|\breve J^{(0)}|\f$ and
 * \f$\max|\breve J^{(2)}|\f$, and aborts if either exceeds its bound,
 * `MaxPartiallyFlatJ0` or `MaxPartiallyFlatJ2`.
 *
 * The worldtube \f$\partial_{\tilde u}\partial_r J|_\Gamma\f$ that enters the
 * \f$H\f$-hypersurface equation is obtained by differentiating an interpolant
 * of the worldtube \f$\partial_r J|_\Gamma\f$ in time with
 * `DuDrJInterpolator`, which this generator supplies to the worldtube data
 * manager. It is separate from the evolution's `H5Interpolator` because the
 * match wants a low interpolation order for that derivative while the
 * evolution wants a high one for the values it interpolates every step.
 */
struct CauchySecondOrder : InitializeJ<false> {
  /// The largest number of iterations of the potential solve. It reaches its
  /// minimum within a few, so this cap only bounds its cost on data it cannot
  /// solve.
  static constexpr size_t max_potential_iterations = 20;
  /// The linearized sweeps continue while the smallest
  /// \f$\max|\breve J^{(0)}|\f$ reached improves by at least the fraction
  /// `j0_plateau_improvement` over `j0_plateau_sweeps` sweeps.
  /// @{
  static constexpr size_t j0_plateau_sweeps = 50;
  static constexpr double j0_plateau_improvement = 1.0e-2;
  /// @}

  struct MaxPartiallyFlatJ0 {
    using type = double;
    static constexpr Options::String help = {
        "Largest allowed max|J^(0)| = max|J| at scri+ in the partially flat "
        "gauge. The potential solve that runs first is enough if it reaches "
        "this value; otherwise linearized sweeps follow until max|J^(0)| "
        "stops improving, and initialization aborts if it is still above this "
        "value."};
    static type lower_bound() { return 1.0e-16; }
    static type upper_bound() { return 1.0; }
    static type suggested_value() { return 5.0e-12; }
  };

  struct J0MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Largest number of angular coordinate iterations, shared by the "
        "potential solve and the linearized sweeps that follow it if it does "
        "not reach MaxPartiallyFlatJ0."};
    static type lower_bound() { return 10; }
    static type upper_bound() { return 2000; }
    static type suggested_value() { return 1500; }
  };

  struct J2Tolerance {
    using type = double;
    static constexpr Options::String help = {
        "Convergence tolerance of the fixed-point iteration that determines "
        "the second radial expansion coefficient J^(2) of the Cauchy-gauge "
        "ansatz from the partially flat gauge condition. The iteration stops "
        "once the largest change of J^(2) over the collocation points falls "
        "below this value; the comparison is absolute, as J is dimensionless "
        "and of the size of the strain."};
    static type lower_bound() { return 1.0e-16; }
    static type upper_bound() { return 1.0e-3; }
    static type suggested_value() { return 1.0e-14; }
  };

  struct J2MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Largest number of fixed-point iterations used to determine the "
        "Cauchy-gauge J^(2). For worldtube data in the radiation zone one or "
        "two iterations suffice. "
        "If this is nonzero and the iteration has not converged after this "
        "many iterations, initialization aborts. "
        "Zero skips the solve and leaves J^(2) = 0, which does not satisfy "
        "the partially flat gauge condition."};
    static type upper_bound() { return 100; }
    static type suggested_value() { return 10; }
  };

  struct MaxCauchyJ0 {
    using type = double;
    static constexpr Options::String help = {
        "Largest allowed max|J^(0)| in the Cauchy gauge, where J^(0) is J at "
        "scri+. The same bound also limits the partially flat J^(0) during "
        "the angular iterations as a divergence guard. Exceeding either "
        "bound aborts initialization. MaxPartiallyFlatJ0 bounds the partially "
        "flat J^(0) after the angular solve. Larger bounds allow larger "
        "asymptotic strains but risk a poorly behaved angular coordinate map."};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e2; }
    static type suggested_value() { return 5.0e-2; }
  };

  struct MaxPartiallyFlatJ2 {
    using type = double;
    static constexpr Options::String help = {
        "Largest allowed max|J^(2)| in the partially flat gauge after the "
        "gauge transformation, where J^(2) = (1/2) Dy^2 J at scri+ is the "
        "coefficient of (1 - y)^2. The fixed-point solve for the Cauchy-gauge "
        "J^(2) makes it vanish up to discretization error, so initialization "
        "aborts above "
        "this value, which usually means the angular or radial resolution is "
        "too low for the worldtube data."};
    static type lower_bound() { return 1.0e-16; }
    static type upper_bound() { return 1.0; }
    static type suggested_value() { return 1.0e-12; }
  };

  struct DuDrJInterpolator {
    using type = std::unique_ptr<intrp::SpanInterpolator>;
    static constexpr Options::String help = {
        "Interpolator used to time-differentiate the worldtube Dr(J) into the "
        "Du(Dr(J)) that this generator matches to. That boundary value feeds "
        "the initial data alone, so this is independent of `H5Interpolator`: "
        "the evolution wants a high interpolation order, while the "
        "second-order match wants a low one (a barycentric order of 2 to 4), "
        "which keeps the high-frequency content of the worldtube out of the "
        "initial data."};
  };

  using options =
      tmpl::list<MaxPartiallyFlatJ0, J0MaxIterations, MaxCauchyJ0, J2Tolerance,
                 J2MaxIterations, MaxPartiallyFlatJ2, DuDrJInterpolator>;
  static constexpr Options::String help = {
      "Second-order initial data generator for the Cauchy CCE evolution."};

  WRAPPED_PUPable_decl_template(CauchySecondOrder);  // NOLINT
  explicit CauchySecondOrder(CkMigrateMessage* /*unused*/) {}

  CauchySecondOrder(
      double max_partially_flat_j0, size_t j0_max_iterations,
      double max_cauchy_j0, double j2_tolerance, size_t j2_max_iterations,
      double max_partially_flat_j2,
      std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator);

  CauchySecondOrder() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator()
      const override;

  // Per-class tag lists. The flexible dispatch in `InitializeJ<false>` reads
  // these via `call_with_dynamic_type` so this generator can request more
  // worldtube boundary values than the simpler sibling classes do.
  using return_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiJ>, Tags::BoundaryValue<Tags::BondiU>,
      Tags::BoundaryValue<Tags::BondiW>, Tags::BoundaryValue<Tags::BondiBeta>,
      Tags::BoundaryValue<Tags::BondiQ>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiR>>,
      Tags::BoundaryValue<Tags::BondiR>, Tags::LMax,
      Tags::NumberOfRadialPoints>;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
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
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  double max_partially_flat_j0_ = std::numeric_limits<double>::signaling_NaN();
  size_t j0_max_iterations_ = 0;
  double max_cauchy_j0_ = std::numeric_limits<double>::signaling_NaN();
  double j2_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  size_t j2_max_iterations_ = 0;
  double max_partially_flat_j2_ = std::numeric_limits<double>::signaling_NaN();
  std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator_;
};
}  // namespace Cce::InitializeJ
