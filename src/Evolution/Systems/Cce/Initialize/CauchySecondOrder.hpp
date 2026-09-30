// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>

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

/// \brief The dependence of the coefficients of the Cauchy-gauge radial ansatz
/// on the free parameter \f$x = \tilde J^{(2)}\f$ at matching order \f$N=2\f$.
///
/// \details The matching conditions at the worldtube are linear, so each
/// coefficient is an affine function of \f$x\f$,
/// \f$\tilde J^{(n)}(x) = \tilde J^{(n)}(0) + \alpha^{(n)} x\f$, with
/// \f$\alpha^{(n)} = (-1)^n 2^{3-n} (N-1)! / (n! (N+1-n)!)\f$.
/// @{
constexpr double j0_x_coefficient = 4.0 / 3.0;
constexpr double j1_x_coefficient = -2.0;
constexpr double j3_x_coefficient = -1.0 / 6.0;
/// @}

/*!
 * \brief Compute the coefficients \f$J^{(0)}\f$, \f$\tilde J^{(1)}\f$ and
 * \f$\tilde J^{(3)}\f$ of the Cauchy-gauge radial ansatz
 * \f$J = J^{(0)} + \tilde J^{(1)} (1 - y) + x (1 - y)^2
 * + \tilde J^{(3)} (1 - y)^3\f$ for a given \f$x = \tilde J^{(2)}\f$.
 *
 * \details The coefficients are fixed by matching the worldtube (\f$y = -1\f$)
 * values of \f$J\f$, \f$\partial_r J\f$ and \f$\partial_y^2 J\f$, using
 * \f$\partial_y J = (R / 2) \partial_r J\f$ at the worldtube radius \f$R\f$.
 * They are affine in \f$x\f$, with slopes `j0_x_coefficient`,
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
 * \brief Determine the second radial expansion coefficient
 * \f$x = \tilde J^{(2)}\f$ of the Cauchy-gauge ansatz by fixed-point iteration.
 *
 * \details Matching the worldtube values of \f$J\f$, \f$\partial_r J\f$ and
 * \f$\partial_y^2 J\f$ leaves one condition on the Cauchy-gauge radial ansatz
 * unused: the partially flat gauge condition \f$\breve J^{(2)} = 0\f$ on the
 * second asymptotic coefficient of the transformed \f$J\f$. Written in terms of
 * Cauchy-gauge quantities alone it is the nonlinear constraint
 *
 * \f{align*}{
 *   x = \Phi(x) \equiv \frac{1}{2} J^{(0)}(x)
 *     \left[\big|\tilde J^{(1)}(x)\big|^2
 *       - \big(\tilde K^{(1)}(x)\big)^2\right], \quad
 *   \tilde K^{(1)} = \frac{\Re\big(J^{(0)} \bar{\tilde J}^{(1)}\big)}
 *     {\sqrt{1 + |J^{(0)}|^2}},
 * \f}
 *
 * where \f$J^{(0)}(x)\f$ and \f$\tilde J^{(1)}(x)\f$ are the affine functions
 * of \f$x\f$ fixed by the matching conditions, whose \f$x\f$-independent parts
 * `j0_at_zero` and `j1_at_zero` are supplied by the caller. \f$\Phi\f$ maps the
 * disc of radius \f$\varrho = \max(\|J^{(0)}(0)\|_\infty,
 * \|\tilde J^{(1)}(0)\|_\infty)\f$ into itself and is a contraction there
 * whenever \f$\varrho \lesssim 7\times 10^{-2}\f$, so the iteration from
 * \f$x = 0\f$ converges to the unique root. The Lipschitz constant of
 * \f$\Phi\f$ is of order \f$\varrho^2\f$, so each pass gains roughly
 * \f$2\log_{10}(1/\varrho)\f$ digits: reaching \f$10^{-14}\f$ takes about two
 * passes for \f$\varrho \sim 10^{-3}\f$ and about six for
 * \f$\varrho \sim 5\times 10^{-2}\f$.
 *
 * Iteration stops once the largest change of \f$x\f$ over the collocation
 * points falls below `tolerance`, or after `max_iterations` passes.
 * `max_iterations` of zero performs no passes and leaves \f$x = 0\f$.
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
}  // namespace CauchySecondOrder_detail

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface using a second-order
 * matching at the worldtube.
 *
 * \details The volume \f$J\f$ is built from the worldtube values of
 * \f$J\f$, \f$\partial_r J\f$, and \f$\partial_y^2 J\f$ computed from the
 * H hypersurface equation. Matching those three worldtube values supplies only
 * three of the four conditions on the Cauchy-gauge radial ansatz; the fourth is
 * the remaining partially flat gauge condition \f$\breve J^{(2)} = 0\f$, which
 * becomes a nonlinear constraint
 *
 * \f{align*}{
 *   \tilde J^{(2)} = \frac{1}{2} J^{(0)} \left[ \big|\tilde J^{(1)}\big|^2
 *     - \big(\tilde K^{(1)}\big)^2 \right], \quad
 *   \tilde K^{(1)} = \frac{\Re\big(J^{(0)} \bar{\tilde J}^{(1)}\big)}
 *     {\sqrt{1 + |J^{(0)}|^2}}
 * \f}
 *
 * on the expansion coefficients. Since the matching conditions make every
 * coefficient an affine function of \f$x \equiv \tilde J^{(2)}\f$, that
 * constraint is a fixed-point problem \f$x = \Phi(x)\f$, which is solved by
 * iterating from \f$x = 0\f$ to `J2Tolerance` within at most `J2MaxIterations`
 * passes (the map is a contraction whenever the worldtube data are small enough
 * that \f$\max(\|J^{(0)}\|_\infty, \|\tilde J^{(1)}\|_\infty)\f$ is at
 * most a few times \f$10^{-2}\f$; see
 * `CauchySecondOrder_detail::solve_asymptotic_j2` for the number of passes).
 *
 * The remaining angular coordinates are determined
 * iteratively to ensure asymptotic flatness. The angular solve can eliminate
 * \f$J\f$ at scri+ only through a well-behaved alteration of the spherical
 * mesh, so it tolerates only a small asymptotic \f$J\f$; the initialization
 * aborts if the asymptotic \f$J\f$ in Cauchy coordinates, or the deviation at
 * any iteration of the solve, exceeds `MaxCauchyJ0`. Both solves must
 * converge within their iteration budgets. After the gauge transformation,
 * initialization prints the maximum absolute values of the partially flat
 * constraints \f$J_0 = J|_{\mathcal{I}^+}\f$ and
 * \f$J_2 = \frac{1}{2}\partial_y^2 J|_{\mathcal{I}^+}\f$, and aborts if
 * \f$\max|J_2|\f$ exceeds `MaxPartiallyFlatJ2`.
 *
 * The worldtube \f$\partial_u \partial_r J\f$ that enters the H hypersurface
 * equation is obtained by differentiating the worldtube \f$\partial_r J\f$ in
 * time with `DuDrJInterpolator`, which this generator supplies to the worldtube
 * data manager. It is separate from the evolution's `H5Interpolator` because
 * the match wants a low interpolation order for that derivative while the
 * evolution wants a high one for the values it interpolates every step.
 */
struct CauchySecondOrder : InitializeJ<false> {
  struct J0Tolerance {
    using type = double;
    static constexpr Options::String help = {
        "Tolerance on the maximum absolute value of J at scri+ in the "
        "partially flat gauge, used by the initial angular coordinate solve."};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
    static type suggested_value() { return 5.0e-12; }
  };

  struct J0MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Maximum number of angular coordinate iterations to reach J0Tolerance. "
        "Initialization aborts if the solve does not converge."};
    static type lower_bound() { return 10; }
    static type upper_bound() { return 2000; }
    static type suggested_value() { return 300; }
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
        "Largest number of fixed-point passes used to determine J^(2). Each "
        "pass gains roughly 2 log10(1/rho) digits, where rho is the larger of "
        "max|J^(0)| and max|J^(1)|, the leading coefficients of the "
        "Cauchy-gauge ansatz. Reaching 1e-14 takes about two passes for "
        "rho ~ 1e-3 and about six for rho ~ 5e-2. "
        "Failing to converge within a nonzero budget aborts initialization. "
        "Zero skips the solve and leaves J^(2) = 0, which does not satisfy "
        "the partially flat gauge condition."};
    static type upper_bound() { return 100; }
    static type suggested_value() { return 10; }
  };

  struct MaxCauchyJ0 {
    using type = double;
    static constexpr Options::String help = {
        "Largest allowed max|J0| in the Cauchy-gauge initial guess, where J0 "
        "is J at scri+. The same bound also limits the transformed J0 during "
        "the angular iterations as a divergence guard. Exceeding either "
        "bound aborts initialization. J0Tolerance sets the convergence target "
        "in the partially flat gauge. Larger bounds allow larger asymptotic "
        "strains but risk a poorly behaved angular coordinate map."};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e2; }
    static type suggested_value() { return 5.0e-2; }
  };

  struct MaxPartiallyFlatJ2 {
    using type = double;
    static constexpr Options::String help = {
        "Largest allowed max|J2| = max|(1/2) Dy^2 J| at scri+ after the gauge "
        "transformation to the partially flat gauge. The J^(2) solve makes J2 "
        "vanish up to discretization error, so initialization aborts above "
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
      tmpl::list<J0Tolerance, J0MaxIterations, MaxCauchyJ0, J2Tolerance,
                 J2MaxIterations, MaxPartiallyFlatJ2, DuDrJInterpolator>;
  static constexpr Options::String help = {
      "Second-order initial data generator for the Cauchy CCE evolution."};

  WRAPPED_PUPable_decl_template(CauchySecondOrder);  // NOLINT
  explicit CauchySecondOrder(CkMigrateMessage* /*unused*/) {}

  CauchySecondOrder(
      double j0_tolerance, size_t j0_max_iterations, double max_cauchy_j0,
      double j2_tolerance, size_t j2_max_iterations,
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
  double j0_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  size_t j0_max_iterations_ = 0;
  double max_cauchy_j0_ = std::numeric_limits<double>::signaling_NaN();
  double j2_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  size_t j2_max_iterations_ = 0;
  double max_partially_flat_j2_ = std::numeric_limits<double>::signaling_NaN();
  std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator_;
};
}  // namespace Cce::InitializeJ
