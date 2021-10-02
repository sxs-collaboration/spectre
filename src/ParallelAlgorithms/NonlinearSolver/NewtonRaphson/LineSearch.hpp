// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace NonlinearSolver::newton_raphson {
/*!
 * \brief Find the next step length for the line-search globalization
 *
 * The next step length is chosen such that it minimizes the quadratic (first
 * globalization step, i.e., when `globalization_iteration_id` is 0) or cubic
 * (subsequent globalization steps) polynomial interpolation. This function
 * implements Algorithm A6.1.3 in \cite DennisSchnabel (p. 325). This is how
 * argument names map to symbols in that algorithm:
 *
 * - `step_length`: \f$\lambda\f$
 * - `prev_step_length`: \f$\lambda_\mathrm{prev}\f$
 * - `residual`: \f$f_c\f$
 * - `residual_slope`: \f$g^T p < 0\f$
 * - `next_residual`: \f$f_+\f$
 * - `prev_residual`: \f$f_{+,\mathrm{prev}}\f$
 *
 * Note that the argument `residual_slope` is the derivative of the residual
 * function \f$f\f$ w.r.t. the step length, i.e.
 * \f$\frac{\mathrm{d}f}{\mathrm{d}\lambda}\f$, which must be negative. For the
 * common scenario where \f$f(x)=|\boldsymbol{r}(x)|^2\f$, i.e. the residual
 * function is the L2 norm of a residual vector \f$\boldsymbol{r}(x)\f$, and
 * where that in turn is the residual of a nonlinear equation
 * \f$\boldsymbol{r}(x)=b-A_\mathrm{nonlinear}(x)\f$ in a Newton-Raphson step as
 * described in `NonlinearSolver::newton_raphson::NewtonRaphson`, then the
 * `residual_slope` reduces to
 *
 * \f{equation}
 * \frac{\mathrm{d}f}{\mathrm{d}\lambda} =
 * \frac{\mathrm{d}f}{\mathrm{d}x^i} \frac{\mathrm{d}x^i}{\mathrm{d}\lambda} =
 * 2 \boldsymbol{r}(x) \cdot \frac{\mathrm{d}\boldsymbol{r}}{\mathrm{d}x^i}
 * \frac{\mathrm{d}x^i}{\mathrm{d}\lambda} =
 * -2 |\boldsymbol{r}(x)|^2 = -2 f(x) \equiv -2 f_c
 * \text{.}
 * \f}
 *
 * Here we have used the relation
 *
 * \f{equation}
 * \frac{\mathrm{d}\boldsymbol{r}}{\mathrm{d}x^i}
 * \frac{\mathrm{d}x^i}{\mathrm{d}\lambda} =
 * -\frac{\delta A_\mathrm{nonlinear}}{\delta x}\cdot\delta x = -r
 * \f}
 *
 * of a Newton-Raphson step of full length \f$\delta x\f$.
 */
double next_step_length(size_t globalization_iteration_id, double step_length,
                        double prev_step_length, double residual,
                        double residual_slope, double next_residual,
                        double prev_residual);
}  // namespace NonlinearSolver::newton_raphson
