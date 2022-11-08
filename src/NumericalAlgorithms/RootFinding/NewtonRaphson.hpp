// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#error "RootFinding/NewtonRaphson.hpp does not provide any functionality."

namespace RootFinder {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * For the nonlinear solver, see NonlinearSolver::newton_raphson.  We
 * do not provide a Newton-Raphson root finder.  The Boost
 * implementation is buggy and can return the wrong answer, and it is
 * not clear that performance would be better than RootFinder::toms748
 * in practice.
 *
 * Newton-Raphson is asymptotically faster than TOMS748 in the ideal
 * case, but that assumes that function evaluations take the same
 * amount of time for both solvers, while in reality the derivatives
 * are often as or even more expensive than the values.  Additionally,
 * for realistic problems, convergence is usually fast enough that
 * neither solver reaches the asymptotic regime.  Newton-Raphson also
 * has the advantage of requiring less work internally in the solver
 * implementation, but, again, for realistic problems solver overhead
 * is usually dwarfed by the function evaluations.
 */
void newton_raphson() = delete;
}  // namespace RootFinder
