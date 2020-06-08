// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Burgers {
namespace AnalyticData {
/*!
 * \brief Analytic data (with an "exact" solution known) that is periodic over
 * the interval \f$[0,2\pi]\f$.
 *
 * The initial data is given by:
 *
 * \f{align}{
 *   u(x, 0) = \sin(x)
 * \f}
 *
 * At future times the analytic solution can be found by solving the
 * transcendental equation \cite Harten19973
 *
 * \f{align}{
 *   \label{eq:transcendental burgers periodic}
 *   \mathcal{F}=\sin\left(x-\mathcal{F}t\right)
 * \f}
 *
 * on the interval \f$x\in(0,\pi)\f$. The solution from \f$x\in(\pi,2\pi)\f$ is
 * given by \f$\mathcal{F}(x, t)=-\mathcal{F}(2\pi-x,t)\f$. The transcendental
 * equation \f$(\ref{eq:transcendental burgers periodic})\f$ can be solved with
 * a Newton-Raphson iterative scheme. Since this can be quite sensitive to the
 * initial guess we implement this solution as analytic data. The python code
 * below can be used to compute the analytic solution if desired.
 *
 * At time \f$1\f$ the solution develops a discontinuity at \f$x=\pi\f$ followed
 * by the amplitude of the solution decaying over time.
 *
 * \note We have rescaled \f$x\f$ and \f$t\f$ by \f$\pi\f$ compared to
 * \cite Harten19973.
 *
 * \code{py}
   import numpy as np
   from scipy.optimize import newton

   # x_grid is a np.array of positions at which to evaluate the solution
   def burgers_periodic(x_grid, time):
       def opt_fun(F, x, t):
           return np.sin((x - F * t)) - F

       results = []
       for i in range(len(x_grid)):
           x = x_grid[i]
           greater_than_pi = False
           if x > np.pi:
               x = x - np.pi
               x = -x
               x = x + np.pi
               greater_than_pi = True

           guess = 0.0
           if len(results) > 0:
               if results[-1] < 0.0:
                   guess = -results[-1]
               else:
                   guess = results[-1]
           res = newton(lambda F: opt_fun(F, x, time), x0=guess)

           if greater_than_pi:
               results.append(-res)
           else:
               results.append(res)

       return np.asarray(results)
 * \endcode
 */
class Sinusoid : public MarkAsAnalyticData {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help{
      "A solution that is periodic over the interval [0,2pi]. The solution "
      "starts as a sinusoid: u(x,0) = sin(x) and develops a "
      "discontinuity at x=pi and t=1."};

  Sinusoid() = default;
  Sinusoid(const Sinusoid&) noexcept = default;
  Sinusoid& operator=(const Sinusoid&) noexcept = default;
  Sinusoid(Sinusoid&&) noexcept = default;
  Sinusoid& operator=(Sinusoid&&) noexcept = default;
  ~Sinusoid() noexcept = default;

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 1>& x) const noexcept;

  tuples::TaggedTuple<Tags::U> variables(const tnsr::I<DataVector, 1>& x,
                                         tmpl::list<Tags::U> /*meta*/) const
      noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/) noexcept;

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) noexcept;
}  // namespace AnalyticData
}  // namespace Burgers
