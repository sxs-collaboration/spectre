// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarAdvection {
namespace Solutions {
/*!
 * \brief Initial data for the 2D scalar advection problem adopted from
 * \cite Kuzmin2014 and its analytic solution.
 *
 * Let \f$r(x,y) = \sqrt{(x-x_0)^2 + (y-y_0)^2}/r_0\f$ be the normalized
 * distance from a point \f$(x_0,y_0)\f$ within a circle with the radius
 * \f$r_0=0.15\f$. The initial proﬁle consists of three bodies:
 *
 * - a slotted cylinder centered at \f$(0.5, 0.75)\f$
 * \f{align*}
 * u(x,y) = \left\{\begin{array}{ll}
 *  1 & \text{if } |x-x_0| \geq 0.025 \text{ or } y \geq 0.85 \\
 *  0 & \text{otherwise} \\
 * \end{array}\right\},
 * \f}
 *
 * - a cone centered at \f$(0.5, 0.25)\f$
 * \f{align*}
 * u(x,y) = 1 - r(x,y) ,
 * \f}
 *
 * - and a hump centered at \f$(0.25, 0.5)\f$
 * \f{align*}
 * u(x,y) = \frac{1 + \cos(\pi r(x,y))}{4} .
 * \f}
 *
 * The system is evolved over the domain \f$[0,1]\times[0,1]\f$ with the
 * advection velocity field \f$v(x,y) = (0.5-y,-0.5+x)\f$, which causes a solid
 * rotation about \f$(x,y)=(0.5,0.5)\f$ with angular velocity being 1.0. We use
 * the periodic boundary condition for evolving this problem.
 */
class Kuzmin : public MarkAsAnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A rotating 2D scalar advecting problem adopted from Kuzmin2014 paper"};

  Kuzmin() = default;
  Kuzmin(const Kuzmin&) noexcept = default;
  Kuzmin& operator=(const Kuzmin&) noexcept = default;
  Kuzmin(Kuzmin&&) noexcept = default;
  Kuzmin& operator=(Kuzmin&&) noexcept = default;
  ~Kuzmin() noexcept = default;

  template <typename DataType>
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataType, 2>& x, double t,
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const Kuzmin& /*lhs*/, const Kuzmin& /*rhs*/) noexcept;

bool operator!=(const Kuzmin& lhs, const Kuzmin& rhs) noexcept;

}  // namespace Solutions
}  // namespace ScalarAdvection
