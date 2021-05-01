// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarAdvection {
namespace AnalyticData {
/*!
 * \brief Initial data for the 2D scalar advection problem adopted from
 * \cite Kuzmin2014.
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
 */
class Kuzmin : public MarkAsAnalyticData {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "An initial data for 2D scalar advection adopted from Kuzmin2014"};

  Kuzmin() = default;
  Kuzmin(const Kuzmin&) noexcept = default;
  Kuzmin& operator=(const Kuzmin&) noexcept = default;
  Kuzmin(Kuzmin&&) noexcept = default;
  Kuzmin& operator=(Kuzmin&&) noexcept = default;
  ~Kuzmin() noexcept = default;

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 2>& x) const noexcept;

  tuples::TaggedTuple<Tags::U> variables(
      const tnsr::I<DataVector, 2>& x,
      tmpl::list<Tags::U> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const Kuzmin& /*lhs*/, const Kuzmin& /*rhs*/) noexcept;

bool operator!=(const Kuzmin& lhs, const Kuzmin& rhs) noexcept;
}  // namespace AnalyticData
}  // namespace ScalarAdvection
