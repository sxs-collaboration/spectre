// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
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
 * \brief An 1D sinusoidal wave advecting with speed 1.0, periodic over the
 * interval \f$[-1, 1]\f$.
 *
 * \f{align}{
 *   u(x,t)=\sin \pi(x-t)
 * \f}
 *
 */
class Sinusoid : public MarkAsAnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "An advecting 1D sine wave u(x,t) = sin(pi(x-t)), periodic over the "
      "interval [-1, 1]"};

  Sinusoid() = default;
  Sinusoid(const Sinusoid&) noexcept = default;
  Sinusoid& operator=(const Sinusoid&) noexcept = default;
  Sinusoid(Sinusoid&&) noexcept = default;
  Sinusoid& operator=(Sinusoid&&) noexcept = default;
  ~Sinusoid() noexcept = default;

  template <typename DataType>
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataType, 1>& x, double t,
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/) noexcept;

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) noexcept;

}  // namespace Solutions
}  // namespace ScalarAdvection
