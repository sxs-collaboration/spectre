// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
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

namespace ScalarAdvection {
namespace AnalyticData {
/*!
 * \brief Sinusoidal analytic data for the ScalarAdvection system, periodic over
 * the interval \f$[-1, 1]\f$.
 *
 * The initial data is given by:
 *
 * \f{align}{
 *   u(x, 0) = \sin \pi x
 * \f}
 *
 * At future times the analytic solution can be found by simply advecting the
 * initial data (sine wave) with the velocity of unity:
 *
 * \f{align}{
 *   u(x,t)=\sin \pi(x-t)
 * \f}
 *
 */
class Sinusoid : public MarkAsAnalyticData {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A sinusoidal initial data u(x,0) = sin(pi x) for 1D scalar advection, "
      "periodic over the interval [-1, 1]"};

  Sinusoid() = default;
  Sinusoid(const Sinusoid&) noexcept = default;
  Sinusoid& operator=(const Sinusoid&) noexcept = default;
  Sinusoid(Sinusoid&&) noexcept = default;
  Sinusoid& operator=(Sinusoid&&) noexcept = default;
  ~Sinusoid() noexcept = default;

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 1>& x) const noexcept;

  tuples::TaggedTuple<Tags::U> variables(
      const tnsr::I<DataVector, 1>& x,
      tmpl::list<Tags::U> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/) noexcept;

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) noexcept;
}  // namespace AnalyticData
}  // namespace ScalarAdvection
