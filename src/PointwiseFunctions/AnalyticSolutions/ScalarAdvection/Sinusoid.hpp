// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
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
class Sinusoid : public InitialDataUtilities::InitialData,
                 public MarkAsAnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "An advecting 1D sine wave u(x,t) = sin(pi(x-t)), periodic over the "
      "interval [-1, 1]"};

  Sinusoid() = default;
  Sinusoid(const Sinusoid&) = default;
  Sinusoid& operator=(const Sinusoid&) = default;
  Sinusoid(Sinusoid&&) = default;
  Sinusoid& operator=(Sinusoid&&) = default;
  ~Sinusoid() = default;

  template <typename DataType>
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataType, 1>& x, double t,
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p);  // NOLINT

  /// \cond
  explicit Sinusoid(CkMigrateMessage* msg) : InitialData(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Sinusoid);
  /// \endcond
};

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/);

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs);

}  // namespace Solutions
}  // namespace ScalarAdvection
