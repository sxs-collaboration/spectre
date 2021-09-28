// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
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

namespace Burgers {
namespace Solutions {
/// A solution that is linear in space at all times.
///
/// \f$u(x, t) = x / (t - t_0)\f$ where \f$t_0\f$ is the shock time.
class Linear : public MarkAsAnalyticSolution {
 public:
  struct ShockTime {
    using type = double;
    static constexpr Options::String help{"The time at which a shock forms"};
  };

  using options = tmpl::list<ShockTime>;
  static constexpr Options::String help{"A spatially linear solution"};

  Linear() = default;
  Linear(const Linear&) = delete;
  Linear& operator=(const Linear&) = delete;
  Linear(Linear&&) = default;
  Linear& operator=(Linear&&) = default;
  ~Linear() = default;

  explicit Linear(double shock_time);

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 1>& x, double t) const;

  template <typename T>
  Scalar<T> du_dt(const tnsr::I<T, 1>& x, double t) const;

  tuples::TaggedTuple<Tags::U> variables(const tnsr::I<DataVector, 1>& x,
                                         double t,
                                         tmpl::list<Tags::U> /*meta*/) const;

  tuples::TaggedTuple<::Tags::dt<Tags::U>> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p);  // NOLINT

 private:
  double shock_time_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Solutions
}  // namespace Burgers
