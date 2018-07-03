// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
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
class Linear {
 public:
  struct ShockTime {
    using type = double;
    static constexpr OptionString help{"The time at which a shock forms"};
  };

  using options = tmpl::list<ShockTime>;
  static constexpr OptionString help{"A spatially linear solution"};

  Linear() = default;
  Linear(const Linear&) noexcept = delete;
  Linear& operator=(const Linear&) noexcept = delete;
  Linear(Linear&&) noexcept = default;
  Linear& operator=(Linear&&) noexcept = default;
  ~Linear() noexcept = default;

  explicit Linear(double shock_time) noexcept;

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 1>& x, double t) const noexcept;

  template <typename T>
  Scalar<T> du_dt(const tnsr::I<T, 1>& x, double t) const noexcept;

  tuples::TaggedTuple<Tags::U> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<Tags::U> /*meta*/) const noexcept;

  tuples::TaggedTuple<::Tags::dt<Tags::U>> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  double shock_time_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Solutions
}  // namespace Burgers
