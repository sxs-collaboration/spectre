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
/// \brief A propagating shock between two constant states
///
/// The shock propagates left-to-right, with profile
/// \f$U(x, t) = U_R + (U_L - U_R) H(-(x - x_0 - v t))\f$.
/// Here \f$U_L\f$ and \f$U_R\f$ are the left and right constant states,
/// satisfying \f$U_L > U_R\f$; \f$H\f$ is the Heaviside function; \f$x_0\f$ is
/// the initial (i.e., \f$t=0\f$) position of the shock; and
/// \f$v = 0.5 (U_L + U_R)\f$ is the speed of the shock.
///
/// \note At the shock, where \f$x = x_0 + vt\f$, we have \f$U(x, t) = U_L\f$.
/// (This is inherited from the Heaviside implementation `step_function`.)
/// Additionally, the time derivative \f$\partial_t u0\f$ is zero, rather than
/// the correct delta function.
class Step {
 public:
  struct LeftValue {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static type default_value() noexcept { return 2.0; }
    static constexpr OptionString help{"The value of U, left of the shock"};
  };
  struct RightValue {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static type default_value() noexcept { return 1.0; }
    static constexpr OptionString help{"The value of U, right of the shock"};
  };
  struct InitialPosition {
    using type = double;
    static constexpr OptionString help{"The shock's position at t==0"};
  };

  using options = tmpl::list<LeftValue, RightValue, InitialPosition>;
  static constexpr OptionString help{"A propagating shock solution"};

  Step(double left_value, double right_value, double initial_shock_position,
       const OptionContext& context = {});

  Step() = default;
  Step(const Step&) noexcept = default;
  Step& operator=(const Step&) noexcept = default;
  Step(Step&&) noexcept = default;
  Step& operator=(Step&&) noexcept = default;
  ~Step() noexcept = default;

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 1>& x, double t) const noexcept;

  template <typename T>
  Scalar<T> du_dt(const tnsr::I<T, 1>& x, double t) const noexcept;

  tuples::TaggedTuple<Tags::U> variables(const tnsr::I<DataVector, 1>& x,
                                         double t,
                                         tmpl::list<Tags::U> /*meta*/) const
      noexcept;

  tuples::TaggedTuple<::Tags::dt<Tags::U>> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  double left_value_ = std::numeric_limits<double>::signaling_NaN();
  double right_value_ = std::numeric_limits<double>::signaling_NaN();
  double initial_shock_position_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Solutions
}  // namespace Burgers
