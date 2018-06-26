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
/*!
 * \brief A solution resembling a bump.
 *
 * At \f$t=0\f$, the solution is a parabola:
 * \f{equation*}
 *  u(x, t) = h \left(1 - \left(\frac{x - c}{w}\right)^2\right),
 * \f}
 * where \f$h\f$ is the height, \f$c\f$ is the center, and \f$w\f$ is
 * the distance from the center to the zeros.  A shock propagates in
 * from infinity and reaches one of the zeros at \f$t = \frac{w}{2
 * h}\f$.
 */
class Bump {
 public:
  struct HalfWidth {
    using type = double;
    static constexpr OptionString help{
        "The distance from the center to the zero of the bump"};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Height {
    using type = double;
    static constexpr OptionString help{"The height of the bump"};
  };

  struct Center {
    using type = double;
    static constexpr OptionString help{"The center of the bump"};
    static type default_value() noexcept { return 0.; }
  };

  using options = tmpl::list<HalfWidth, Height, Center>;
  static constexpr OptionString help{"A bump solution"};

  Bump() = default;
  Bump(const Bump&) noexcept = delete;
  Bump& operator=(const Bump&) noexcept = delete;
  Bump(Bump&&) noexcept = default;
  Bump& operator=(Bump&&) noexcept = default;
  ~Bump() noexcept = default;

  Bump(double half_width, double height, double center = 0.) noexcept;

  template <typename T>
  Scalar<T> u(const tnsr::I<T, 1>& x, double t) const noexcept;

  template <typename T>
  Scalar<T> du_dt(const tnsr::I<T, 1>& x, double t) const noexcept;

  tuples::TaggedTuple<Tags::U> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<Tags::U> /*meta*/) const noexcept;

  tuples::TaggedTuple<::Tags::dt<Burgers::Tags::U>> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  double half_width_ = std::numeric_limits<double>::signaling_NaN();
  double height_ = std::numeric_limits<double>::signaling_NaN();
  double center_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Solutions
}  // namespace Burgers
