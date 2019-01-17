// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/Tags.hpp"    // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson {
namespace Solutions {

/*!
 * \brief A Lorentzian solution to the Poisson equation
 *
 * \details This implements the Lorentzian solution
 * \f$u(\boldsymbol{x})=\left(1+r^2\right)^{-\frac{1}{2}}\f$ to the
 * three-dimensional Poisson equation
 * \f$-\Delta u(\boldsymbol{x})=f(\boldsymbol{x})\f$, where
 * \f$r^2=x^2+y^2+z^2\f$. The corresponding source is
 * \f$f(\boldsymbol{x})=3\left(1+r^2\right)^{-\frac{5}{2}}\f$.
 *
 * \note Corresponding 1D and 2D solutions are not implemented yet.
 */
template <size_t Dim>
class Lorentzian {
  static_assert(
      Dim == 3,
      "This solution is currently implemented in 3 spatial dimensions only");

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help{
      "A Lorentzian solution to the Poisson equation."};

  Lorentzian() = default;
  Lorentzian(const Lorentzian&) noexcept = delete;
  Lorentzian& operator=(const Lorentzian&) noexcept = delete;
  Lorentzian(Lorentzian&&) noexcept = default;
  Lorentzian& operator=(Lorentzian&&) noexcept = default;
  ~Lorentzian() noexcept = default;

  // @{
  /// Retrieve variable at coordinates `x`
  static auto variables(const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
                        tmpl::list<Field> /*meta*/) noexcept
      -> tuples::TaggedTuple<Field>;

  static auto variables(const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
                        tmpl::list<::Tags::Source<Field>> /*meta*/) noexcept
      -> tuples::TaggedTuple<::Tags::Source<Field>>;

  static auto variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<::Tags::Source<AuxiliaryField<Dim>>> /*meta*/) noexcept
      -> tuples::TaggedTuple<::Tags::Source<AuxiliaryField<Dim>>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  static tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

template <size_t Dim>
bool operator==(const Lorentzian<Dim>& /*lhs*/,
                const Lorentzian<Dim>& /*rhs*/) noexcept;

template <size_t Dim>
bool operator!=(const Lorentzian<Dim>& lhs,
                const Lorentzian<Dim>& rhs) noexcept;

}  // namespace Solutions
}  // namespace Poisson
