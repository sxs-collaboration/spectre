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
template <typename T>
class DataVectorImpl;
using DataVector = DataVectorImpl<double>;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson {
namespace Solutions {

/*!
 * \brief A solution to the Poisson equation with a discontinuous first
 * derivative.
 *
 * \details This implements the solution \f$u(x,y)=x\left(1-x\right)
 * y\left(1-y\right)\left(\left(x-\frac{1}{2}\right)^2+\left(y-
 * \frac{1}{2}\right)^2\right)^\frac{3}{2}\f$ to the Poisson equation
 * in two dimensions, and
 * \f$u(x)=x\left(1-x\right)\left|x-\frac{1}{2}\right|^3\f$ in one dimension.
 * Their boundary conditions vanish on the square \f$[0,1]^2\f$ or interval
 * \f$[0,1]\f$, respectively.
 *
 * The corresponding source \f$f=-\Delta u\f$ has a discontinuous first
 * derivative at \f$\frac{1}{2}\f$. This accomplishes two things:
 *
 * - It makes it useful to test the convergence behaviour of our elliptic DG
 * solver.
 * - It makes it look like a moustache (at least in 1D).
 *
 * This solution is taken from _B. Stamm and T. Wihler, Mathematics of
 * Computation 79, 2117 (2010)_. It is also investigated in _T. Vincent and H.P.
 * Pfeiffer, in preparation_.
 */
template <size_t Dim>
class Moustache {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help{
      "A solution with a discontinuous first derivative of its source at 1/2 "
      "that also happens to look like a moustache. It vanishes at zero and one "
      "in each dimension"};

  Moustache() = default;
  Moustache(const Moustache&) noexcept = delete;
  Moustache& operator=(const Moustache&) noexcept = delete;
  Moustache(Moustache&&) noexcept = default;
  Moustache& operator=(Moustache&&) noexcept = default;
  ~Moustache() noexcept = default;

  auto field_variables(const tnsr::I<DataVector, Dim>& x) const noexcept
      -> tuples::TaggedTuple<Field, AuxiliaryField<Dim>>;

  auto source_variables(const tnsr::I<DataVector, Dim>& x) const noexcept
      -> tuples::TaggedTuple<::Tags::Source<Field>,
                             ::Tags::Source<AuxiliaryField<Dim>>>;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT
};

template <size_t Dim>
constexpr bool operator==(const Moustache<Dim>& /*lhs*/,
                          const Moustache<Dim>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim>
constexpr bool operator!=(const Moustache<Dim>& lhs,
                          const Moustache<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace Poisson
