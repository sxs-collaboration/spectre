// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/AnalyticSolution.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

namespace detail {
template <typename DataType, size_t Dim>
struct MoustacheVariables {
  using Cache = CachedTempBuffer<
      MoustacheVariables, Tags::Field,
      ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Field>>;

  const tnsr::I<DataType, Dim>& x;

  void operator()(gsl::not_null<Scalar<DataType>*> field,
                  gsl::not_null<Cache*> cache, Tags::Field /*meta*/) const
      noexcept;
  void operator()(gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>,
                                Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>,
                               Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::FixedSource<Tags::Field> /*meta*/) const noexcept;
};
}  // namespace detail

/// \cond
template <size_t Dim, typename Registrars>
struct Moustache;

namespace Registrars {
template <size_t Dim>
struct Moustache {
  template <typename Registrars>
  using f = Solutions::Moustache<Dim, Registrars>;
};
}  // namespace Registrars
/// \endcond

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
 * This solution is taken from \cite Stamm2010.
 */
template <size_t Dim, typename Registrars =
                          tmpl::list<Solutions::Registrars::Moustache<Dim>>>
class Moustache : public AnalyticSolution<Dim, Registrars> {
 private:
  using Base = AnalyticSolution<Dim, Registrars>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A solution with a discontinuous first derivative of its source at 1/2 "
      "that also happens to look like a moustache. It vanishes at zero and one "
      "in each dimension"};

  Moustache() = default;
  Moustache(const Moustache&) noexcept = default;
  Moustache& operator=(const Moustache&) noexcept = default;
  Moustache(Moustache&&) noexcept = default;
  Moustache& operator=(Moustache&&) noexcept = default;
  ~Moustache() noexcept override = default;

  /// \cond
  explicit Moustache(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Moustache);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using VarsComputer = detail::MoustacheVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin()), VarsComputer{x}};
    return {cache.get_var(RequestedTags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept override {}
};

/// \cond
template <size_t Dim, typename Registrars>
PUP::able::PUP_ID Moustache<Dim, Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim, typename Registrars>
constexpr bool operator==(const Moustache<Dim, Registrars>& /*lhs*/,
                          const Moustache<Dim, Registrars>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim, typename Registrars>
constexpr bool operator!=(const Moustache<Dim, Registrars>& lhs,
                          const Moustache<Dim, Registrars>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
