// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

/// \cond
template <size_t Dim, typename Registrars>
struct Zero;

namespace Registrars {
template <size_t Dim>
struct Zero {
  template <typename Registrars>
  using f = Solutions::Zero<Dim, Registrars>;
};
}  // namespace Registrars
/// \endcond

/// The trivial solution \f$u=0\f$ of a Poisson equation. Useful as initial
/// guess.
template <size_t Dim,
          typename Registrars = tmpl::list<Solutions::Registrars::Zero<Dim>>>
class Zero : public AnalyticSolution<Dim, Registrars> {
 private:
  using Base = AnalyticSolution<Dim, Registrars>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "The trivial solution, useful as initial guess."};

  Zero() = default;
  Zero(const Zero&) noexcept = default;
  Zero& operator=(const Zero&) noexcept = default;
  Zero(Zero&&) noexcept = default;
  Zero& operator=(Zero&&) noexcept = default;
  ~Zero() noexcept override = default;

  /// \cond
  explicit Zero(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Zero);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using supported_tags = tmpl::list<
        Tags::Field,
        ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
        ::Tags::FixedSource<Tags::Field>>;
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept override {}
};

/// \cond
template <size_t Dim, typename Registrars>
PUP::able::PUP_ID Zero<Dim, Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim, typename Registrars>
bool operator==(const Zero<Dim, Registrars>& /*lhs*/,
                const Zero<Dim, Registrars>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim, typename Registrars>
bool operator!=(const Zero<Dim, Registrars>& lhs,
                const Zero<Dim, Registrars>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
