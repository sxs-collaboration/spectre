// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Elasticity::Solutions {

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

/*!
 * \brief The trivial solution \f$\xi^i(x)=0\f$ of the Elasticity equations.
 * Useful as initial guess.
 *
 * \note This class derives off `::AnalyticData` because, while it is
 * technically an analytic solution, it will only be used as initial guess and
 * thus doesn't need to implement the additional requirements of the
 * `Elasticity::Solutions::AnalyticSolution` base class. In particular, it
 * doesn't need to provide a constitutive relation.
 */
template <size_t Dim,
          typename Registrars = tmpl::list<Solutions::Registrars::Zero<Dim>>>
class Zero : public ::AnalyticData<Dim, Registrars> {
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
  explicit Zero(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Zero);  // NOLINT
  /// \endcond

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using supported_tags =
        tmpl::list<Tags::Displacement<Dim>, Tags::Strain<Dim>,
                   Tags::MinusStress<Dim>, Tags::PotentialEnergyDensity<Dim>,
                   ::Tags::FixedSource<Tags::Displacement<Dim>>>;
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
bool operator!=(const Zero<Dim, Registrars>& /*lhs*/,
                const Zero<Dim, Registrars>& /*rhs*/) noexcept {
  return false;
}
}  // namespace Elasticity::Solutions
