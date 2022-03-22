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
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Elasticity::Solutions {

/*!
 * \brief The trivial solution \f$\xi^i(x)=0\f$ of the Elasticity equations.
 * Useful as initial guess.
 */
template <size_t Dim>
class Zero : public elliptic::analytic_data::AnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "The trivial solution, useful as initial guess."};

  Zero() = default;
  Zero(const Zero&) = default;
  Zero& operator=(const Zero&) = default;
  Zero(Zero&&) = default;
  Zero& operator=(Zero&&) = default;
  ~Zero() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Zero>(*this);
  }

  /// \cond
  explicit Zero(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Zero);  // NOLINT
  /// \endcond

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using supported_tags =
        tmpl::list<Tags::Displacement<Dim>, Tags::Strain<Dim>,
                   Tags::MinusStress<Dim>, Tags::PotentialEnergyDensity<Dim>,
                   ::Tags::FixedSource<Tags::Displacement<Dim>>>;
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID Zero<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim>
bool operator==(const Zero<Dim>& /*lhs*/, const Zero<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const Zero<Dim>& /*lhs*/, const Zero<Dim>& /*rhs*/) {
  return false;
}
}  // namespace Elasticity::Solutions
