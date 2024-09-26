// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

/// The trivial solution \f$u=0\f$ of a Poisson equation. Useful as initial
/// guess.
template <size_t Dim, typename DataType = DataVector>
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

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using supported_tags = tmpl::list<
        Tags::Field<DataType>,
        ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                      Frame::Inertial>,
        ::Tags::Flux<Tags::Field<DataType>, tmpl::size_t<Dim>, Frame::Inertial>,
        ::Tags::FixedSource<Tags::Field<DataType>>>;
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }
};

/// \cond
template <size_t Dim, typename DataType>
PUP::able::PUP_ID Zero<Dim, DataType>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim, typename DataType>
bool operator==(const Zero<Dim, DataType>& /*lhs*/,
                const Zero<Dim, DataType>& /*rhs*/) {
  return true;
}

template <size_t Dim, typename DataType>
bool operator!=(const Zero<Dim, DataType>& lhs,
                const Zero<Dim, DataType>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
