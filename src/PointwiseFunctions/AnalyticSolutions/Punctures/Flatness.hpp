// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Punctures::Solutions {

/// Flat spacetime. Useful as initial guess.
class Flatness : public elliptic::analytic_data::AnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Flat spacetime, useful as initial guess."};

  Flatness() = default;
  Flatness(const Flatness&) = default;
  Flatness& operator=(const Flatness&) = default;
  Flatness(Flatness&&) = default;
  Flatness& operator=(Flatness&&) = default;
  ~Flatness() = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Flatness>(*this);
  }

  /// \cond
  explicit Flatness(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Flatness);
  /// \endcond

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    // These are all zero
    using supported_tags =
        tmpl::list<Tags::Field,
                   ::Tags::deriv<Tags::Field, tmpl::size_t<3>, Frame::Inertial>,
                   ::Tags::Flux<Tags::Field, tmpl::size_t<3>, Frame::Inertial>,
                   Punctures::Tags::Alpha,
                   Punctures::Tags::TracelessConformalExtrinsicCurvature,
                   Punctures::Tags::Beta,
                   ::Tags::FixedSource<Punctures::Tags::Field>>;
    static_assert(
        std::is_same_v<
            tmpl::list_difference<tmpl::list<RequestedTags...>, supported_tags>,
            tmpl::list<>>,
        "Not all requested tags are supported. The static_assert lists the "
        "unsupported tags.");
    const auto make_value = [&x](auto tag_v) {
      using tag = std::decay_t<decltype(tag_v)>;
      return make_with_value<typename tag::type>(x, 0.);
    };
    return {make_value(RequestedTags{})...};
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& /*mesh*/,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }
};

bool operator==(const Flatness& lhs, const Flatness& rhs);

bool operator!=(const Flatness& lhs, const Flatness& rhs);

}  // namespace Punctures::Solutions
