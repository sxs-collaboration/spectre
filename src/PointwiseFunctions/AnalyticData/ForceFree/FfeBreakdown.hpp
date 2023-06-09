// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::AnalyticData {
/*!
 * \brief A test problem designed to show that the system initially satisfying
 * the force-free conditions may violate those in a later time.
 *
 * This test was originally performed by \cite Komissarov2002. We use the
 * initial data with a linear transition layer as \cite Etienne2017.
 *
 * \f{align*}{
 *  B^x & = 1.0 , \\
 *  B^y & = B^z = \left\{\begin{array}{ll}
 *  1.0         & \text{if } x < -0.1 \\
 *  -10x        & \text{if } -0.1 \leq x \leq 0.1 \\
 *  -1.0        & \text{if } x > 0.1 \\
 * \end{array}\right\}, \\
 *  E^x & = 0.0  , \\
 *  E^y & = 0.5  , \\
 *  E^z & = -0.5 .
 * \f}
 *
 * As time progresses, $B^2-E^2$ approaches to zero.
 *
 */
class FfeBreakdown : public evolution::initial_data::InitialData,
                     public MarkAsAnalyticData {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{"A FFE breakdown problem"};

  FfeBreakdown() = default;
  FfeBreakdown(const FfeBreakdown&) = default;
  FfeBreakdown& operator=(const FfeBreakdown&) = default;
  FfeBreakdown(FfeBreakdown&&) = default;
  FfeBreakdown& operator=(FfeBreakdown&&) = default;
  ~FfeBreakdown() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit FfeBreakdown(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FfeBreakdown);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables at (x,t).
  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeE> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeE>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeB> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeB>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildePsi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePsi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildePhi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePhi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeQ> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeQ>;
  /// @}

  /// Retrieve a collection of EM variables at position x
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

 private:
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const FfeBreakdown& lhs, const FfeBreakdown& rhs);
  friend bool operator!=(const FfeBreakdown& lhs, const FfeBreakdown& rhs);
};

}  // namespace ForceFree::AnalyticData
