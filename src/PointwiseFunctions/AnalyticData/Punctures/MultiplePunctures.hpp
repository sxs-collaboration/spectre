// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <vector>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Punctures::AnalyticData {

/// A puncture representing a black hole
struct Puncture {
  struct Position {
    using type = std::array<double, 3>;
    static constexpr Options::String help{"The position C of the puncture"};
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help{"The puncture mass (bare mass) M"};
    static double lower_bound() { return 0.; }
  };
  struct Momentum {
    using type = std::array<double, 3>;
    static constexpr Options::String help{
        "The dimensionless linear momentum P / M, where M is the bare mass of "
        "the puncture."};
  };
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help{
        "The dimensionless angular momentum S / M^2, where M is the bare mass "
        "of the puncture."};
  };
  using options = tmpl::list<Position, Mass, Momentum, Spin>;
  static constexpr Options::String help{"A puncture representing a black hole"};

  std::array<double, 3> position{
      {std::numeric_limits<double>::signaling_NaN()}};
  double mass = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> dimensionless_momentum{
      {std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> dimensionless_spin{
      {std::numeric_limits<double>::signaling_NaN()}};

  void pup(PUP::er& p) {
    p | position;
    p | mass;
    p | dimensionless_momentum;
    p | dimensionless_spin;
  }
};

bool operator==(const Puncture& lhs, const Puncture& rhs);

bool operator!=(const Puncture& lhs, const Puncture& rhs);

namespace detail {

namespace Tags {
struct OneOverAlpha : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

struct MultiplePuncturesVariables {
  static constexpr size_t Dim = 3;

  using Cache =
      CachedTempBuffer<detail::Tags::OneOverAlpha, Punctures::Tags::Alpha,
                       Punctures::Tags::TracelessConformalExtrinsicCurvature,
                       Punctures::Tags::Beta,
                       ::Tags::FixedSource<Punctures::Tags::Field>>;

  const tnsr::I<DataVector, 3>& x;
  const std::vector<Puncture>& punctures;

  void operator()(gsl::not_null<Scalar<DataVector>*> one_over_alpha,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::OneOverAlpha /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataVector>*> alpha,
                  gsl::not_null<Cache*> cache,
                  Punctures::Tags::Alpha /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::II<DataVector, 3>*>
          traceless_conformal_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      Punctures::Tags::TracelessConformalExtrinsicCurvature /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataVector>*> beta,
                  gsl::not_null<Cache*> cache,
                  Punctures::Tags::Beta /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataVector>*> fixed_source_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::FixedSource<Punctures::Tags::Field> /*meta*/) const;
};
}  // namespace detail

/*!
 * \brief Superposition of multiple punctures
 *
 * This class provides the source fields $\alpha$ and $\beta$ for the puncture
 * equation (see \ref ::Punctures) representing any number of black holes. Each
 * black hole is characterized by its "puncture mass" (or "bare mass") $M_I$,
 * position $\mathbf{C}_I$, linear momentum $\mathbf{P}_I$, and angular momentum
 * $\mathbf{S}_I$. The corresponding Bowen-York solution to the momentum
 * constraint for the conformal traceless extrinsic curvature is:
 *
 * \begin{equation}
 * \bar{A}^{ij} = \frac{3}{2} \sum_I \frac{1}{r_I} \left(
 * 2 P_I^{(i} n_I^{j)} - (\delta^{ij} - n_I^i n_I^j) P_I^k n_I^k
 * + \frac{4}{r_I} n_I^{(i} \epsilon^{j)kl} S_I^k n_I^l\right)
 * \end{equation}
 *
 * From it, we compute $\alpha$ and $\beta$ as:
 *
 * \begin{align}
 * \frac{1}{\alpha} &= \sum_I \frac{M_I}{2 r_I} \\
 * \beta &= \frac{1}{8} \alpha^7 \bar{A}_{ij} \bar{A}^{ij}
 * \end{align}
 *
 * \see ::Punctures
 */
class MultiplePunctures : public elliptic::analytic_data::Background,
                          public elliptic::analytic_data::InitialGuess {
 public:
  struct Punctures {
    static constexpr Options::String help =
        "Parameters for each puncture, representing black holes";
    using type = std::vector<Puncture>;
  };
  using options = tmpl::list<Punctures>;
  static constexpr Options::String help = "Any number of black holes";

  MultiplePunctures() = default;
  MultiplePunctures(const MultiplePunctures&) = default;
  MultiplePunctures& operator=(const MultiplePunctures&) = default;
  MultiplePunctures(MultiplePunctures&&) = default;
  MultiplePunctures& operator=(MultiplePunctures&&) = default;
  ~MultiplePunctures() = default;

  MultiplePunctures(std::vector<Puncture> punctures)
      : punctures_(std::move(punctures)) {}

  explicit MultiplePunctures(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MultiplePunctures);

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    typename detail::MultiplePuncturesVariables::Cache cache{
        get_size(*x.begin())};
    const detail::MultiplePuncturesVariables computer{x, punctures_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& /*mesh*/,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    elliptic::analytic_data::Background::pup(p);
    elliptic::analytic_data::InitialGuess::pup(p);
    p | punctures_;
  }

  const std::vector<Puncture>& punctures() const { return punctures_; }

 private:
  std::vector<Puncture> punctures_{};
};

}  // namespace Punctures::AnalyticData
