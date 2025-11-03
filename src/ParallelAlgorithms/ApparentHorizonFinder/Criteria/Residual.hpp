// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
/*!
 * \brief Recommend an updated resolution $L_{\rm new}$ based on the
 * resolution $L$ of the Strahlkorper and its residual $\delta$.
 *
 * \details The returned recommended resolution $L_{\rm new}$ depends on
 * the following options:
 * - MinResidual: a minimum residual $\delta_{\rm min}$
 * - MaxResidual: a maximum residual $\delta_{\rm max}$
 * - MinResolutionL: a minimum resolution $L_{\rm min}$
 *
 * The maximum resolution $L_{\rm max}$ is provided by the global cache entry
 * `ah::Tags::LMax`. The value returned for $L_{\rm new}$ is
 * then determined as follows:
 * - If $L > L_{\rm min}$ and $\delta < \delta_{\rm min}$, then
 * $L_{\rm new} = L - 1$
 * - If $L < L_{\rm max}$ and $\delta > \delta_{\rm max}$, then
 * $L_{\rm new} = L + 1$
 * - Otherwise, $L_{\rm new} = L$
 * Note that the residual is obtained from the provided FastFlow::IterInfo. The
 * residual is the gr::surfaces::expansion weighted by the FastFlow weights.
 */
class Residual : public Criterion {
 public:
  struct MinResidual {
    using type = double;
    static constexpr Options::String help = {"The minimum residual."};
    static type lower_bound() { return 0.0; }
  };
  struct MaxResidual {
    using type = double;
    static constexpr Options::String help = {"The maximum residual."};
    static type lower_bound() { return 0.0; }
  };
  struct MinResolutionL {
    using type = size_t;
    static constexpr Options::String help = {"The minimum resolution."};
    // Strahlkorper default constructor sets L=2, so don't allow L below that
    static type lower_bound() { return 2; }
  };
  using options = tmpl::list<MinResidual, MaxResidual, MinResolutionL>;
  static constexpr Options::String help = {
      "Use Strahlkorper residual and resolution to suggest a new "
      "resolution."};

  Residual() = default;
  Residual(double min_residual, double max_residual, size_t min_resolution_l,
           const Options::Context& context = {});

  /// \cond
  explicit Residual(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Residual);  // NOLINT
  /// \endcond

  std::string observation_name() override { return "Residual"; }

  using argument_tags = tmpl::list<>;
  using compute_tags_for_observation_box = tmpl::list<>;

  template <typename Metavariables, typename Frame>
  size_t operator()(const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ylm::Strahlkorper<Frame>& strahlkorper,
                    const FastFlow::IterInfo& info) const;

  void pup(PUP::er& p) override;

  bool is_equal(const Criterion& other) const override;

 private:
  double min_residual_{std::numeric_limits<double>::signaling_NaN()};
  double max_residual_{std::numeric_limits<double>::signaling_NaN()};
  size_t min_resolution_l_{};
};

// Out-of-line definition
/// \cond
template <typename Metavariables, typename Frame>
size_t Residual::operator()(const Parallel::GlobalCache<Metavariables>& cache,
                            const ylm::Strahlkorper<Frame>& strahlkorper,
                            const FastFlow::IterInfo& info) const {
  const auto& max_resolution_l = Parallel::get<ah::Tags::LMax>(cache);
  ASSERT(min_resolution_l_ <= max_resolution_l,
         "MinResolutionL (" << min_resolution_l_ << ") must not exceed LMax ("
                            << max_resolution_l << ").");
  if (UNLIKELY(min_resolution_l_ == max_resolution_l)) {
    ASSERT(min_resolution_l_ == strahlkorper.l_max(),
           "If MinResolutionL == LMax, strahlkorper "
           "resolution must also equal MinResolutionL, but here MinResolutionL "
           "is "
               << min_resolution_l_ << ", LMax is " << max_resolution_l
               << ", and the current resolution is " << strahlkorper.l_max());
    return strahlkorper.l_max();
  }
  if (strahlkorper.l_max() > min_resolution_l_ and
      info.residual_ylm < min_residual_ and strahlkorper.l_max() > 0) {
    return strahlkorper.l_max() - 1;
  }
  if (strahlkorper.l_max() < max_resolution_l and
      info.residual_ylm > max_residual_) {
    return strahlkorper.l_max() + 1;
  }
  return strahlkorper.l_max();
}
/// \endcond
}  // namespace ah::Criteria
