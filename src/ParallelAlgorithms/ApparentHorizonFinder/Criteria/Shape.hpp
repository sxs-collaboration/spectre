// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>

#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
/*!
 * \brief Recommend an updated resolution $L_{\rm new}$ based on how well
 * the shape of the apparent horizon is resolved in terms of its
 * truncation error and pile up modes.
 *
 * \details The returned recommended resolution $L_{\rm new}$ depends on
 * the following options:
 * - MinTruncationError: a minimum truncation error $\mathcal{T}_{\rm min}$
 * - MaxTruncationError: a maximum truncation error $\mathcal{T}_{\rm max}$
 * - MaxPileUpModes: a maximum number of pile up modes $\mathcal{P}_{\rm max}$
 * - MinResolutionL: a minimum resolution $L_{\rm min}$
 * - MaxResolutionL: a maximum resolution $L_{\rm max}$
 * The value returned for $L_{\rm new}$ is then determined as follows:
 * - If $L > L_{\rm min}$ and $\mathcal{T} < \mathcal{T}_{\rm min}$, then
 * $L_{\rm new} = L - 1$
 * - If $L < L_{\rm max}$ and $\mathcal{T} > \mathcal{T}_{\rm max}$ and
 * $\mathcal{P} \le \mathcal{P}_{\rm max}$, then
 * $L_{\rm new} = L + 1$
 * - Otherwise, $L_{\rm new} = L$
 * Note that the truncation error is obtained via
 * PowerMonitors::relative_truncation_error, and the number of pileup modes
 * is given by PowerMonitors::convergence_rate_and_number_of_pile_up_modes
 */
class Shape : public Criterion {
 public:
  struct MinTruncationError {
    using type = double;
    static constexpr Options::String help = {"The minimum truncation error."};
    static type lower_bound() { return 0.0; }
  };
  struct MaxTruncationError {
    using type = double;
    static constexpr Options::String help = {"The maximum truncation error."};
    static type lower_bound() { return 0.0; }
  };
  struct MaxPileUpModes {
    using type = size_t;
    static constexpr Options::String help = {
        "The maximum number of pile up modes."};
    static type lower_bound() { return 0; }
  };
  struct MinResolutionL {
    using type = size_t;
    static constexpr Options::String help = {"The minimum resolution."};
    // Power montior requires at least L=4, so don't allow L below that
    static type lower_bound() { return 4; }
  };
  struct MaxResolutionL {
    using type = size_t;
    static constexpr Options::String help = {"The maximum resolution."};
    // Power monitor requires at least L=4, so don't allow L below that
    static type lower_bound() { return 4; }
  };

  using options = tmpl::list<MinTruncationError, MaxTruncationError,
                             MaxPileUpModes, MinResolutionL, MaxResolutionL>;
  static constexpr Options::String help = {
      "Use Strahlkorper truncation error, number of pile up modes, and "
      "resolution to suggest a new resolution."};

  Shape() = default;
  Shape(double min_truncation_error, double max_truncation_error,
        size_t max_pile_up_modes, size_t min_resolution_l,
        size_t max_resolution_l, const Options::Context& context = {});

  /// \cond
  explicit Shape(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Shape);  // NOLINT
  /// \endcond

  std::string observation_name() override { return "Shape"; }

  using argument_tags = tmpl::list<>;
  using compute_tags_for_observation_box = tmpl::list<>;

  template <typename Metavariables, typename Frame>
  size_t operator()(const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ylm::Strahlkorper<Frame>& strahlkorper,
                    const FastFlow::IterInfo& /*info*/) const;

  void pup(PUP::er& p) override;

 private:
  double min_truncation_error_{std::numeric_limits<double>::signaling_NaN()};
  double max_truncation_error_{std::numeric_limits<double>::signaling_NaN()};
  size_t max_pile_up_modes_{};
  size_t min_resolution_l_{};
  size_t max_resolution_l_{};
};

// Out-of-line definition
/// \cond
template <typename Metavariables, typename Frame>
size_t Shape::operator()(const Parallel::GlobalCache<Metavariables>& /*cache*/,
                         const ylm::Strahlkorper<Frame>& strahlkorper,
                         const FastFlow::IterInfo& /*info*/) const {
  if (UNLIKELY(min_resolution_l_ == max_resolution_l_)) {
    ASSERT(min_resolution_l_ == strahlkorper.l_max(),
           "If MinResolutionL == MaxResolutionL, strahlkorper resolution must "
           "also equal MinResolution L, but here MinResolutionL is "
               << min_resolution_l_ << ", MaxResolutionL is "
               << max_resolution_l_ << ", and the current resolution is "
               << strahlkorper.l_max());
    return strahlkorper.l_max();
  }

  const DataVector& power_monitor = ylm::power_monitor(strahlkorper);
  const double truncation_error = PowerMonitors::relative_truncation_error(
      power_monitor, power_monitor.size());
  const size_t pile_up_modes =
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(power_monitor)
          .number_of_pile_up_modes;
  if (strahlkorper.l_max() > min_resolution_l_ and
      truncation_error < min_truncation_error_ and strahlkorper.l_max() > 0) {
    return strahlkorper.l_max() - 1;
  }
  if (strahlkorper.l_max() < max_resolution_l_ and
      truncation_error > max_truncation_error_ and
      pile_up_modes <= max_pile_up_modes_) {
    return strahlkorper.l_max() + 1;
  }
  return strahlkorper.l_max();
}
/// \endcond
}  // namespace ah::Criteria
