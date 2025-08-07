// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
/*!
 * \brief Increases the resolution of the Strahlkorper by 2.
 *
 * Useful to force the horizon finder to adopt the maximum allowed resolution
 * and as a simple criterion for testing.
 */
class IncreaseResolution : public Criterion {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Increases the resolution of the Strahlkorper by 2."};

  IncreaseResolution() = default;

  /// \cond
  explicit IncreaseResolution(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(IncreaseResolution);  // NOLINT
  /// \endcond

  std::string observation_name() override { return "IncreaseResolution"; }

  using argument_tags = tmpl::list<>;
  using compute_tags_for_observation_box = tmpl::list<>;

  template <typename Metavariables, typename Frame>
  size_t operator()(const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ylm::Strahlkorper<Frame>& strahlkorper,
                    const FastFlow::IterInfo& /*info*/) const {
    return strahlkorper.l_max() + 2;
  }

  bool is_equal(const Criterion& other) const override;
};
}  // namespace ah::Criteria
