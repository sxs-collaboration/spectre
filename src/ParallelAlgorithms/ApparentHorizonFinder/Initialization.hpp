// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
/*!
 * \brief Initialize items related to the horizon finder
 *
 * GlobalCache:
 * - Uses:
 *   - `ah::Tags::ApparentHorizonOptions`
 *   - `ah::Tags::BlocksForHorizonFind`
 *   - `ah::Tags::LMax`
 *
 * DataBox:
 * - Uses: Nothing
 * - Adds:
 *   - `ah::Tags::Verbosity`
 *   - `ah::Tags::FastFlow`
 *   - `ah::Tags::CurrentTime`
 *   - `ah::Tags::PendingTimes`
 *   - `ah::Tags::CompletedTimes`
 *   - `ah::Tags::Storage`
 *   - `ah::Tags::PreviousSurfaces`
 *   - `ah::Tags::Strahlkorper`
 *   - `ah::Tags::TimeDerivStrahlkorper`
 *   - `ah::Tags::Dependency`
 *   - `::Tags::Variables<ah::vars_to_interpolate_to_target>`
 *   - `ah::Tags::CurrentResolutionL`
 * - Modifies:
 *   - `ah::Tags::Verbosity`
 *   - `ah::Tags::FastFlow`
 *   - `ah::Tags::CurrentTime`
 */
template <typename HorizonMetavars>
struct Initialize {
  using Fr = typename HorizonMetavars::frame;

  using simple_tags_from_options = tmpl::list<>;

  using simple_tags = tmpl::append<
      tmpl::list<Tags::CurrentResolutionL, Tags::Verbosity, Tags::FastFlow,
                 Tags::CurrentTime, Tags::PendingTimes, Tags::CompletedTimes,
                 Tags::Storage<Fr>, Tags::BlockSearchOrder,
                 Tags::PreviousSurfaces<Fr>, ylm::Tags::Strahlkorper<Fr>,
                 ylm::Tags::TimeDerivStrahlkorper<Fr>, ah::Tags::Dependency,
                 ::Tags::Variables<ah::vars_to_interpolate_to_target<3, Fr>>>,
      tmpl::conditional_t<
          std::is_same_v<Fr, Frame::Inertial>, tmpl::list<>,
          tmpl::list<ylm::Tags::CartesianCoords<Frame::Inertial>>>>;

  using const_global_cache_tags =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::append<
          tmpl::list<Tags::ApparentHorizonOptions<HorizonMetavars>,
                     Tags::BlocksForHorizonFind, Tags::LMax>,
          Parallel::get_const_global_cache_tags_from_actions<tmpl::flatten<
              tmpl::list<typename HorizonMetavars::horizon_find_callbacks,
                         typename HorizonMetavars::
                             horizon_find_failure_callbacks>>>>>>;

  using mutable_global_cache_tags =
      tmpl::list<Tags::PreviousSurface<HorizonMetavars>>;

  using compute_tags = ah::compute_items_on_target<3, Fr>;

  using return_tags = tmpl::list<Tags::CurrentResolutionL, Tags::Verbosity,
                                 Tags::FastFlow, Tags::CurrentTime>;

  using argument_tags =
      tmpl::list<Tags::ApparentHorizonOptions<HorizonMetavars>>;

  static void apply(
      const gsl::not_null<std::optional<size_t>*> current_resolution_l,
      const gsl::not_null<::Verbosity*> verbosity,
      const gsl::not_null<::FastFlow*> fast_flow,
      const gsl::not_null<std::optional<LinkedMessageId<double>>*> current_time,
      const HorizonOptions<Fr>& options) {
    (*verbosity) = options.verbosity;
    (*fast_flow) = options.fast_flow;
    current_time->reset();
    current_resolution_l->reset();
  }
};
}  // namespace ah
