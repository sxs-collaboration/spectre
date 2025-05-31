// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
template <class Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget;
namespace Tags {
template <typename TemporalId>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
namespace Tags {
struct Verbosity;
}  // namespace Tags
/// \endcond

namespace ah {
namespace OptionHolders {
/// Options for finding an apparent horizon.
template <typename Frame>
struct ApparentHorizon {
 private:
  struct All {};

 public:
  /// See Strahlkorper for suboptions.
  struct InitialGuess {
    static constexpr Options::String help = {"Initial guess"};
    using type = ylm::Strahlkorper<Frame>;
  };
  /// See ::FastFlow for suboptions.
  struct FastFlow {
    static constexpr Options::String help = {"FastFlow options"};
    using type = ::FastFlow;
  };
  struct Verbosity {
    static constexpr Options::String help = {"Verbosity"};
    using type = ::Verbosity;
  };
  struct MaxInterpolationRetries {
    static constexpr Options::String help = {
        "Number of times to retry the interpolation where, with each retry, "
        "the two previous surfaces are averaged and that new surface is used."};
    using type = size_t;
  };
  struct BlocksForInterpolation {
    static constexpr Options::String help = {
        "Volume data will be sent to the interpolator for these block group "
        "names. Set to 'All' to send volume data from the entire domain."};
    using type = Options::Auto<std::vector<std::string>, All>;
  };
  using options = tmpl::list<InitialGuess, FastFlow, Verbosity,
                             MaxInterpolationRetries, BlocksForInterpolation>;
  static constexpr Options::String help = {
      "Provide an initial guess for the apparent horizon surface\n"
      "(Strahlkorper) and apparent-horizon-finding-algorithm (FastFlow)\n"
      "options."};

  ApparentHorizon(
      ylm::Strahlkorper<Frame> initial_guess_in, ::FastFlow fast_flow_in,
      ::Verbosity verbosity_in, size_t max_interpolation_retries_in,
      std::optional<std::vector<std::string>> blocks_for_interpolation_in);

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon& /*rhs*/) = default;
  ApparentHorizon& operator=(const ApparentHorizon& /*rhs*/) = delete;
  ApparentHorizon(ApparentHorizon&& /*rhs*/) = default;
  ApparentHorizon& operator=(ApparentHorizon&& /*rhs*/) = default;
  ~ApparentHorizon() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  ylm::Strahlkorper<Frame> initial_guess{};
  ::FastFlow fast_flow{};
  ::Verbosity verbosity{::Verbosity::Quiet};
  size_t max_interpolation_retries{};
  std::optional<std::vector<std::string>> blocks_for_interpolation;
};

template <typename Frame>
bool operator==(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs);
template <typename Frame>
bool operator!=(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs);

}  // namespace OptionHolders

namespace OptionTags {
struct ApparentHorizons {
  static constexpr Options::String help{"Options for apparent horizon finders"};
};

template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon {
  using type = OptionHolders::ApparentHorizon<Frame>;
  static constexpr Options::String help{
      "Options for interpolation onto apparent horizon."};
  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }
  using group = ApparentHorizons;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon : db::SimpleTag {
  using type = OptionHolders::ApparentHorizon<Frame>;
  using option_tags =
      tmpl::list<OptionTags::ApparentHorizon<InterpolationTargetTag, Frame>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

namespace detail {
template <typename InterpolationTargetTags>
struct get_horizon_options;

template <typename... InterpolationTargetTags>
struct get_horizon_options<tmpl::list<InterpolationTargetTags...>> {
  using type = tmpl::list<OptionTags::ApparentHorizon<
      InterpolationTargetTags,
      typename InterpolationTargetTags::compute_target_points::frame>...>;
};

CREATE_GET_TYPE_ALIAS_OR_DEFAULT(component_being_mocked)
}  // namespace detail

/*!
 * \brief Holds a map between interpolation target tag name (aka a horizon) and
 * a set of block names that should be used for interpolation for that target.
 */
struct BlocksForInterpolation : db::SimpleTag,
                                intrp::Tags::BlocksForInterpolationBase {
  using type = std::unordered_map<std::string, std::unordered_set<std::string>>;
  template <typename Metavariables>
  using option_tags = tmpl::push_front<
      typename detail::get_horizon_options<
          intrp::InterpolationTarget_detail::get_sequential_target_tags<
              Metavariables>>::type,
      ::domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables, typename... HorizonOptions>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const HorizonOptions&... all_horizon_options) {
    return create_from_options_impl<Metavariables>(
        domain_creator, std::forward_as_tuple(all_horizon_options...),
        std::make_index_sequence<sizeof...(HorizonOptions)>{});
  }

 private:
  // Need the names of the target tags which are in the option tags, but not the
  // horizon options themselves. This just expands a tuple to be able to index
  // the `option_tags` type alias so we can get the name of the target horizon
  template <typename Metavariables, typename HorizonOptionsTuple, size_t... Is>
  static type create_from_options_impl(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const HorizonOptionsTuple& all_horizon_options,
      const std::index_sequence<Is...>& /*index_sequence*/
  ) {
    std::unordered_map<std::string, std::unordered_set<std::string>> result{};

    const auto block_names = domain_creator->block_names();
    const auto block_groups = domain_creator->block_groups();

    const auto append_to_result = [&](const std::string& name,
                                      const auto& horizon_options) {
      if (horizon_options.blocks_for_interpolation.has_value()) {
        result[name] = domain::expand_block_groups_to_block_names(
            horizon_options.blocks_for_interpolation.value(), block_names,
            block_groups);
      } else {
        // Insert all blocks
        result[name].insert(block_names.begin(), block_names.end());
      }

      // Needed for the expand_pack below
      return 0;
    };

    expand_pack(
        append_to_result(tmpl::at_c<option_tags<Metavariables>, Is + 1>::name(),
                         std::get<Is>(all_horizon_options))...);

    return result;
  }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points on a trial apparent horizon`.
///
/// This differs from `KerrHorizon` in the following ways:
/// - It supplies points on a prolonged Strahlkorper, at a higher resolution
///   than the Strahlkorper in the DataBox, as needed for horizon finding.
/// - It uses a `FastFlow` in the DataBox.
/// - It has different options (including those for `FastFlow`).
///
/// Conforms to the intrp::protocols::ComputeTargetPoints protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using const_global_cache_tags =
      tmpl::list<Tags::BlocksForInterpolation,
                 Tags::ApparentHorizon<InterpolationTargetTag, Frame>>;
  using is_sequential = std::true_type;
  using frame = Frame;

  using common_tags =
      tmpl::push_back<ylm::Tags::items_tags<Frame>, ::ah::Tags::FastFlow,
                      logging::Tags::Verbosity<InterpolationTargetTag>,
                      ylm::Tags::PreviousStrahlkorpers<Frame>,
                      ::ah::Tags::PreviousIterationStrahlkorper<Frame>,
                      ::ah::Tags::FailedInterpolationIterations>;
  using simple_tags = tmpl::append<
      common_tags,
      tmpl::conditional_t<
          std::is_same_v<Frame, ::Frame::Inertial>, tmpl::list<>,
          tmpl::list<ylm::Tags::CartesianCoords<::Frame::Inertial>>>>;
  using compute_tags =
      tmpl::append<typename ylm::Tags::compute_items_tags<Frame>,
                   ylm::Tags::TimeDerivStrahlkorperCompute<Frame>>;

  template <typename DbTags, typename Metavariables>
  static void initialize(const gsl::not_null<db::DataBox<DbTags>*> box,
                         const Parallel::GlobalCache<Metavariables>& cache) {
    const auto& options =
        Parallel::get<Tags::ApparentHorizon<InterpolationTargetTag, Frame>>(
            cache);

    // Put Strahlkorper and its ComputeItems, FastFlow, and verbosity
    // into a new DataBox.
    //
    // Note that if frame is not inertial,
    // ylm::Tags::Strahlkorper<::Frame::Inertial> is already
    // default initialized so there is no need to do anything special
    // here for ylm::Tags::Strahlkorper<::Frame::Inertial>.
    Initialization::mutate_assign<common_tags>(
        box, options.initial_guess, options.fast_flow, options.verbosity,
        std::deque<std::pair<double, ylm::Strahlkorper<Frame>>>{},
        options.initial_guess, 0_st);
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame> points(
      db::DataBox<DbTags>& box, const tmpl::type_<Metavariables>& /*meta*/,
      const TemporalId& temporal_id) {
    const auto& fast_flow = db::get<::ah::Tags::FastFlow>(box);

    if (fast_flow.current_iteration() == 0) {
      // Put new initial guess into ylm::Tags::Strahlkorper<Frame>.
      // We need to do this now, and not at the end of the previous horizon
      // search, because only now do we know the temporal_id of this horizon
      // search.
      db::mutate<ylm::Tags::Strahlkorper<Frame>>(
          [&temporal_id](
              const gsl::not_null<ylm::Strahlkorper<Frame>*> strahlkorper,
              const std::deque<std::pair<double, ylm::Strahlkorper<Frame>>>&
                  previous_strahlkorpers) {
            // If we have zero previous_strahlkorpers, then the
            // initial guess is already in strahlkorper, so do
            // nothing.
            //
            // If we have one previous_strahlkorper, then we have had
            // a successful horizon find, and the initial guess for the
            // next horizon find is already in strahlkorper, so
            // again we do nothing.
            //
            // If we have 2 valid previous_strahlkorpers, then
            // we set the initial guess by linear extrapolation in time
            // using the last 2 previous_strahlkorpers.
            //
            // If we have 3 valid previous_strahlkorpers, then
            // we set the initial guess by quadratic extrapolation in time
            // using the last 3 previous_strahlkorpers.
            //
            // For extrapolation, we assume that
            // * Expansion center of all the Strahlkorpers are equal.
            // * Maximum L of all the Strahlkorpers are equal.
            // It is easy to relax the max L assumption once we start
            // adaptively changing the L of the strahlkorpers.
            if (LIKELY(previous_strahlkorpers.size() > 2)) {
              // Quadratic extrapolation
              const double new_time =
                  intrp::InterpolationTarget_detail::get_temporal_id_value(
                      temporal_id);
              const double dt_0 = previous_strahlkorpers[0].first - new_time;
              const double dt_1 = previous_strahlkorpers[1].first - new_time;
              const double dt_2 = previous_strahlkorpers[2].first - new_time;
              const double fac_0 =
                  dt_1 * dt_2 / ((dt_1 - dt_0) * (dt_2 - dt_0));
              const double fac_1 =
                  dt_0 * dt_2 / ((dt_2 - dt_1) * (dt_0 - dt_1));
              const double fac_2 = 1.0 - fac_0 - fac_1;
              strahlkorper->coefficients() =
                  fac_0 * previous_strahlkorpers[0].second.coefficients() +
                  fac_1 * previous_strahlkorpers[1].second.coefficients() +
                  fac_2 * previous_strahlkorpers[2].second.coefficients();
            } else if (previous_strahlkorpers.size() > 1) {
              // Linear extrapolation
              const double new_time =
                  intrp::InterpolationTarget_detail::get_temporal_id_value(
                      temporal_id);
              const double dt_0 = previous_strahlkorpers[0].first - new_time;
              const double dt_1 = previous_strahlkorpers[1].first - new_time;
              const double fac_0 = dt_1 / (dt_1 - dt_0);
              const double fac_1 = 1.0 - fac_0;
              strahlkorper->coefficients() =
                  fac_0 * previous_strahlkorpers[0].second.coefficients() +
                  fac_1 * previous_strahlkorpers[1].second.coefficients();
            }
          },
          make_not_null(&box),
          db::get<ylm::Tags::PreviousStrahlkorpers<Frame>>(box));
    }

    const auto& strahlkorper = db::get<ylm::Tags::Strahlkorper<Frame>>(box);

    const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);

    const auto prolonged_strahlkorper =
        ylm::Strahlkorper<Frame>(L_mesh, L_mesh, strahlkorper);

    return ylm::cartesian_coords(prolonged_strahlkorper);
  }
};

}  // namespace TargetPoints
}  // namespace ah
