// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <set>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

/// \cond
class FastFlow;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm
namespace ah {
template <class Metavariables, typename HorizonMetavars>
struct Component;
}  // namespace ah
/// \endcond

/*!
 * \brief Tags for the apparent horizon finder.
 */
namespace ah::Tags {
/*!
 * \brief Verbosity of horizon finder
 */
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;
};

/*!
 * \brief Holds a `::FastFlow` object. Needs to be reset after each horizon
 * find.
 */
struct FastFlow : db::SimpleTag {
  using type = ::FastFlow;
};

/*!
 * \brief Tag that holds all completed times
 */
struct CurrentTime : db::SimpleTag {
  using type = std::optional<LinkedMessageId<double>>;
};

/*!
 * \brief List of times waiting for previous horizon finds to finish
 * before they can be started.
 */
struct PendingTimes : db::SimpleTag {
  using type = std::set<LinkedMessageId<double>>;
};

/*!
 * \brief Tag that holds all completed times
 */
struct CompletedTimes : db::SimpleTag {
  using type = std::set<LinkedMessageId<double>>;
};

/*!
 * \brief Holds potential dependency for apparent horizon callbacks.
 */
struct Dependency : db::SimpleTag {
  using type = std::optional<std::string>;
};

/*!
 * \brief Storage of all variables (volume or interpolated) for all times of the
 * horizon finder.
 */
template <typename Fr>
struct Storage : db::SimpleTag {
  using type = std::unordered_map<LinkedMessageId<double>,
                                  ah::Storage::SingleTimeStorage<Fr>>;
};

/*!
 * \brief Deque of `ah::Storage::PreviousSurface`s.
 */
template <typename Fr>
struct PreviousSurfaces : db::SimpleTag {
  using type = std::deque<ah::Storage::PreviousSurface<Fr>>;
};

/*!
 * \brief Global cache tag that holds horizon finder options
 */
template <typename HorizonMetavars>
struct ApparentHorizonOptions : db::SimpleTag {
  using type = HorizonOptions<typename HorizonMetavars::frame>;
  using option_tags =
      tmpl::list<OptionTags::ApparentHorizonOptions<HorizonMetavars>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

namespace tags_detail {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(component_being_mocked)

template <typename HorizonComponent>
struct get_horizon_metavars_from_component {
  using type = typename HorizonComponent::horizon_metavars;
};

template <typename Metavariables>
using get_horizon_metavars = tmpl::transform<
    tmpl::filter<tmpl::transform<
                     typename Metavariables::component_list,
                     get_component_being_mocked_or_default<tmpl::_1, tmpl::_1>>,
                 tt::is_a<ah::Component, tmpl::_1>>,
    get_horizon_metavars_from_component<tmpl::_1>>;

template <typename HorizonMetavars>
struct get_horizon_options;

template <typename... HorizonMetavars>
struct get_horizon_options<tmpl::list<HorizonMetavars...>> {
  using type =
      tmpl::list<OptionTags::ApparentHorizonOptions<HorizonMetavars>...>;
};

}  // namespace tags_detail

/*!
 * \brief Holds a map between horizon name and a set of block names that should
 * be used for interpolation for that horizon.
 */
struct BlocksForInterpolation2 : db::SimpleTag {
  using type = std::unordered_map<std::string, std::unordered_set<std::string>>;
  template <typename Metavariables>
  using option_tags = tmpl::push_front<
      typename tags_detail::get_horizon_options<
          tags_detail::get_horizon_metavars<Metavariables>>::type,
      ::domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables, typename... HorizonOptionClasses>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const HorizonOptionClasses&... all_horizon_options) {
    return create_from_options_impl<Metavariables>(
        domain_creator, std::forward_as_tuple(all_horizon_options...),
        std::make_index_sequence<sizeof...(HorizonOptionClasses)>{});
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

/*!
 * \brief Tag that holds the strahlkorper of the previous FastFlow iteration
 * (not the strahlkorper of the entire previous horizon find.)
 */
template <typename Frame>
struct PreviousIterationStrahlkorper : db::SimpleTag {
  using type = ylm::Strahlkorper<Frame>;
};

/*!
 * \brief Tag to hold the number of failed interpolations to a surface during
 * iterations of the FastFlow algorithm.
 */
struct FailedInterpolationIterations : db::SimpleTag {
  using type = size_t;
};

/// Base tag for whether or not to write the centers of the horizons to disk.
/// Most likely to be used in the `ObserveCenters` post horizon find callback
///
/// Other things can control whether the horizon centers are output by defining
/// their own simple tag from this base tag.
struct ObserveCentersBase : db::BaseTag {};

/// Simple tag for whether to write the centers of the horizons to disk.
/// Currently this tag is not creatable by options
struct ObserveCenters : ObserveCentersBase, db::SimpleTag {
  using type = bool;
};
}  // namespace ah::Tags
