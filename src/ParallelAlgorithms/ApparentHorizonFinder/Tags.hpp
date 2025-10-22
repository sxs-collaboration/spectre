// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <optional>
#include <set>
#include <string>
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
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

/// \cond
namespace control_system::OptionTags {
struct WriteDataToDisk;
}  // namespace control_system::OptionTags
class FastFlow;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm
namespace ah {
template <class Metavariables, typename HorizonMetavars>
struct Component;
}  // namespace ah
namespace Tags {
struct Time;
}  // namespace Tags
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
 * \brief Tag that holds the current time.
 *
 * \details The value of this tag is `std::nullopt` if the current time isn't
 * set.
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
 * \brief Tag that holds the current resolution L.
 *
 * \details The value of this tag is `std::nullopt` if the current resolution L
 * isn't set.
 */
struct CurrentResolutionL : db::SimpleTag {
  using type = std::optional<size_t>;
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
 * \brief Order in which blocks are searched for horizon finding. See
 * `::block_logical_coordinates` for details.
 */
struct BlockSearchOrder : db::SimpleTag {
  using type = std::vector<size_t>;
};

/*!
 * \brief Deque of `ah::Storage::PreviousSurface`s.
 */
template <typename Fr>
struct PreviousSurfaces : db::SimpleTag {
  using type = std::deque<ah::Storage::PreviousSurface<Fr>>;
};

/*!
 * \brief Holds the previous surface. Used to determine which elements will send
 * data for the next horizon find.
 */
template <typename HorizonMetavars>
struct PreviousSurface : db::SimpleTag {
  using type =
      ah::Storage::LockedPreviousSurface<typename HorizonMetavars::frame>;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return {}; }
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
  static type create_from_options(const type& option) {
    return {deserialize<type>(serialize<type>(option).data())};
  }
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
struct BlocksForHorizonFind : db::SimpleTag {
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
      if (horizon_options.blocks_for_horizon_find.has_value()) {
        result[name] = domain::expand_block_groups_to_block_names(
            horizon_options.blocks_for_horizon_find.value(), block_names,
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

/// @{
/*!
 * \brief Tag to be used for the `time_tag` alias of a `HorizonMetavars` for an
 * observation horizon find.
 *
 * \details We need separate time tags for all horizon finders because of the
 * current design of the horizon finder. So we just make a simple compute tag
 * that takes the actual time out of the box since we still want the actual time
 * to be the same, just a different tag.
 *
 */
template <size_t Index>
struct ObservationTime : db::SimpleTag {
  static std::string name() { return "AhObservationTime" + get_output(Index); }
  using type = LinkedMessageId<double>;
};

template <size_t Index>
struct ObservationTimeCompute : ObservationTime<Index>, db::ComputeTag {
  using argument_tags = tmpl::list<::Tags::Time>;
  using base = ObservationTime<Index>;
  using return_type = LinkedMessageId<double>;

  static void function(const gsl::not_null<LinkedMessageId<double>*> ah_time,
                       const double time) {
    // The horizon finder knows how to handle the nullopt
    *ah_time = LinkedMessageId<double>{time, std::nullopt};
  }
};
/// @}

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

/// Simple tag for whether to write the centers of the horizons to disk.
struct ObserveCenters : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<control_system::OptionTags::WriteDataToDisk>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace ah::Tags
