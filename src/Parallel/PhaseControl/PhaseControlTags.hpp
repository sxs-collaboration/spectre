// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "Options/String.hpp"
#include "Parallel/ExitCode.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace PhaseControl {

/// Option-creatable pair of a trigger and associated phase change objects.
struct TriggerAndPhaseChanges {
  struct Trigger {
    using type = std::unique_ptr<::Trigger>;
    static constexpr Options::String help =
        "Determines when the phase changes are evaluated.";
  };
  struct PhaseChanges {
    using type = std::vector<std::unique_ptr<::PhaseChange>>;
    static constexpr Options::String help =
        "These phase changes are evaluated when the Trigger fires.";
  };
  static constexpr Options::String help =
      "Phase changes that are evaluated when the Trigger fires.";
  using options = tmpl::list<Trigger, PhaseChanges>;
  void pup(PUP::er& p) {
    p | trigger;
    p | phase_changes;
  }
  std::unique_ptr<::Trigger> trigger;
  std::vector<std::unique_ptr<::PhaseChange>> phase_changes;
};

namespace OptionTags {
/// Option tag for the collection of triggers that indicate synchronization
/// points at which phase changes should be considered, and the associated
/// `PhaseChange` objects for making the phase change decisions.
///
/// When the phase control is arbitrated on the main chare, the `PhaseChange`
/// objects will be queried for their phase request in order of appearance in
/// the nested list (i.e. first all of the `PhaseChange`s associated with the
/// first trigger, in order, then those associated with the second trigger,
/// etc.). The order therefore determines the order of resolution of
/// simultaneous requests.
struct PhaseChangeAndTriggers {
  static constexpr Options::String help{
      "A collection of pairs of triggers and collections of phase change "
      "objects to determine runtime phase control-flow decisions. The order of "
      "the phase change objects determines the order of the requests processed "
      "by the Main chare during phase change arbitration."};

  using type = std::vector<TriggerAndPhaseChanges>;
};
}  // namespace OptionTags

namespace Tags {
/// Tag for the collection of triggers that indicate synchronization points at
/// which phase changes should be considered, and the associated `PhaseChange`
/// objects for making the phase change decisions.
struct PhaseChangeAndTriggers : db::SimpleTag {
  using type = std::vector<TriggerAndPhaseChanges>;

  using option_tags = tmpl::list<OptionTags::PhaseChangeAndTriggers>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& phase_control_and_triggers) {
    return deserialize<type>(
        serialize<type>(phase_control_and_triggers).data());
  }
};
}  // namespace Tags

namespace TagsAndCombines {
/// A tag for indicating that a halt was called by a trigger associated with
/// `PhaseChange`s.
///
/// This is needed to disambiguate different quiescence conditions in the main
/// chare. It is automatically included in
/// `PhaseControl::get_phase_change_tags`, so shouldn't be explicitly included
/// in the `phase_change_tags_and_combines` in derived classes of
/// `PhaseChange`.
struct UsePhaseChangeArbitration {
  using type = bool;
  using combine_method = funcl::Or<>;
  using main_combine_method = combine_method;
};
}  // namespace TagsAndCombines

namespace detail {
template <typename Metavariables, typename = std::void_t<>>
struct phase_change_derived_classes {
  using type = tmpl::list<>;
};

template <typename Metavariables>
struct phase_change_derived_classes<
    Metavariables,
  std::void_t<typename Metavariables::factory_creation::factory_classes>> {
 private:
  using factory_entry =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               PhaseChange>;

 public:
  using type =
      tmpl::conditional_t<std::is_same_v<factory_entry, tmpl::no_such_type_>,
                          tmpl::list<>, factory_entry>;
};

template <typename PhaseChangeDerived>
struct get_phase_change_tags_and_combines {
  using type = typename PhaseChangeDerived::phase_change_tags_and_combines;
};
}  // namespace detail

/// Metafunction for determining the merged collection of tags in
/// `phase_change_tags_and_combines`s from all `PhaseChange` derived
/// classes in `Metavariables::factory_creation`
template <typename Metavariables>
using get_phase_change_tags = tmpl::push_back<
    tmpl::flatten<tmpl::transform<
        typename detail::phase_change_derived_classes<Metavariables>::type,
        detail::get_phase_change_tags_and_combines<tmpl::_1>>>,
    TagsAndCombines::UsePhaseChangeArbitration, Parallel::Tags::ExitCode>;
}  // namespace PhaseControl
