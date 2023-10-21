// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/CombinedName.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/IsSize.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Time/OptionTags/InitialTime.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
namespace control_system {
template <typename ControlSystem>
struct OptionHolder;
}  // namespace control_system
namespace domain::OptionTags {
template <size_t Dim>
struct DomainCreator;
}  // namespace domain::OptionTags
namespace OptionTags {
struct InitialTime;
}  // namespace OptionTags
/// \endcond

/// \ingroup ControlSystemGroup
/// All DataBox tags related to the control system
namespace control_system::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for writing control system data to disk
struct WriteDataToDisk : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::WriteDataToDisk>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for writing the centers of the horizons to disk.
///
/// This is controlled by the `control_system::OptionTags::WriteDataToDisk`
/// option in the input file.
struct ObserveCenters : ::ah::Tags::ObserveCentersBase, db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::WriteDataToDisk>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the averager
///
/// To compute the `deriv_order`th derivative of a control error, the max
/// derivative we need from the averager is the `deriv_order - 1`st derivative.
template <typename ControlSystem>
struct Averager : db::SimpleTag {
  using type = ::Averager<ControlSystem::deriv_order - 1>;

  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder) {
    return option_holder.averager;
  }
};

namespace detail {
template <bool AllowDecrease, size_t Dim>
void initialize_tuner(
    gsl::not_null<::TimescaleTuner<AllowDecrease>*> tuner,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
    const double initial_time, const std::string& name);
}  // namespace detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the timescale tuner
template <typename ControlSystem>
struct TimescaleTuner : db::SimpleTag {
 private:
  static constexpr bool is_size =
      control_system::size::is_size_v<ControlSystem>;

 public:
  using type = ::TimescaleTuner<not is_size>;

  template <typename Metavariables>
  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>,
                 domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
                 ::OptionTags::InitialTime>;
  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  static type create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder,
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time) {
    auto tuner = option_holder.tuner;
    detail::initialize_tuner(make_not_null(&tuner), domain_creator,
                             initial_time, ControlSystem::name());
    return tuner;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the controller
template <typename ControlSystem>
struct Controller : db::SimpleTag {
  using type = ::Controller<ControlSystem::deriv_order>;

  template <typename Metavariables>
  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>,
                 domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
                 ::OptionTags::InitialTime>;
  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  static type create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder,
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time) {
    type controller = option_holder.controller;
    auto tuner = option_holder.tuner;
    detail::initialize_tuner(make_not_null(&tuner), domain_creator,
                             initial_time, ControlSystem::name());

    controller.set_initial_update_time(initial_time);
    controller.assign_time_between_updates(min(tuner.current_timescale()));

    return controller;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the control error
template <typename ControlSystem>
struct ControlError : db::SimpleTag {
  using type = typename ControlSystem::control_error;

  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder) {
    return option_holder.control_error;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// Tag that determines how many measurements will occur per control
/// system update. This will usually be stored in the global cache.
struct MeasurementsPerUpdate : db::SimpleTag {
  using type = int;

  using option_tags = tmpl::list<OptionTags::MeasurementsPerUpdate>;
  static constexpr bool pass_metavariables = false;
  static int create_from_options(const int measurements_per_update) {
    return measurements_per_update;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag that keeps track of which measurement we are on.
struct CurrentNumberOfMeasurements : db::SimpleTag {
  using type = int;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag that holds the verbosity used to print info about the control
/// system algorithm.
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;

  using option_tags = tmpl::list<OptionTags::Verbosity>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const ::Verbosity verbosity) {
    return verbosity;
  }
};

/*!
 * \brief Tag meant to be stored in the GlobalCache that stores a map between
 * names of control systems and the "combined" name that that control system is
 * part of.
 *
 * \details The "combined" name for each control system is computed using
 * `control_system::system_to_combined_names` where the list of control systems
 * is taken from the `component_list` type alias of the metavariables. Each
 * "combined" name corresponds to a different
 * `control_system::protocols::Measurement`.
 */
struct SystemToCombinedNames : db::SimpleTag {
  using type = std::unordered_map<std::string, std::string>;

  template <typename Metavariables>
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = true;

 private:
  template <typename Component>
  using system = typename Component::control_system;

 public:
  template <typename Metavariables>
  static type create_from_options() {
    using all_control_components =
        metafunctions::all_control_components<Metavariables>;
    using all_control_systems =
        tmpl::transform<all_control_components, tmpl::bind<system, tmpl::_1>>;

    return system_to_combined_names<all_control_systems>();
  }
};

/*!
 * \brief Map between "combined" names and the
 * `control_system::UpdateAggregator`s that go with each.
 */
struct UpdateAggregators : db::SimpleTag {
  using type =
      std::unordered_map<std::string, control_system::UpdateAggregator>;
};
}  // namespace control_system::Tags
