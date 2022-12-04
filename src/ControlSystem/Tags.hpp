// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

/// \cond
namespace ah {
enum class ObjectLabel;
}  // namespace ah
namespace control_system {
template <typename ControlSystem>
struct OptionHolder;
}  // namespace control_system
namespace domain {
namespace OptionTags {
template <size_t Dim>
struct DomainCreator;
}  // namespace OptionTags
namespace FunctionsOfTime::OptionTags {
struct FunctionOfTimeFile;
struct FunctionOfTimeNameMap;
}  // namespace FunctionsOfTime::OptionTags
}  // namespace domain
namespace OptionTags {
struct InitialTime;
}  // namespace OptionTags
/// \endcond

namespace control_system {
/// \ingroup ControlSystemGroup
/// All tags that will be used in the LinkedMessageQueue's within control
/// systems.
///
/// These tags will be used to retrieve the results of the measurements that
/// were sent to the control system which have been placed inside a
/// LinkedMessageQueue.
namespace QueueTags {
/// \ingroup ControlSystemGroup
/// Holds the centers of each horizon from measurements as DataVectors
template <::ah::ObjectLabel Horizon>
struct Center {
  using type = DataVector;
};

/// \ingroup ControlSystemGroup
/// Holds a full strahlkorper from measurements
template <typename Frame>
struct Strahlkorper {
  using type = ::Strahlkorper<Frame>;
};
}  // namespace QueueTags

/// \ingroup ControlSystemGroup
/// All option tags related to the control system
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Options group for all control system options
struct ControlSystemGroup {
  static std::string name() { return "ControlSystems"; }
  static constexpr Options::String help = {
      "Options for all control systems used in a simulation."};
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Option tag for each individual control system. The name of this option is
/// the name of the \p ControlSystem struct it is templated on. This way all
/// control systems will have a unique name.
template <typename ControlSystem>
struct ControlSystemInputs {
  using type = control_system::OptionHolder<ControlSystem>;
  static constexpr Options::String help{"Options for a control system."};
  static std::string name() { return ControlSystem::name(); }
  using group = ControlSystemGroup;
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Option tag on whether to write data to disk.
struct WriteDataToDisk {
  using type = bool;
  static constexpr Options::String help = {
      "Whether control system data should be saved during an evolution."};
  using group = ControlSystemGroup;
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Option tag that determines how many measurements will occur per control
/// system update.
struct MeasurementsPerUpdate {
  using type = int;
  static constexpr Options::String help = {
      "How many AH measurements are to be done between control system "
      "updates."};
  static int lower_bound() { return 1; }
  using group = ControlSystemGroup;
};
}  // namespace OptionTags

/// \ingroup ControlSystemGroup
/// Alias to get all the option holders from a list of control systems. This is
/// useful in the `option_tags` alias of simple tags for getting all the options
/// from control systems.
template <typename ControlSystems>
using inputs =
    tmpl::transform<ControlSystems,
                    tmpl::bind<OptionTags::ControlSystemInputs, tmpl::_1>>;

/// \ingroup ControlSystemGroup
/// All DataBox tags related to the control system
namespace Tags {
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
template <size_t Dim>
void initialize_tuner(
    const gsl::not_null<::TimescaleTuner*> tuner,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
    const double initial_time, const std::string& name);
}  // namespace detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the timescale tuner
template <typename ControlSystem>
struct TimescaleTuner : db::SimpleTag {
  using type = ::TimescaleTuner;

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
    ::TimescaleTuner tuner = option_holder.tuner;
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

namespace detail {

CREATE_HAS_STATIC_MEMBER_VARIABLE(override_functions_of_time)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(override_functions_of_time)

template <typename Metavariables, typename ControlSystem,
          bool HasOverrideFunctionsOfTime>
struct IsActiveOptionList {
  using type = tmpl::append<
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>,
      tmpl::conditional_t<
          Metavariables::override_functions_of_time,
          tmpl::list<
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>,
          tmpl::list<>>>;
};

template <typename Metavariables, typename ControlSystem>
struct IsActiveOptionList<Metavariables, ControlSystem, false> {
  using type = tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>;
};
}  // namespace detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag to determine if this control system is active.
///
/// This effectively lets us choose control systems at runtime. The OptionHolder
/// has an option for whether the control system is active.
///
/// That option can be overridden if we are overriding the functions of time. If
/// the metavariables specifies `static constexpr bool
/// override_functions_of_time = true`, then this will check the
/// `domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile` option. If the
/// file is defined, it will loop over the map between SpEC and SpECTRE names
/// from `domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap`. If the
/// function of time corresponding to this control system is being overriden
/// with data from the file, then this tag will be `false` so the control system
/// doesn't actually update the function of time.
///
/// If the metavariables doesn't specify `override_functions_of_time`, or it is
/// set to `false`, then this control system is active by default so the tag
/// will be `true`.
template <typename ControlSystem>
struct IsActive : db::SimpleTag {
  using type = bool;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags = typename detail::IsActiveOptionList<
      Metavariables, ControlSystem,
      detail::has_override_functions_of_time_v<Metavariables>>::type;

  template <typename Metavariables>
  static bool create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder,
      const std::optional<std::string>& function_of_time_file,
      const std::map<std::string, std::string>& function_of_time_name_map) {
    if (not function_of_time_file.has_value()) {
      // `None` was specified as the option for the file so we aren't replacing
      // anything
      return option_holder.is_active;
    }

    const std::string& name = ControlSystem::name();

    for (const auto& spec_and_spectre_names : function_of_time_name_map) {
      if (spec_and_spectre_names.second == name) {
        return false;
      }
    }

    return option_holder.is_active;
  }

  template <typename Metavariables>
  static bool create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder) {
    return option_holder.is_active;
  }
};
}  // namespace Tags

/// \ingroup ControlSystemGroup
/// Holds all options for a single control system
///
/// This struct collects all the options for a given control system during
/// option parsing. Then during initialization, the options can be retrieved via
/// their public member names and assigned to their corresponding DataBox tags.
template <typename ControlSystem>
struct OptionHolder {
  static_assert(tt::assert_conforms_to_v<
                ControlSystem, control_system::protocols::ControlSystem>);
  using control_system = ControlSystem;
  static constexpr size_t deriv_order = control_system::deriv_order;
  struct IsActive {
    using type = bool;
    static constexpr Options::String help = {
        "Whether the control system is actually active. If it isn't active, no "
        "measurements (horizon finds) will be done and the functions of time "
        "will never expire."};
  };

  struct Averager {
    using type = ::Averager<deriv_order - 1>;
    static constexpr Options::String help = {
        "Averages the derivatives of the control error and possibly the "
        "control error itself."};
  };

  struct Controller {
    using type = ::Controller<deriv_order>;
    static constexpr Options::String help = {
        "Computes the control signal which will be used to reset the functions "
        "of time."};
  };

  struct TimescaleTuner {
    using type = ::TimescaleTuner;
    static constexpr Options::String help = {
        "Keeps track of the damping timescales for the control system upon "
        "which other timescales are based of off."};
  };

  struct ControlError {
    using type = typename ControlSystem::control_error;
    static constexpr Options::String help = {
        "Computes the control error for the control system based on quantities "
        "in the simulation."};
  };

  using options =
      tmpl::list<IsActive, Averager, Controller, TimescaleTuner, ControlError>;
  static constexpr Options::String help = {"Options for a control system."};

  OptionHolder(const bool input_is_active,
               ::Averager<deriv_order - 1> input_averager,
               ::Controller<deriv_order> input_controller,
               ::TimescaleTuner input_tuner,
               typename ControlSystem::control_error input_control_error)
      : is_active(input_is_active),
        averager(std::move(input_averager)),
        controller(std::move(input_controller)),
        tuner(std::move(input_tuner)),
        control_error(std::move(input_control_error)) {}

  OptionHolder() = default;
  OptionHolder(const OptionHolder& /*rhs*/) = default;
  OptionHolder& operator=(const OptionHolder& /*rhs*/) = default;
  OptionHolder(OptionHolder&& /*rhs*/) = default;
  OptionHolder& operator=(OptionHolder&& /*rhs*/) = default;
  ~OptionHolder() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | is_active;
    p | averager;
    p | controller;
    p | tuner;
    p | control_error;
  };

  // These members are specifically made public for easy access during
  // initialization
  bool is_active{true};
  ::Averager<deriv_order - 1> averager{};
  ::Controller<deriv_order> controller{};
  ::TimescaleTuner tuner{};
  typename ControlSystem::control_error control_error{};
};
}  // namespace control_system
