// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

/// \cond
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

  template <typename Metavariables>
  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>,
                 domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;
  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  static type create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder,
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    const auto domain = domain_creator->create_domain();
    const auto& excision_spheres = domain.excision_spheres();

    constexpr size_t expected_number_of_excisions =
        type::expected_number_of_excisions;

    const auto print_error =
        [&excision_spheres](const std::string& excision_sphere_name) {
          ERROR_NO_TRACE(
              "The control error for the"
              << pretty_type::name<type>()
              << " control system expected there to be at least one excision "
                 "sphere named '"
              << excision_sphere_name
              << "' in the domain, but there wasn't. The existing excision "
                 "spheres are: "
              << keys_of(excision_spheres)
              << ". Check that the domain you have chosen has excision spheres "
                 "implemented.");
        };

    if constexpr (expected_number_of_excisions == 1) {
      if (excision_spheres.count("ExcisionSphereA") != 1 and
          excision_spheres.count("ExcisionSphereB") != 1 and
          excision_spheres.count("ExcisionSphere") != 1) {
        print_error("ExcisionSphereA' or 'ExcisionSphereB' or 'ExcisionSphere");
      }
    }
    if constexpr (expected_number_of_excisions == 2) {
      if (excision_spheres.count("ExcisionSphereA") != 1) {
        print_error("ExcisionSphereA");
      }
      if (excision_spheres.count("ExcisionSphereB") != 1) {
        print_error("ExcisionSphereB");
      }
    }

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
}  // namespace control_system::Tags
