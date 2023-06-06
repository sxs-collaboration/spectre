// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/MemoryMonitor/Tags.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessArray.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessGroups.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessSingleton.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel::Algorithms {
struct Array;
struct Group;
struct Nodegroup;
struct Singleton;
}  // namespace Parallel::Algorithms
/// \endcond

namespace Events {
/*!
 * \brief Event run on the DgElementArray that will monitor the memory usage of
 * parallel components in megabytes.
 *
 * \details Given a list of parallel component names from Options, this will
 * calculate the memory usage of each component and write it to disk in the
 * reductions file under the `/MemoryMonitors/` group. The name of each file is
 * the `pretty_type::name` of each parallel component.
 *
 * The parallel components available to monitor are the ones defined in the
 * `component_list` type alias in the metavariables. In addition to these
 * components, you can also monitor the size of the GlobalCache. Currently, you
 * cannot monitor the size of the MutableGlobalCache. To see which parallel
 * components are available to monitor, request to monitor an invalid parallel
 * component ("Blah" for example) in the input file. An ERROR will occur and a
 * list of the available components to monitor will be printed.
 *
 * \note Currently, the only Parallel::Algorithms::Array parallel component that
 * can be monitored is the DgElementArray itself.
 */

template <size_t Dim, typename ObservationValueTag>
class MonitorMemory : public Event {
 private:
  // Reduction data for arrays
  using ReductionData = Parallel::ReductionData<
      // Time
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Vector of total mem usage on each node
      Parallel::ReductionDatum<std::vector<double>,
                               funcl::ElementWise<funcl::Plus<>>>>;

 public:
  explicit MonitorMemory(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MonitorMemory);  // NOLINT

  struct ComponentsToMonitor {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "Names of parallel components to monitor the memory usage of. If you'd "
        "like to monitor all available parallel components, pass 'All' "
        "instead."};
  };

  using options = tmpl::list<ComponentsToMonitor>;

  static constexpr Options::String help =
      "Observe memory usage of parallel components.";

  MonitorMemory() = default;

  template <typename Metavariables>
  MonitorMemory(
      const std::optional<std::vector<std::string>>& components_to_monitor,
      const Options::Context& context, Metavariables /*meta*/);

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags =
      tmpl::list<ObservationValueTag, domain::Tags::Element<Dim>>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const ::Element<Dim>& element,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const;

  using observation_registration_tags = tmpl::list<>;

  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration() const {
    return {};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::unordered_set<std::string> components_to_monitor_{};
};

/// \cond
template <size_t Dim, typename ObservationValueTag>
MonitorMemory<Dim, ObservationValueTag>::MonitorMemory(CkMigrateMessage* msg)
    : Event(msg) {}

template <size_t Dim, typename ObservationValueTag>
template <typename Metavariables>
MonitorMemory<Dim, ObservationValueTag>::MonitorMemory(
    const std::optional<std::vector<std::string>>& components_to_monitor,
    const Options::Context& context, Metavariables /*meta*/) {
  using component_list = tmpl::push_back<typename Metavariables::component_list,
                                         Parallel::GlobalCache<Metavariables>>;
  std::unordered_map<std::string, std::string> existing_components{};
  std::string str_component_list{};

  tmpl::for_each<component_list>(
      [&existing_components, &str_component_list](auto component_v) {
        using component = tmpl::type_from<decltype(component_v)>;
        const std::string component_name = pretty_type::name<component>();
        const std::string chare_type_name =
            pretty_type::name<typename component::chare_type>();
        existing_components[component_name] = chare_type_name;
        // Only Array we can monitor is DgElementArray
        if (chare_type_name != "Array") {
          str_component_list += " - " + component_name + "\n";
        } else if (component_name == "DgElementArray") {
          str_component_list += " - " + component_name + "\n";
        }
      });

  // A list of names was specified
  if (components_to_monitor.has_value()) {
    for (const auto& component : *components_to_monitor) {
      // Do some checks:
      //  1. Make sure the components requested are viable components. This
      //     protects against spelling errors.
      //  2. Currently the only charm Array you can monitor the memory of is
      //     the DgElementArray so enforce this.
      if (existing_components.count(component) != 1) {
        PARSE_ERROR(
            context,
            "Cannot monitor memory usage of unknown parallel component '"
                << component
                << "'. Please choose from the existing parallel components:\n"
                << str_component_list);
      } else if (existing_components.at(component) == "Array" and
                 component != "DgElementArray") {
        PARSE_ERROR(
            context,
            "Cannot monitor the '"
                << component
                << "' parallel component. Currently, the only Array parallel "
                   "component allowed to be monitored is the "
                   "DgElementArray.");
      }

      components_to_monitor_.insert(component);
    }
  } else {
    // 'All' was specified. Filter out Array components that are not the
    // DgElementArray
    for (const auto& [name, chare] : existing_components) {
      if (chare != "Array") {
        components_to_monitor_.insert(name);
      } else if (name == "DgElementArray") {
        components_to_monitor_.insert(name);
      }
    }
  }
}

template <size_t Dim, typename ObservationValueTag>
template <typename Metavariables, typename ArrayIndex,
          typename ParallelComponent>
void MonitorMemory<Dim, ObservationValueTag>::operator()(
    const typename ObservationValueTag::type& observation_value,
    const ::Element<Dim>& element, Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index,
    const ParallelComponent* const /*meta*/) const {
  using component_list = tmpl::push_back<typename Metavariables::component_list,
                                         Parallel::GlobalCache<Metavariables>>;

  tmpl::for_each<component_list>([this, &observation_value, &element, &cache,
                                  &array_index](auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;

    // If we aren't monitoring this parallel component, then just exit now
    if (components_to_monitor_.count(pretty_type::name<component>()) != 1) {
      return;
    }

    // Certain components only need to be triggered once, so we have a special
    // element designated to be the one that triggers memory monitoring, the
    // 0th element.
    const auto& element_id = element.id();
    // Avoid GCC-7 compiler warning about unused variable (in the Array if
    // constexpr branch)
    [[maybe_unused]] const bool designated_element =
        is_zeroth_element(element_id);

    // If this is an array, this is run on every element. It has already
    // been asserted in the constructor that the only Array the MemoryMonitor
    // can monitor is the DgElementArray itself. If you want to monitor other
    // Arrays, the implementation will need to be generalized.
    if constexpr (Parallel::is_array_v<component>) {
      auto& memory_monitor_proxy = Parallel::get_parallel_component<
          mem_monitor::MemoryMonitor<Metavariables>>(cache);
      auto array_element_proxy =
          Parallel::get_parallel_component<component>(cache)[array_index];
      const double size_in_bytes = static_cast<double>(
          size_of_object_in_bytes(*Parallel::local(array_element_proxy)));
      const double size_in_megabytes = size_in_bytes / 1.0e6;

      // vector the size of the number of nodes we are running on. Set the
      // 'my_node'th element of the vector to the size of this Element. Then
      // when we reduce, we will have a vector with 'num_nodes' elements, each
      // of which represents the total memory usage of all Elements on that
      // node.
      const size_t num_nodes = Parallel::number_of_nodes<size_t>(
          *Parallel::local(array_element_proxy));
      const size_t my_node =
          Parallel::my_node<size_t>(*Parallel::local(array_element_proxy));
      std::vector<double> data(num_nodes, 0.0);
      data[my_node] = size_in_megabytes;

      Parallel::contribute_to_reduction<
          mem_monitor::ProcessArray<ParallelComponent>>(
          ReductionData{static_cast<double>(observation_value), data},
          array_element_proxy, memory_monitor_proxy);
    } else if constexpr (Parallel::is_singleton_v<component>) {
      // If this is a singleton, we only run this once so use the designated
      // element. Nothing to reduce with singletons so just call the simple
      // action on the singleton
      if (designated_element) {
        auto& singleton_proxy =
            Parallel::get_parallel_component<component>(cache);

        Parallel::simple_action<mem_monitor::ProcessSingleton>(
            singleton_proxy, static_cast<double>(observation_value));
      }
    } else if constexpr (Parallel::is_nodegroup_v<component> or
                         Parallel::is_group_v<component>) {
      // If this is a (node)group, call a simple action on each branch if on
      // the designated element
      if (designated_element) {
        // Can't run simple actions on the cache so broadcast a specific entry
        // method that will calculate the size and send it to the memory
        // monitor
        if constexpr (std::is_same_v<component,
                                     Parallel::GlobalCache<Metavariables>>) {
          auto cache_proxy = cache.get_this_proxy();

          // This will be called on all branches of the GlobalCache
          cache_proxy.compute_size_for_memory_monitor(
              static_cast<double>(observation_value));
        } else if constexpr (std::is_same_v<
                                 component,
                                 Parallel::MutableGlobalCache<Metavariables>>) {
          // Currently, this branch is unreachable because the
          // MutableGlobalCache is not in `component_list` above. We keep the
          // branch in anyways for consistency and because we anticipate to be
          // able to monitor the MutableGlobalCache in the future.

          // Can't run simple actions on the mutable cache so broadcast a
          // specific entry method that will calculate the size and send it to
          // the memory monitor
          auto mutable_global_cache_proxy = cache.mutable_global_cache_proxy();

          // This will be called on all branches of the MutableGlobalCache
          mutable_global_cache_proxy.compute_size_for_memory_monitor(
              cache.get_this_proxy(), static_cast<double>(observation_value));
        } else {
          // Groups and nodegroups share an action
          auto& group_proxy =
              Parallel::get_parallel_component<component>(cache);

          // This will be called on all branches of the (node)group
          Parallel::simple_action<mem_monitor::ProcessGroups>(
              group_proxy, static_cast<double>(observation_value));
        }
      }
    }
  });
}

template <size_t Dim, typename ObservationValueTag>
void MonitorMemory<Dim, ObservationValueTag>::pup(PUP::er& p) {
  Event::pup(p);
  p | components_to_monitor_;
}

template <size_t Dim, typename ObservationValueTag>
PUP::able::PUP_ID MonitorMemory<Dim, ObservationValueTag>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Events
