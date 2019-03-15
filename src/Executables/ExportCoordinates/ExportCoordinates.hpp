// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Time.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ObservationType {};
}  // namespace

namespace Actions {

template <size_t Dim>
struct InitializeElement {
  using return_tag_list = tmpl::append<
      typename Elliptic::Initialization::Domain<Dim>::simple_tags,
      typename Elliptic::Initialization::Domain<Dim>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain) noexcept {
    auto domain_box = Elliptic::Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    return std::make_tuple(std::move(domain_box));
  }
};

template <size_t Dim>
struct ExportCoordinates {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const auto& mesh = get<Tags::Mesh<Dim>>(box);
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';
    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(Dim);
    for (size_t d = 0; d < Dim; d++) {
      components.emplace_back(element_name + "InertialCoordinates_" +
                                  inertial_coordinates.component_name(
                                      inertial_coordinates.get_tensor_index(d)),
                              inertial_coordinates.get(d));
    }
    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(0., ObservationType{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());
  }
};
}  // namespace Actions

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<OptionTags::DomainCreator<Dim, Frame::Inertial>>;
  using initial_databox = db::compute_databox_type<
      typename Actions::InitializeElement<Dim>::return_tag_list>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const std::unique_ptr<DomainCreator<Dim, Frame::Inertial>>
          domain_creator) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);

    auto domain = domain_creator->create_domain();
    for (const auto& block : domain.blocks()) {
      const auto initial_ref_levs =
          domain_creator->initial_refinement_levels()[block.id()];
      const std::vector<ElementId<Dim>> element_ids =
          initial_element_ids(block.id(), initial_ref_levs);
      int which_proc = 0;
      const int number_of_procs = Parallel::number_of_procs();
      for (size_t i = 0; i < element_ids.size(); ++i) {
        element_array(ElementIndex<Dim>(element_ids[i]))
            .insert(global_cache, which_proc);
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
    }
    element_array.doneInserting();

    element_array.template simple_action<Actions::InitializeElement<Dim>>(
        std::make_tuple(domain_creator->initial_extents(), std::move(domain)));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);
    switch (next_phase) {
      case Metavariables::Phase::RegisterWithObserver:
        Parallel::simple_action<observers::Actions::RegisterWithObservers<
            observers::TypeOfObservation::Volume>>(
            element_array, observers::ObservationId(0., ObservationType{}));
        break;
      case Metavariables::Phase::Export:
        element_array.template simple_action<Actions::ExportCoordinates<Dim>>();
        break;
      default:
        break;
    }
  }
};

template <size_t Dim>
struct Metavariables {
  static constexpr OptionString help{
      "Export the inertial coordinates of the Domain specified in the input "
      "file. The output can be used to compute initial data externally, for "
      "instance."};

  using const_global_cache_tag_list = tmpl::list<>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>,
                                    observers::Observer<Metavariables>,
                                    observers::ObserverWriter<Metavariables>>;
  using observed_reduction_data_tags = tmpl::list<>;

  enum class Phase { Initialization, RegisterWithObserver, Export, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Export;
      case Phase::Export:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR("Unknown type of phase.");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
