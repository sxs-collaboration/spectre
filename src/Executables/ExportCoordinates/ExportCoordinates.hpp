// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

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
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Time.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Actions {
template <size_t Dim>
struct InitializeElement {
  struct InitialExtents : db::SimpleTag {
    static std::string name() noexcept { return "InitialExtents"; }
    using type = std::vector<std::array<size_t, Dim>>;
  };
  struct Domain : db::SimpleTag {
    static std::string name() noexcept { return "Domain"; }
    using type = ::Domain<Dim, Frame::Inertial>;
  };

  using AddOptionsToDataBox =
      Parallel::ForwardAllOptionsToDataBox<tmpl::list<InitialExtents, Domain>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTagsList, Domain>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto initial_extents = db::get<InitialExtents>(box);
    ::Domain<Dim, Frame::Inertial> domain{};
    db::mutate<Domain>(
        make_not_null(&box), [&domain](const auto domain_ptr) noexcept {
          domain = std::move(*domain_ptr);
        });
    return std::make_tuple(
        Elliptic::Initialization::Domain<Dim>::initialize(
            db::create_from<typename AddOptionsToDataBox::simple_tags>(
                std::move(box)),
            array_index, initial_extents, domain),
        true);
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<not tmpl::list_contains_v<DbTagsList, Domain>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), true};
  }
};

template <size_t Dim>
struct ExportCoordinates {
  // Compile-time interface for observers
  struct ObservationType {};
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::ReductionAndVolume,
            observers::ObservationId(0., ObservationType{})};
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::Mesh<Dim>>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
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
    return {std::move(box), true};
  }
};
}  // namespace Actions

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<OptionTags::DomainCreator<Dim, Frame::Inertial>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::InitializeElement<Dim>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterWithObservers<
                         Actions::ExportCoordinates<Dim>>,
                     Parallel::Actions::TerminatePhase>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Export,
                             tmpl::list<Actions::ExportCoordinates<Dim>>>>;

  using add_options_to_databox =
      typename Actions::InitializeElement<Dim>::AddOptionsToDataBox;

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
            .insert(global_cache,
                    {domain_creator->initial_extents(),
                     domain_creator->create_domain()},
                    which_proc);
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
    }
    element_array.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);
    element_array.start_phase(next_phase);
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
/// \endcond
