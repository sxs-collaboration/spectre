// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Actions {
template <size_t Dim>
struct ExportCoordinates {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::Volume,
            observers::ObservationKey("ObserveCoords")};
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, domain::Tags::Mesh<Dim>>> =
          nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const double time = get<Tags::Time>(box);

    const auto& mesh = get<domain::Tags::Mesh<Dim>>(box);
    const auto& inertial_coordinates =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';
    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(Dim + 1);
    for (size_t d = 0; d < Dim; d++) {
      components.emplace_back(element_name + "InertialCoordinates_" +
                                  inertial_coordinates.component_name(
                                      inertial_coordinates.get_tensor_index(d)),
                              inertial_coordinates.get(d));
    }
    // Also output the determinant of the inverse jacobian, which measures
    // the expansion and compression of the grid
    const auto& det_inv_jac =
        db::get<domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(
            box);
    components.emplace_back(
        element_name +
            db::tag_name<domain::Tags::DetInvJacobian<Frame::Logical,
                                                      Frame::Inertial>>(),
        get(det_inv_jac));
    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(time, "ObserveCoords"),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<Dim>>(array_index)),
        std::move(components), mesh.extents(), mesh.basis(), mesh.quadrature());
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool local_time_stepping = false;

  using triggers = Triggers::time_triggers;
  using events = tmpl::list<>;

  using const_global_cache_tags =
      tmpl::list<Tags::TimeStepper<TimeStepper>,
                 Tags::EventsAndTriggers<events, triggers>>;

  static constexpr Options::String help{
      "Export the inertial coordinates of the Domain specified in the input "
      "file. The output can be used to compute initial data externally, for "
      "instance. Also outputs the determinant of the inverse jacobian as a "
      "diagnostic of Domain quality: values far from unity indicate "
      "compression or expansion of the grid."};

  enum class Phase { Initialization, RegisterWithObserver, Export, Exit };

  using component_list = tmpl::list<
      DgElementArray<
          Metavariables,
          tmpl::list<
              Parallel::PhaseActions<
                  typename Metavariables::Phase,
                  Metavariables::Phase::Initialization,
                  tmpl::list<
                      Initialization::Actions::TimeAndTimeStep<Metavariables>,
                      evolution::dg::Initialization::Domain<Dim>,
                      ::Initialization::Actions::
                          RemoveOptionsAndTerminatePhase>>,
              Parallel::PhaseActions<
                  typename Metavariables::Phase,
                  Metavariables::Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 Actions::ExportCoordinates<Dim>>,
                             Parallel::Actions::TerminatePhase>>,
              Parallel::PhaseActions<
                  typename Metavariables::Phase, Metavariables::Phase::Export,
                  tmpl::list<Actions::AdvanceTime,
                             Actions::ExportCoordinates<Dim>,
                             Actions::RunEventsAndTriggers>>>>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>>;

  using observed_reduction_data_tags = tmpl::list<>;

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
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
    &setup_error_handling, &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
