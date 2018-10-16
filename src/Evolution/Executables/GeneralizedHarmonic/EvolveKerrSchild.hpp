// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <sstream>
#include <vector>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/FinalTime.hpp"              // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename Tag>
struct dt;
template <class>
class Variables;
}  // namespace Tags
/// \endcond

namespace Actions {
struct Initialize {
  template <typename System>
  struct GrTags {
    using gr_variables_tag = ::Tags::Variables<
        tmpl::list<gr::Tags::SpacetimeMetric<3>, GeneralizedHarmonic::Pi<3>,
                   GeneralizedHarmonic::Phi<3>>>;
    using GrVars = gr_variables_tag::type;
    using simple_tags = db::AddSimpleTags<gr_variables_tag>;
    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      const size_t num_grid_points =
          db::get<Tags::Mesh<3>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<Tags::Coordinates<3, Frame::Inertial>>(box);

      // Set initial data from analytic solution
      using solution_tag = OptionTags::AnalyticSolutionBase;
      const auto& solution_vars = Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time,
          typename gr::Solutions::KerrSchild::template tags<DataVector>{});

      using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                       tmpl::size_t<3>, Frame::Inertial>;
      using DerivShift =
          ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>;
      using DerivSpatialMetric =
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>;

      const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
      const auto& dt_lapse =
          get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
      const auto& deriv_lapse = get<DerivLapse>(solution_vars);

      const auto& shift =
          get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(solution_vars);
      const auto& dt_shift =
          get<::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
              solution_vars);
      const auto& deriv_shift = get<DerivShift>(solution_vars);

      const auto& spatial_metric =
          get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
              solution_vars);
      const auto& dt_spatial_metric = get<
          ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          solution_vars);
      const auto& deriv_spatial_metric = get<DerivSpatialMetric>(solution_vars);

      const auto& spacetime_metric =
          gr::spacetime_metric(lapse, shift, spatial_metric);
      const auto& phi =
          GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric);
      const auto& pi =
          GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift,
                                  spatial_metric, dt_spatial_metric, phi);
      const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<3>,
                                GeneralizedHarmonic::Pi<3>,
                                GeneralizedHarmonic::Phi<3>>
          solution_tuple(spacetime_metric, pi, phi);

      GrVars gr_vars{num_grid_points};
      gr_vars.assign_subset(solution_tuple);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(gr_vars));
    }
  };

  // Add quantities for dealing with time
  struct Evolution {
    using simple_tags =
        db::AddSimpleTags<Tags::TimeId, Tags::Next<Tags::TimeId>,
                          Tags::TimeStep>;
    using compute_tags = db::AddComputeTags<Tags::Time>;

    // Global time stepping initial time step
    template <typename Metavariables,
              Requires<not Metavariables::local_time_stepping> = nullptr>
    static TimeDelta get_initial_time_step(
        const Time& initial_time, const double initial_dt_value,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
      return (initial_dt_value > 0.0 ? 1 : -1) * initial_time.slab().duration();
    }

    // Local time stepping initial time step
    template <typename Metavariables,
              Requires<Metavariables::local_time_stepping> = nullptr>
    static TimeDelta get_initial_time_step(
        const Time& initial_time, const double initial_dt_value,
        const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
      const auto& step_controller =
          Parallel::get<OptionTags::StepController>(cache);
      return step_controller.choose_step(initial_time, initial_dt_value);
    }

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time_value, const double initial_dt_value,
        const double initial_slab_size) noexcept {
      const bool time_runs_forward = initial_dt_value > 0.0;
      const Slab initial_slab =
          time_runs_forward ? Slab::with_duration_from_start(initial_time_value,
                                                             initial_slab_size)
                            : Slab::with_duration_to_end(initial_time_value,
                                                         initial_slab_size);
      const Time initial_time =
          time_runs_forward ? initial_slab.start() : initial_slab.end();
      const TimeDelta initial_dt =
          get_initial_time_step(initial_time, initial_dt_value, cache);

      const auto& time_stepper =
          Parallel::get<OptionTags::TypedTimeStepper<TimeStepper>>(cache);
      const TimeId time_id(
          time_runs_forward,
          -static_cast<int64_t>(time_stepper.number_of_past_steps()),
          initial_time);
      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), TimeId{}, time_id, initial_dt);
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename Initialization::Domain<3>::simple_tags,
      typename Initialization::Domain<3>::compute_tags,
      typename GrTags<typename Metavariables::system>::simple_tags,
      typename GrTags<typename Metavariables::system>::compute_tags,
      typename Evolution::simple_tags, typename Evolution::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<3>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, 3>> initial_extents,
                    Domain<3, Frame::Inertial> domain,
                    const double initial_time, const double initial_dt,
                    const double initial_slab_size) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = Initialization::Domain<3>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto gr_box =
        GrTags<system>::initialize(std::move(domain_box), cache, initial_time);
    auto time_box = Evolution::initialize(
        std::move(gr_box), cache, initial_time, initial_dt, initial_slab_size);

    return std::make_tuple(std::move(time_box));
  }
};

struct Observe {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Get the inertial-frame coordinates of the current element's points
    // from databox box
    const auto& inertial_coordinates =
        db::get<Tags::MappedCoordinates<Tags::ElementMap<3, Frame::Inertial>,
                                        Tags::LogicalCoordinates<3>>>(box);
    const auto& time = db::get<Tags::Time>(box);

    // Get the Kerr-Schild solution at the current time
    using solution_tag = OptionTags::AnalyticSolutionBase;
    const auto& variables = Parallel::get<solution_tag>(cache).variables(
        inertial_coordinates, time.value(),
        typename gr::Solutions::KerrSchild::template tags<DataVector>{});

    // Get the current element_id, and make a name for writing to disk
    const ElementId<3> element_id{array_index};
    const std::string element_name = MakeString{} << element_id << '/';

    // Get the lapse, for writing to disk
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(variables);

    // Compute the Ricci scalar with a numerical derivative of the
    // Christoffel symbols
    const auto& deriv_spatial_metric =
        get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(variables);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(variables);
    const auto& inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto& christoffel_second_kind = raise_or_lower_first_index(
        gr::christoffel_first_kind(deriv_spatial_metric),
        inverse_spatial_metric);
    // Create a Variables
    const tuples::TaggedTuple<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>>
        christoffel_second_kind_tuple(christoffel_second_kind);
    Variables<tmpl::list<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>>>
        christoffel_second_kind_variables(
            get<0, 0, 0>(christoffel_second_kind).size());
    // Load in christoffel_second_kind
    get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>>(
        christoffel_second_kind_variables) = christoffel_second_kind;
    // Get InverseJacobian and the mesh
    const auto& inverse_jacobian =
        db::get<Tags::InverseJacobian<Tags::ElementMap<3, Frame::Inertial>,
                                      Tags::LogicalCoordinates<3>>>(box);
    const auto& mesh = db::get<Tags::Mesh<3>>(box);
    // Take numerical derivative
    const auto& deriv_christoffel_second_kind_variables =
        partial_derivatives<tmpl::list<gr::Tags::SpatialChristoffelSecondKind<
            3, Frame::Inertial, DataVector>>>(christoffel_second_kind_variables,
                                              mesh, inverse_jacobian);
    const auto& d_christoffel_second_kind = get<Tags::deriv<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
        tmpl::size_t<3>, Frame::Inertial>>(
        deriv_christoffel_second_kind_variables);
    // Get the ricci scalar
    const auto& spatial_ricci_scalar = trace(
        gr::ricci_tensor(christoffel_second_kind, d_christoffel_second_kind),
        inverse_spatial_metric);

    // Prepare the ingredients for writing volume data
    const auto& extents = db::get<Tags::Mesh<3>>(box).extents();

    // Tensor or scalar components, i.e. DataVectors, are stored in a
    // std::vector<TensorComponent>
    std::vector<TensorComponent> components;

    // Reserve 3 components for the x,y,z inertial coordinates
    // and 1 component for the lapse
    components.reserve(5);
    components.emplace_back(element_name + "InertialCoordinates_x",
                            get<0>(inertial_coordinates));
    components.emplace_back(element_name + "InertialCoordinates_y",
                            get<1>(inertial_coordinates));
    components.emplace_back(element_name + "InertialCoordinates_z",
                            get<2>(inertial_coordinates));
    components.emplace_back(element_name + "Lapse", get(lapse));
    components.emplace_back(element_name + "SpatialRicciScalar",
                            get(spatial_ricci_scalar));

    // Send the volume data for writing to disk
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(time),
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<3>>(array_index)),
        std::move(components), extents);

    // Required for iterable observers
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions

struct EvolutionMetavars {
  // For now, set up a scalar wave system, even though we won't use it
  // for any output, because this lets us use DgElementArray.
  // Customization/"input options" to simulation
  using system = GeneralizedHarmonic::System<3>;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<gr::Solutions::KerrSchild>;

  // Set up the ingredients for the simulation that we will use
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using domain_creator_tag = OptionTags::DomainCreator<3, Frame::Inertial>;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<TimeStepper>>;

  // Set up observers
  using Redum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                         funcl::Sqrt<funcl::Divides<>>,
                                         std::index_sequence<1>>;
  using reduction_data_tags = tmpl::list<observers::Tags::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>, Redum, Redum>>;

  // The components I will use: observers and DgElementArray
  // Note that Action::AdvanceTime and Action::FinalTime just increment
  // the TimeId; they don't actually call the time stepper.
  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<EvolutionMetavars, Actions::Initialize,
                     tmpl::list<Actions::AdvanceTime, Actions::Observe,
                                Actions::FinalTime>>>;

  static constexpr OptionString help{
      "'Evolve' a Kerr Schild black hole: evaluate the analytic\n"
      "solution.\n\n"};

  enum class Phase { Initialization, RegisterWithObserver, Evolve, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &DomainCreators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
