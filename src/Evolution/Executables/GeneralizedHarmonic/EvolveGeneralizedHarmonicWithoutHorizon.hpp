// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

// Second template parameter specifies the source of the initial data, which
// could be an analytic solution, analytic data, or imported numerical data.
// Third template parameter specifies the analytic solution used when imposing
// dirichlet boundary conditions or against which to compute error norms.
template <size_t VolumeDim, typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<
          false,
          EvolutionMetavars<VolumeDim, InitialData, BoundaryConditions>> {
  using gh_base = GeneralizedHarmonicTemplateBase<
      false, EvolutionMetavars<VolumeDim, InitialData, BoundaryConditions>>;
  using typename gh_base::component_list;
  using typename gh_base::frame;
  using typename gh_base::observed_reduction_data_tags;
  using typename gh_base::Phase;

  using phase_selection = typename gh_base::template PhaseSelection<
      EvolutionMetavars, typename gh_base::gh_dg_element_array,
      typename gh_base::dg_registration_list>;

  using const_global_cache_tags = tmpl::list<
      typename gh_base::analytic_solution_tag, Tags::EventsAndTriggers,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          VolumeDim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          VolumeDim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          VolumeDim, Frame::Grid>,
      PhaseControl::Tags::PhaseChangeAndTriggers<
          typename phase_selection::phase_changes>>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"};

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
          cache_proxy) noexcept {
    const auto next_phase = PhaseControl::arbitrate_phase_change<
        typename phase_selection::phase_changes>(
        phase_change_decision_data, current_phase,
        *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
    switch (current_phase) {
      case Phase::Initialization:
        return evolution::is_numeric_initial_data_v<InitialData>
                   ? Phase::RegisterWithElementDataReader
                   : Phase::InitializeInitialDataDependentQuantities;
      case Phase::RegisterWithElementDataReader:
        return Phase::ImportInitialData;
      case Phase::ImportInitialData:
        return Phase::InitializeInitialDataDependentQuantities;
      case Phase::InitializeInitialDataDependentQuantities:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
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
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_selection::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
