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

template <size_t VolumeDim, bool UseNumericalInitialData>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<
          EvolutionMetavars<VolumeDim, UseNumericalInitialData>> {
  using gh_base = GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<VolumeDim, UseNumericalInitialData>>;
  using typename gh_base::component_list;
  using typename gh_base::const_global_cache_tags;
  using typename gh_base::observed_reduction_data_tags;
  template <typename ParallelComponent>
  using registration_list =
      typename gh_base::template registration_list<ParallelComponent>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"};
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
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
