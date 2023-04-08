// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Executables/Examples/RandomAmr/InitializeDomain.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct DummySystem {};
}  // namespace

/// \page RandomAmrExecutablePage RandomAmr Executable
/// The RandomAmr executable is being used to develop the mechanics of
/// adaptive mesh refinement.
///
/// See RandomAmrMetavars for a description of the metavariables of this
/// executable.

/// \brief The metavariables for the RandomAmr executable
template <size_t Dim>
struct RandomAmrMetavars {
  static constexpr size_t volume_dim = Dim;
  using system = DummySystem;

  static constexpr Options::String help{
      "Test anisotropic refinement by randomly refining a grid.\n"};

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>>;
  };

  using const_global_cache_tags = tmpl::list<>;

  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization, Parallel::Phase::CheckDomain,
                 Parallel::Phase::Exit};

  using amr_element_component = DgElementArray<
      RandomAmrMetavars,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<Initialization::Actions::InitializeItems<
                             amr::Initialization::Domain<volume_dim>,
                             amr::Initialization::Initialize<volume_dim>>,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::CheckDomain,
              tmpl::list<::amr::Actions::SendAmrDiagnostics,
                         Parallel::Actions::TerminatePhase>>>>;

  using component_list =
      tmpl::list<amr::Component<RandomAmrMetavars>, amr_element_component>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
