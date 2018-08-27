// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <vector>

#include "AlgorithmArray.hpp"                            // IWYU pragma: keep
#include "AlgorithmGroup.hpp"                            // IWYU pragma: keep
#include "AlgorithmSingleton.hpp"                        // IWYU pragma: keep
#include "ApparentHorizons/HorizonComponent.hpp"         // IWYU pragma: keep
#include "ApparentHorizons/HorizonManagerComponent.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/Strahlkorper.hpp"             // IWYU pragma: keep
#include "DataStructures/Tensor/IndexType.hpp"           // IWYU pragma: keep
#include "Domain/DomainCreators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Time.hpp"                                 // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ApparentHorizons/TestHorizonManagerHelper_ElementActions.hpp"  // IWYU pragma: keep
#include "tests/Unit/ApparentHorizons/TestHorizonManagerHelper_ElementComponent.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare DgElementArray
// IWYU pragma: no_forward_declare Frame::Inertial
// IWYU pragma: no_forward_declare HorizonComponent
// IWYU pragma: no_forward_declare HorizonManagerComponent

// This is a function that each ah::Finder will call when it
// converges.
template <typename Metavariables, typename AhTag, typename Frame>
struct TestFunction {
  static void apply(
      const Strahlkorper<Frame>& strahlkorper, const Time& timestep,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
    Parallel::printf(
        "### Node:%d  Proc:%d ###\n"
        "{%s, time %s}: Radius is %f\n\n",
        Parallel::my_node(), Parallel::my_proc(), AhTag::label(), timestep,
        strahlkorper.average_radius());
    SPECTRE_PARALLEL_REQUIRE(strahlkorper.average_radius() < 2.01);
    SPECTRE_PARALLEL_REQUIRE(strahlkorper.average_radius() > 1.99);
  }
};

struct TestMetavariables {
  // Tags for horizons.
  struct AhATag {
    using frame = Frame::Inertial;
    static std::string label() noexcept { return "AhA"; }
    using option_tag = ah::OptionTags::AhA<frame>;
    using convergence_hook = TestFunction<TestMetavariables, AhATag, frame>;
  };
  struct AhBTag {
    using frame = Frame::Inertial;
    static std::string label() noexcept { return "AhB"; }
    using option_tag = ah::OptionTags::AhB<frame>;
    using convergence_hook = TestFunction<TestMetavariables, AhBTag, frame>;
  };
  using horizon_tags = tmpl::list<AhATag, AhBTag>;

  using component_list = tmpl::list<ah::DataInterpolator<TestMetavariables>,
                                    test_ah::DgElementArray<TestMetavariables>,
                                    ah::Finder<TestMetavariables, AhATag>,
                                    ah::Finder<TestMetavariables, AhBTag>>;

  static constexpr const char* const help{"Test HorizonManager in parallel"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  using domain_creator_tag = OptionTags::DomainCreator<3, Frame::Inertial>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase {
    Initialization,
    InitialCommunication,
    CheckAnswer,
    BeginHorizonSearch,
    Exit
  };
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    if (current_phase == Phase::Initialization) {
      return Phase::InitialCommunication;
    }
    if (current_phase == Phase::InitialCommunication) {
      return Phase::CheckAnswer;
    }
    if (current_phase == Phase::CheckAnswer) {
      return Phase::BeginHorizonSearch;
    }
    return Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &DomainCreators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.cpp"
