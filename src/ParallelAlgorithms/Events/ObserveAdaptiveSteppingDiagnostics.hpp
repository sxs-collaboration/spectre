// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct AdaptiveSteppingDiagnostics;
struct Time;
}  // namespace Tags
/// \endcond

namespace Events {
/*!
 * \brief %Observe diagnostics about adaptive time-stepping
 *
 * Writes reduction quantities:
 * - `%Time`
 * - `Number of slabs`
 * - `Number of slab size changes`
 * - `Total steps on all elements`
 * - `Number of LTS step changes`
 * - `Number of step rejections`
 *
 * The slab information is the same on all elements.  The step
 * information is summed over the elements.
 */
class ObserveAdaptiveSteppingDiagnostics : public Event {
 private:
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<uint64_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<uint64_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<uint64_t, funcl::Plus<>>,
      Parallel::ReductionDatum<uint64_t, funcl::Plus<>>,
      Parallel::ReductionDatum<uint64_t, funcl::Plus<>>>;

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveAdaptiveSteppingDiagnostics(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveAdaptiveSteppingDiagnostics);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName>;
  static constexpr Options::String help =
      "Observe diagnostics about adaptive time-stepping\n"
      "\n"
      "Writes reduction quantities:\n"
      " - Time\n"
      " - Number of slabs\n"
      " - Number of slab size changes\n"
      " - Total steps on all elements\n"
      " - Number of LTS step changes\n"
      " - Number of step rejections\n"
      "\n"
      "The slab information is the same on all elements.  The step\n"
      "information is summed over the elements.";

  ObserveAdaptiveSteppingDiagnostics() = default;
  explicit ObserveAdaptiveSteppingDiagnostics(const std::string& subfile_name)
      : subfile_path_("/" + subfile_name) {}

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics>;

  template <typename ArrayIndex, typename ParallelComponent,
            typename Metavariables>
  void operator()(const double time, const AdaptiveSteppingDiagnostics& diags,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const {
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer, observers::ObservationId(time, subfile_path_ + ".dat"),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ArrayIndex>(array_index)},
        subfile_path_,
        std::vector<std::string>{
            "Time", "Number of slabs", "Number of slab size changes",
            "Total steps on all elements", "Number of LTS step changes",
            "Number of step rejections"},
        ReductionData{time, diags.number_of_slabs,
                      diags.number_of_slab_size_changes, diags.number_of_steps,
                      diags.number_of_step_fraction_changes,
                      diags.number_of_step_rejections});
  }

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey(subfile_path_ + ".dat")};
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
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_;
};
}  // namespace Events
