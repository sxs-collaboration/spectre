// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {
namespace Events {
template <typename ObservationValueTag, typename EventRegistrars>
class ObserveTime;

namespace Registrars {
template <typename ObservationValueTag>
using ObserveTime =
    ::Registration::Registrar<Events::ObserveTime, ObservationValueTag>;
}  // namespace Registrars

template <typename ObservationValueTag,
          typename EventRegistrars =
              tmpl::list<Registrars::ObserveTime<ObservationValueTag>>>
class ObserveTime;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Prints the time of observation in the simulation.
 *
 * Prints reduction quantities:
 * - `ObservationValueTag`
 *
 * \warning Currently, only one reduction observation event can be
 * triggered at a given observation value.  Causing multiple events to run at
 * once will produce unpredictable results.
 */
template <typename ObservationValueTag, typename EventRegistrars>
class ObserveTime : public Event<EventRegistrars> {
 private:
  using ReductionData = Parallel::ReductionData<Parallel::ReductionDatum<
      std::string, funcl::AssertEqual<>>>;  // may want to assertequal on
                                            // observation value

 public:
  /// Unique string tag to identify for this print reduction observation
  struct PrintTag {
    using type = std::string;
    static constexpr Options::String help = {
        "Unique string that tags the print reduction observation."};
  };

  /// \cond
  explicit ObserveTime(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTime);  // NOLINT
  /// \endcond

  using options = tmpl::list<PrintTag>;
  static constexpr Options::String help =
      "Prints the time of observation in the simulation.\n"
      "\n"
      "Writes reduction quantities:\n"
      " * ObservationValueTag";

  ObserveTime() = default;
  explicit ObserveTime(const std::string& print_tag) noexcept;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags = tmpl::list<ObservationValueTag>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const noexcept {
    // Send data to reduction observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();

    std::string info_to_print =
        "Current time: " +
        std::to_string(static_cast<double>(observation_value));

    Parallel::simple_action<observers::Actions::ContributeReductionData<
        observers::ThreadedActions::PrintReductionData>>(
        local_observer, observers::ObservationId(observation_value, print_tag_),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ArrayIndex>(array_index)},
        print_tag_,  // should NOT be used
        std::vector<std::string>{"StringToPrint"},
        ReductionData{info_to_print});
  }

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey(print_tag_)};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event<EventRegistrars>::pup(p);
    p | print_tag_;
  }

 private:
  std::string print_tag_;
};

template <typename ObservationValueTag, typename EventRegistrars>
ObserveTime<ObservationValueTag, EventRegistrars>::ObserveTime(
    const std::string& print_tag) noexcept
    : print_tag_(print_tag) {}

/// \cond
template <typename ObservationValueTag, typename EventRegistrars>
PUP::able::PUP_ID ObserveTime<ObservationValueTag, EventRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
