// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {
/// \ingroup EventsAndTriggersGroup
/// Sets the termination flag for the code to exit.  This event is
/// automatically registered.
template <typename EventRegistrars = tmpl::list<>>
class Completion : public Event<EventRegistrars> {
 public:
  /// \cond
  explicit Completion(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Completion);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help = {
    "Sets the termination flag for the code to exit."};

  Completion() = default;

  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::ConstGlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* const /*meta*/) const noexcept {
    auto al_gore =
        Parallel::get_parallel_component<Component>(cache)[array_index]
        .ckLocal();
    al_gore->set_terminate(true);
  }
};

/// \cond
template <typename EventRegistrars>
PUP::able::PUP_ID Completion<EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
