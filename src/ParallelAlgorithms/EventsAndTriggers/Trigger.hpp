// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EventsAndTriggersGroup
namespace Triggers {}

/// \ingroup EventsAndTriggersGroup
/// Base class for checking whether to run an Event.
class Trigger : public PUP::able {
 protected:
  /// \cond
  Trigger() = default;
  Trigger(const Trigger&) = default;
  Trigger(Trigger&&) = default;
  Trigger& operator=(const Trigger&) = default;
  Trigger& operator=(Trigger&&) = default;
  /// \endcond

 public:
  ~Trigger() override = default;

  WRAPPED_PUPable_abstract(Trigger);  // NOLINT

  template <typename DbTags>
  bool is_triggered(const db::DataBox<DbTags>& box) const {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    return call_with_dynamic_type<bool, tmpl::at<factory_classes, Trigger>>(
        this, [&box](auto* const trigger) { return db::apply(*trigger, box); });
  }
};
