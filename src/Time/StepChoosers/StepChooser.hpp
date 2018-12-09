// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

/// \ingroup TimeSteppersGroup
///
/// Holds all the StepChoosers
namespace StepChoosers {
/// Holds all the StepChooser registrars
///
/// These can be passed in a list to the template argument of
/// StepChooser to choose which StepChoosers can be constructed.
namespace Registrars {}
}  // namespace StepChoosers

/// \ingroup TimeSteppersGroup
///
/// StepChoosers suggest upper bounds on step sizes.  Concrete
/// StepChoosers should define operator() returning the magnitude of
/// the desired step (as a double).
///
/// The step choosers valid for the integration being controlled are
/// specified by passing a `tmpl::list` of the corresponding
/// registrars.
template <typename StepChooserRegistrars>
class StepChooser : public PUP::able {
 protected:
  /// \cond HIDDEN_SYMBOLS
  StepChooser() = default;
  StepChooser(const StepChooser&) = default;
  StepChooser(StepChooser&&) = default;
  StepChooser& operator=(const StepChooser&) = default;
  StepChooser& operator=(StepChooser&&) = default;
  /// \endcond

 public:
  ~StepChooser() override = default;

  WRAPPED_PUPable_abstract(StepChooser);  // NOLINT

  using creatable_classes = Registration::registrants<StepChooserRegistrars>;

  template <typename Metavariables, typename DbTags>
  double desired_step(
      const db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) const noexcept {
    const auto result = call_with_dynamic_type<double, creatable_classes>(
        this, [&box, &cache](const auto* const chooser) noexcept {
          using ChooserType = std::decay_t<decltype(*chooser)>;
          return db::apply<typename ChooserType::argument_tags>(*chooser, box,
                                                                cache);
        });
    ASSERT(
        result > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result;
  }
};
