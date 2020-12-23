// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
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

  /// The `last_step_magnitude` parameter describes the step size to be
  /// adjusted.  It may be the step size or the slab size, or may be
  /// infinite if the appropriate size cannot be determined.
  template <typename Metavariables, typename DbTags>
  double desired_step(
      const double last_step_magnitude, const db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    const auto result = call_with_dynamic_type<double, creatable_classes>(
        this,
        [&last_step_magnitude, &box, &cache](
            const auto* const chooser) noexcept {
          return db::apply(*chooser, box, last_step_magnitude, cache);
        });
    ASSERT(
        result > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result;
  }
};
