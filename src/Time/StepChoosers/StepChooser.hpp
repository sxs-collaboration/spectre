// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
template <typename StepChooserRegistrars>
class StepChooser;
// IWYU pragma: no_forward_declare db::DataBox
template <typename StepChooserRegistrars>
class StepChooser;
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

namespace detail {
CREATE_HAS_TYPE_ALIAS(slab_choosers)
CREATE_HAS_TYPE_ALIAS_V(slab_choosers)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(slab_choosers)
CREATE_HAS_TYPE_ALIAS(step_choosers)
CREATE_HAS_TYPE_ALIAS_V(step_choosers)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(step_choosers)
CREATE_HAS_TYPE_ALIAS(compute_tags)
CREATE_HAS_TYPE_ALIAS_V(compute_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(compute_tags)
CREATE_HAS_TYPE_ALIAS(simple_tags)
CREATE_HAS_TYPE_ALIAS_V(simple_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(simple_tags)
}  // namespace detail

/// Designation for the context in which a step chooser may be used
enum class UsableFor { OnlyLtsStepChoice, OnlySlabChoice, AnyStepChoice };

template <typename Metavariables>
using step_chooser_compute_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        typename StepChooser<tmpl::remove_duplicates<tmpl::append<
            detail::get_step_choosers_or_default_t<Metavariables, tmpl::list<>>,
            detail::get_slab_choosers_or_default_t<
                Metavariables, tmpl::list<>>>>>::creatable_classes,
        tmpl::bind<detail::get_compute_tags_or_default_t, tmpl::_1,
                   tmpl::pin<tmpl::list<>>>>>>;

template <typename Metavariables>
using step_chooser_simple_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        typename StepChooser<tmpl::remove_duplicates<tmpl::append<
            detail::get_step_choosers_or_default_t<Metavariables, tmpl::list<>>,
            detail::get_slab_choosers_or_default_t<
                Metavariables, tmpl::list<>>>>>::creatable_classes,
        tmpl::bind<detail::get_simple_tags_or_default_t, tmpl::_1,
                   tmpl::pin<tmpl::list<>>>>>>;
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
///
/// Derived classes must specify the static member variable
/// `StepChoosers::UsableFor usable_for`, indicating whether the chooser is
/// usable as a step chooser, slab chooser, or both. If it is usable as a step
/// chooser (`StepChoosers::UsableFor::OnlyLtsStepChoice` or
/// `StepChoosers::UsableFor::AnyStepChoice`), it must specify type aliases
/// `argument_tags` and `return_tags` for the arguments to the call operator. If
/// it is usable only for slab choosing
/// (`StepChoosers::UsableFor::OnlySlabChoice`), it need only specify
/// `argument_tags`, as slab choosers are not permitted mutable access to the
/// \ref DataBoxGroup "DataBox".
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
  ///
  /// The return value of this function contains the desired step size
  /// and a `bool` indicating whether the step should be accepted. If the `bool`
  /// is `false`, the current time step will be recomputed with a step size
  /// informed by the desired step value returned by this function. The
  /// implementations of the call operator in derived classes should always
  /// return a strictly smaller step than the `last_step_magnitude` when they
  /// return `false` for the second member of the pair (indicating step
  /// rejection).
  template <typename Metavariables, typename DbTags>
  std::pair<double, bool> desired_step(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    const auto result =
        call_with_dynamic_type<std::pair<double, bool>, creatable_classes>(
            this, [&last_step_magnitude, &box,
                   &cache](const auto* const chooser) noexcept {
              using chooser_type = typename std::decay_t<decltype(*chooser)>;
              static_assert(chooser_type::usable_for ==
                                    StepChoosers::UsableFor::AnyStepChoice or
                                chooser_type::usable_for ==
                                    StepChoosers::UsableFor::OnlyLtsStepChoice,
                            "The chosen step chooser is not usable for a local "
                            "time-stepping step choice.");
              return db::mutate_apply<typename chooser_type::return_tags,
                                      typename chooser_type::argument_tags>(
                  *chooser, box, last_step_magnitude, cache);
            });
    ASSERT(
        result.first > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result;
  }

  /// The `last_step_magnitude` parameter describes the slab size to be
  /// adjusted; It may be infinite if the appropriate size cannot be determined.
  ///
  /// This function is distinct from `desired_step` because the slab change
  /// decision must be callable from an event (so cannot store state information
  /// in the \ref DataBoxGroup "DataBox"), and we do not have the capability to
  /// reject a slab so this function returns only a `double` indicating the
  /// desired slab size.
  template <typename Metavariables, typename DbTags>
  double desired_slab(
      const double last_step_magnitude,
      const db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    const auto result = call_with_dynamic_type<
        std::pair<double, bool>,
        creatable_classes>(this, [&last_step_magnitude, &box,
                                  &cache](const auto* const chooser) noexcept {
          using chooser_type = typename std::decay_t<decltype(*chooser)>;
          static_assert(chooser_type::usable_for ==
                                StepChoosers::UsableFor::AnyStepChoice or
                            chooser_type::usable_for ==
                                StepChoosers::UsableFor::OnlySlabChoice,
                        "The chosen step chooser is not usable for making a "
                        "slab choice.");
          return db::apply(*chooser, box, last_step_magnitude, cache);
    });
    ASSERT(
        result.first > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result.first;
  }
};
