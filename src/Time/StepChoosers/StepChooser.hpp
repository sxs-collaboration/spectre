// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

/// The intended use for a step chooser.  This is used to control the
/// classes via factories.
namespace StepChooserUse {
struct Slab;
struct LtsStep;
}  // namespace StepChooserUse

/// \ingroup TimeGroup
///
/// \brief StepChoosers suggest upper bounds on step sizes.
///
/// Concrete StepChoosers should define `operator()` returning the
/// magnitude of the desired step (as a double).
///
/// Derived classes must indicate whether the chooser is usable as a
/// step chooser, slab chooser, or both by inheriting from StepChooser
/// with the appropriate `StepChooserUse` template argument.  A class
/// cannot inherit from both base classes simultaneously; if both uses
/// are supported the use must be chosen using a template parameter.
/// If a derived class is usable as a step chooser (i.e., inherits
/// from StepChooser<StepChooserUse::LtsStep>), it must specify type
/// aliases `argument_tags` and `return_tags` for the arguments to the
/// call operator. If it is usable only for slab choosing (i.e.,
/// unconditionally inherits from StepChooser<StepChooserUse::Slab>),
/// it need only specify `argument_tags`, as slab choosers are not
/// permitted mutable access to the \ref DataBoxGroup "DataBox".
///
/// For the class interface, see the specializations
/// StepChooser<StepChooserUse::LtsStep> and
/// StepChooser<StepChooserUse::Slab>.
template <typename StepChooserUse>
class StepChooser;

/// \ingroup TimeGroup
///
/// Holds all the StepChoosers
namespace StepChoosers {

namespace detail {
CREATE_HAS_TYPE_ALIAS(compute_tags)
CREATE_HAS_TYPE_ALIAS_V(compute_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(compute_tags)
CREATE_HAS_TYPE_ALIAS(simple_tags)
CREATE_HAS_TYPE_ALIAS_V(simple_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(simple_tags)

template <typename Metavariables>
using all_step_choosers = tmpl::join<tmpl::transform<
    tmpl::filter<
        tmpl::wrap<typename Metavariables::factory_creation::factory_classes,
                   tmpl::list>,
        tt::is_a_lambda<StepChooser, tmpl::bind<tmpl::front, tmpl::_1>>>,
    tmpl::bind<tmpl::back, tmpl::_1>>>;
}  // namespace detail

template <typename Metavariables>
using step_chooser_compute_tags = tmpl::remove_duplicates<
    tmpl::join<tmpl::transform<detail::all_step_choosers<Metavariables>,
                               tmpl::bind<detail::get_compute_tags_or_default_t,
                                          tmpl::_1, tmpl::pin<tmpl::list<>>>>>>;

template <typename Metavariables>
using step_chooser_simple_tags = tmpl::remove_duplicates<
    tmpl::join<tmpl::transform<detail::all_step_choosers<Metavariables>,
                               tmpl::bind<detail::get_simple_tags_or_default_t,
                                          tmpl::_1, tmpl::pin<tmpl::list<>>>>>>;
}  // namespace StepChoosers

template <>
class StepChooser<StepChooserUse::Slab> : public PUP::able {
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

  /// The `last_step_magnitude` parameter describes the slab size to be
  /// adjusted; It may be infinite if the appropriate size cannot be determined.
  ///
  /// As opposed to the decision made by `desired_step` in the
  /// StepChooser<StepChooserUse::LtsStep> specialization, the slab
  /// change decision must be callable from an event (so cannot store
  /// state information in the \ref DataBoxGroup "DataBox"), and we do
  /// not have the capability to reject a slab so this function
  /// returns only a `double` indicating the desired slab size.
  template <typename Metavariables, typename DbTags>
  double desired_slab(
      const double last_step_magnitude,
      const db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    const auto result =
        call_with_dynamic_type<std::pair<double, bool>,
                               tmpl::at<factory_classes, StepChooser>>(
            this, [&last_step_magnitude, &box,
                   &cache](const auto* const chooser) noexcept {
              return db::apply(*chooser, box, last_step_magnitude, cache);
            });
    ASSERT(
        result.first > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result.first;
  }
};

/// A placeholder type to indicate that all constructible step choosers should
/// be used in step chooser utilities that permit a list of choosers to be
/// specified.
struct AllStepChoosers {};

template <>
class StepChooser<StepChooserUse::LtsStep> : public PUP::able {
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
  /// The optional template parameter `StepChoosersToUse` may be used to
  /// indicate a subset of the constructable step choosers to use for the
  /// current application of `ChangeStepSize`. Passing `AllStepChoosers`
  /// (default) indicates that any constructible step chooser may be used. This
  /// option is used when multiple components need to invoke `ChangeStepSize`
  /// with step choosers that may not be compatible with all components.
  template <typename StepChoosersToUse = AllStepChoosers,
            typename Metavariables, typename DbTags>
  std::pair<double, bool> desired_step(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::factory_creation::factory_classes;
    using step_choosers =
        tmpl::conditional_t<std::is_same_v<StepChoosersToUse, AllStepChoosers>,
                            tmpl::at<factory_classes, StepChooser>,
                            StepChoosersToUse>;
    const auto result =
        call_with_dynamic_type<std::pair<double, bool>, step_choosers>(
            this, [&last_step_magnitude, &box,
                   &cache](const auto* const chooser) noexcept {
              using chooser_type = typename std::decay_t<decltype(*chooser)>;
              return db::mutate_apply<typename chooser_type::return_tags,
                                      typename chooser_type::argument_tags>(
                  *chooser, box, last_step_magnitude, cache);
            });
    ASSERT(
        result.first > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result;
  }
};
