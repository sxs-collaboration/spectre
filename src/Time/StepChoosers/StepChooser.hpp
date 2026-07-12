// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

/// The intended use for a step chooser.  This is used to control the
/// classes via factories.
namespace StepChooserUse {
struct Slab;
struct LtsStep;
}  // namespace StepChooserUse

/// \cond
template <typename StepChooserUse>
class StepChooser;
/// \endcond

/// \ingroup TimeGroup
///
/// Holds all the StepChoosers
namespace StepChoosers {

namespace detail {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(compute_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(simple_tags)

template <typename Metavariables>
using all_step_choosers = tmpl::join<tmpl::remove<
    tmpl::list<
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 StepChooser<StepChooserUse::Slab>>,
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 StepChooser<StepChooserUse::LtsStep>>>,
    tmpl::no_such_type_>>;
}  // namespace detail

template <typename Metavariables>
using step_chooser_compute_tags = tmpl::remove_duplicates<
    tmpl::join<tmpl::transform<detail::all_step_choosers<Metavariables>,
                               detail::get_compute_tags_or_default<
                                   tmpl::_1, tmpl::pin<tmpl::list<>>>>>>;

template <typename Metavariables>
using step_chooser_simple_tags = tmpl::remove_duplicates<
    tmpl::join<tmpl::transform<detail::all_step_choosers<Metavariables>,
                               detail::get_simple_tags_or_default<
                                   tmpl::_1, tmpl::pin<tmpl::list<>>>>>>;
}  // namespace StepChoosers

/// A placeholder type to indicate that all constructible step choosers should
/// be used in step chooser utilities that permit a list of choosers to be
/// specified.
struct AllStepChoosers {};

/// \ingroup TimeGroup
///
/// \brief StepChoosers suggest upper bounds on step sizes.  See
/// `TimeStepRequest` for details on how the results are used.
///
/// Concrete StepChoosers should define `operator()` returning the
/// desired step and taking the `last_step` and arguments specified by
/// the class's `argument_tags` type alias.
///
/// Derived classes must indicate whether the chooser is usable as a
/// step chooser, slab chooser, or both by inheriting from StepChooser
/// with the appropriate `StepChooserUse` template argument.
template <typename StepChooserUse>
class StepChooser : public virtual PUP::able {
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

  /// Whether the result can differ on different elements, so
  /// requiring communication to synchronize the result across the
  /// domain.  This is ignored for LTS step changing.
  ///
  /// \note As this is only used for slab-size changing, the
  /// `last_step` passed to the call operator is *not* considered
  /// local data.
  virtual bool uses_local_data() const = 0;

  /// Whether the result can be applied with a delay.
  ///
  /// StepChoosers setting the `.end` or `.end_hard_limit` fields of
  /// `TimeStepRequest` must return false here.
  virtual bool can_be_delayed() const = 0;

  /// Whether the StepChooser's output only makes sense for setting
  /// the step size, as opposed to using it to set the slab size in an
  /// LTS evolution.
  ///
  /// This is generally true for StepChoosers that explicitly use past
  /// step sizes to compute their suggestion and false for others.
  virtual bool must_set_step_size() const = 0;

  /// The `last_step` parameter describes the step size to be
  /// adjusted.  It may be the step size or the slab size, or may be
  /// infinite if the appropriate size cannot be determined.
  ///
  /// The optional template parameter `StepChoosersToUse` may be used to
  /// indicate a subset of the constructable step choosers to use for the
  /// current application of `ChangeStepSize`. Passing `AllStepChoosers`
  /// (default) indicates that any constructible step chooser may be used. This
  /// option is used when multiple components need to invoke `ChangeStepSize`
  /// with step choosers that may not be compatible with all components.
  template <typename StepChoosersToUse = AllStepChoosers, typename DbTags>
  TimeStepRequest desired_step(const double last_step,
                               const db::DataBox<DbTags>& box) const {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    using step_choosers =
        tmpl::conditional_t<std::is_same_v<StepChoosersToUse, AllStepChoosers>,
                            tmpl::at<factory_classes, StepChooser>,
                            StepChoosersToUse>;
    const auto result = call_with_dynamic_type<TimeStepRequest, step_choosers>(
        this, [&last_step, &box](const auto* const chooser) {
          return db::apply(*chooser, box, last_step);
        });
    ASSERT(not result.size_goal.has_value() or
               (*result.size_goal > 0.) == (last_step > 0.),
           "Step size changed sign: " << *result.size_goal);
    ASSERT(
        not result.size.has_value() or (*result.size > 0.) == (last_step > 0.),
        "Step size changed sign: " << *result.size);
    ASSERT(not result.size_hard_limit.has_value() or
               (*result.size_hard_limit > 0.) == (last_step > 0.),
           "Step size changed sign: " << *result.size_hard_limit);
    ASSERT(not(result.end.has_value() and can_be_delayed()),
           "Delayable end limits are not allowed.");
    ASSERT(not(result.end_hard_limit.has_value() and can_be_delayed()),
           "Delayable end limits are not allowed.");
    return result;
  }
};
