// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Tags/Metavariables.hpp"
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

template <typename Metavariables, bool UsingLts>
using all_step_choosers = tmpl::join<tmpl::remove<
    tmpl::list<
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 StepChooser<StepChooserUse::Slab>>,
        tmpl::conditional_t<
            UsingLts,
            tmpl::at<typename Metavariables::factory_creation::factory_classes,
                     StepChooser<StepChooserUse::LtsStep>>,
            tmpl::no_such_type_>>,
    tmpl::no_such_type_>>;
}  // namespace detail

template <typename Metavariables, bool UsingLts>
using step_chooser_compute_tags = tmpl::remove_duplicates<tmpl::join<
    tmpl::transform<detail::all_step_choosers<Metavariables, UsingLts>,
                    detail::get_compute_tags_or_default<
                        tmpl::_1, tmpl::pin<tmpl::list<>>>>>>;

template <typename Metavariables, bool UsingLts>
using step_chooser_simple_tags = tmpl::remove_duplicates<tmpl::join<
    tmpl::transform<detail::all_step_choosers<Metavariables, UsingLts>,
                    detail::get_simple_tags_or_default<
                        tmpl::_1, tmpl::pin<tmpl::list<>>>>>>;
}  // namespace StepChoosers

/// A placeholder type to indicate that all constructible step choosers should
/// be used in step chooser utilities that permit a list of choosers to be
/// specified.
struct AllStepChoosers {};

/// \ingroup TimeGroup
///
/// \brief StepChoosers suggest upper bounds on step sizes.
///
/// Concrete StepChoosers should define `operator()` returning the
/// information described as the return type of `desired_step` and
/// taking the `last_step_magnitude` and arguments specified by the
/// class's `argument_tags` type alias.
///
/// Derived classes must indicate whether the chooser is usable as a
/// step chooser, slab chooser, or both by inheriting from StepChooser
/// with the appropriate `StepChooserUse` template argument.  A class
/// cannot inherit from both base classes simultaneously; if both uses
/// are supported the use must be chosen using a template parameter.
template <typename StepChooserUse>
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

  /// Whether the result can differ on different elements, so
  /// requiring communication to synchronize the result across the
  /// domain.  This is ignored for LTS step changing.
  ///
  /// \note As this is only used for slab-size changing, the
  /// `last_step_magnitude` passed to the call operator is *not*
  /// considered local data.
  virtual bool uses_local_data() const = 0;

  /// The `last_step_magnitude` parameter describes the step size to be
  /// adjusted.  It may be the step size or the slab size, or may be
  /// infinite if the appropriate size cannot be determined.
  ///
  /// The return value of this function contains the desired step size
  /// and a `bool` indicating whether the step should be accepted.
  /// When adjusting LTS step sizes, if the `bool` is `false`, the
  /// current time step will be recomputed with a step size informed
  /// by the desired step value returned by this function.  We do not
  /// have the capability to reject a slab, so the `bool` is ignored
  /// for slab adjustment.
  ///
  /// The implementations of the call operator in derived classes
  /// should always return a strictly smaller step than the
  /// `last_step_magnitude` when they return `false` for the second
  /// member of the pair (indicating step rejection).
  ///
  /// The optional template parameter `StepChoosersToUse` may be used to
  /// indicate a subset of the constructable step choosers to use for the
  /// current application of `ChangeStepSize`. Passing `AllStepChoosers`
  /// (default) indicates that any constructible step chooser may be used. This
  /// option is used when multiple components need to invoke `ChangeStepSize`
  /// with step choosers that may not be compatible with all components.
  template <typename StepChoosersToUse = AllStepChoosers, typename DbTags>
  std::pair<double, bool> desired_step(const double last_step_magnitude,
                                       const db::DataBox<DbTags>& box) const {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    using step_choosers =
        tmpl::conditional_t<std::is_same_v<StepChoosersToUse, AllStepChoosers>,
                            tmpl::at<factory_classes, StepChooser>,
                            StepChoosersToUse>;
    const auto result =
        call_with_dynamic_type<std::pair<double, bool>, step_choosers>(
            this, [&last_step_magnitude, &box](const auto* const chooser) {
              return db::apply(*chooser, box, last_step_magnitude);
            });
    ASSERT(
        result.first > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result;
  }
};
