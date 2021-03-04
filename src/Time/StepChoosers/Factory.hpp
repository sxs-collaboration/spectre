// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ElementSizeCfl.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace StepChoosers {
namespace Factory_detail {
template <typename Use, typename System, bool HasCharSpeedFunctions>
using common_step_choosers = tmpl::push_back<
    tmpl::conditional_t<
        HasCharSpeedFunctions,
        tmpl::list<
            StepChoosers::Cfl<Use, Frame::Inertial, System>,
            StepChoosers::ElementSizeCfl<Use, System::volume_dim, System>>,
        tmpl::list<>>,
    StepChoosers::Constant<Use>, StepChoosers::Increase<Use>>;
template <typename Use>
using step_choosers_for_step_only =
    tmpl::list<StepChoosers::PreventRapidIncrease<Use>>;
using step_choosers_for_slab_only = tmpl::list<StepChoosers::StepToTimes>;

template <typename System, bool HasCharSpeedFunctions>
using lts_slab_choosers = tmpl::append<
    common_step_choosers<StepChooserUse::Slab, System, HasCharSpeedFunctions>,
    step_choosers_for_slab_only>;
}  // namespace Factory_detail

template <typename System, bool HasCharSpeedFunctions = true>
using standard_step_choosers = tmpl::append<
    Factory_detail::common_step_choosers<StepChooserUse::LtsStep, System,
                                         HasCharSpeedFunctions>,
    Factory_detail::step_choosers_for_step_only<StepChooserUse::LtsStep>,
    tmpl::list<StepChoosers::ErrorControl<typename System::variables_tag>>>;

template <typename System, bool LocalTimeStepping,
          bool HasCharSpeedFunctions = true>
using standard_slab_choosers = tmpl::conditional_t<
    LocalTimeStepping,
    Factory_detail::lts_slab_choosers<System, HasCharSpeedFunctions>,
    tmpl::append<
        Factory_detail::lts_slab_choosers<System, HasCharSpeedFunctions>,
        Factory_detail::step_choosers_for_step_only<StepChooserUse::Slab>>>;
}  // namespace StepChoosers
