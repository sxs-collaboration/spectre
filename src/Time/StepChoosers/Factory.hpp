// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ElementSizeCfl.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/Maximum.hpp"
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
        tmpl::list<StepChoosers::Cfl<Frame::Inertial, System>,
                   StepChoosers::ElementSizeCfl<System::volume_dim, System>>,
        tmpl::list<>>,
    StepChoosers::Constant,
    StepChoosers::ErrorControl<Use, typename System::variables_tag>,
    StepChoosers::LimitIncrease, StepChoosers::Maximum,
    StepChoosers::PreventRapidIncrease<typename System::variables_tag>>;
}  // namespace Factory_detail

template <typename System, bool HasCharSpeedFunctions = true>
using standard_step_choosers =
    Factory_detail::common_step_choosers<StepChooserUse::LtsStep, System,
                                         HasCharSpeedFunctions>;

template <typename System, bool HasCharSpeedFunctions = true>
using standard_slab_choosers =
    tmpl::push_back<Factory_detail::common_step_choosers<
                        StepChooserUse::Slab, System, HasCharSpeedFunctions>,
                    StepChoosers::StepToTimes>;
}  // namespace StepChoosers
