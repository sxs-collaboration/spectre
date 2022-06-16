// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ApparentHorizons/ObjectLabel.hpp"
#include "ControlSystem/ApparentHorizons/Measurements.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "Utilities/ProtocolHelpers.hpp"

static_assert(
    tt::assert_conforms_to_v<
        control_system::ah::BothHorizons::FindHorizon<::ah::ObjectLabel::A>::
            interpolation_target_tag<
                tmpl::list<control_system::TestHelpers::ExampleControlSystem>>,
        intrp::protocols::InterpolationTargetTag>);
