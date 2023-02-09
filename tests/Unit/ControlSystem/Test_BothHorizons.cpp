// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ControlSystem/ApparentHorizons/BothHorizons.hpp"
#include "Domain/ObjectLabel.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "Utilities/ProtocolHelpers.hpp"

static_assert(
    tt::assert_conforms_to_v<
        control_system::ah::BothHorizons::
            FindHorizon<::domain::ObjectLabel::A>::interpolation_target_tag<
                tmpl::list<control_system::TestHelpers::ExampleControlSystem>>,
        intrp::protocols::InterpolationTargetTag>);
