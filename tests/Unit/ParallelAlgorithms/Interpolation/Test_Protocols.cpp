// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ParallelAlgorithms/Interpolation/Examples.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "Utilities/ProtocolHelpers.hpp"

static_assert(
    tt::assert_conforms_to<intrp::TestHelpers::ExampleComputeTargetPoints,
                           intrp::protocols::ComputeTargetPoints>);
static_assert(
    tt::assert_conforms_to<intrp::TestHelpers::ExampleComputeVarsToInterpolate,
                           intrp::protocols::ComputeVarsToInterpolate>);
