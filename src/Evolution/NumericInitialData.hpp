// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace evolution {

/// Use in place of an analytic solution or analytic data to start an evolution
/// with numeric initial data loaded from a data file.
struct NumericInitialData : tt::ConformsTo<protocols::NumericInitialData> {};

}  // namespace evolution
