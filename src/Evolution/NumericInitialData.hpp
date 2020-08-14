// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace evolution {

/*!
 * \brief Provides compile-time information to import numeric initial data for
 * the given `System` from a volume data file.
 */
template <typename System>
struct NumericInitialData : tt::ConformsTo<protocols::NumericInitialData> {
  using import_fields = typename System::variables_tag::tags_list;
};

}  // namespace evolution
