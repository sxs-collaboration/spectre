// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Time/History.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace imex::Tags {
/// Tag for the history of one of the implicit sectors of an IMEX
/// system.
template <typename ImplicitSector>
struct ImplicitHistory : db::SimpleTag {
  static_assert(
      tt::assert_conforms_to_v<ImplicitSector, protocols::ImplicitSector>);
  using type =
      TimeSteppers::History<Variables<typename ImplicitSector::tensors>>;
};
}  // namespace imex::Tags
