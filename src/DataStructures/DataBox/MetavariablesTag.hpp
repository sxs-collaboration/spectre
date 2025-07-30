// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// \brief Tag to retrieve the `Metavariables` from the DataBox.
///
/// \details To insert the metavariables into the DataBox use
/// `Parallel::Tags::MetavariablesImpl<metavariables>`
struct Metavariables {};

/// \ingroup DataBoxTagsGroup
/// \brief Tag to insert Metavars into the DataBox
///
/// \details Can be retrieved via `Parallel::Tags::Metavariables` (i.e. without
/// the template parameter)
template <typename Metavars>
struct MetavariablesImpl : Metavariables, db::SimpleTag {
  using type = Metavars;
  static std::string name() { return "Metavariables"; }
};
}  // namespace Parallel::Tags
