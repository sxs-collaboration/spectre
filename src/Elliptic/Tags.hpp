// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/DataImporter/Tags.hpp"
#include "Options/Options.hpp"

namespace elliptic {
namespace OptionTags {
/*!
 * \brief Holds option tags for importing numeric data as initial guess for an
 * elliptic solve.
 */
struct NumericInitialGuess {
  using group = importer::OptionTags::Group;
  static constexpr OptionString help = "Initial guess";
};
}  // namespace OptionTags
}  // namespace elliptic
