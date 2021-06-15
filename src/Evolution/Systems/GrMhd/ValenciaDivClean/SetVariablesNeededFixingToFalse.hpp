// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean {
/*!
 * \brief Mutator used with `Initialization::Actions::AddSimpleTags` to
 * initialize the `VariablesNeededFixing` to `false`
 */
struct SetVariablesNeededFixingToFalse {
  using return_tags = tmpl::list<Tags::VariablesNeededFixing>;
  using argument_tags = tmpl::list<>;

  static void apply(gsl::not_null<bool*> variables_needed_fixing) noexcept;
};
}  // namespace grmhd::ValenciaDivClean
