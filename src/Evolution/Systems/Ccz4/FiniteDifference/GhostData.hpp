// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename T>
class Variables;
/// \endcond

namespace Ccz4::fd {
/*!
 * \brief Get the Ccz4 evolution variables for ghost
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
class GhostVariables {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<Ccz4::fd::System::variables_tag_list>>;

  static DataVector apply(
      const Variables<Ccz4::fd::System::variables_tag_list>& evolved_vars,
      size_t rdmp_size);
};
}  // namespace Ccz4::fd
