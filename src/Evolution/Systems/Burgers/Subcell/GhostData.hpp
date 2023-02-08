// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename T>
class Variables;
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief Returns \f$U\f$, the variables needed for reconstruction.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
class GhostVariables {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<Burgers::Tags::U>>>;

  static DataVector apply(const Variables<tmpl::list<Burgers::Tags::U>>& vars,
                          size_t rdmp_size);
};
}  // namespace Burgers::subcell
