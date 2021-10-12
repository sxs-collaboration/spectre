// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace Sources {

/*!
 * \brief Used to mark that the initial data do not require source
 * terms in the evolution equations.
 */
struct NoSource {
  using sourced_variables = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
}  // namespace Sources
}  // namespace NewtonianEuler
