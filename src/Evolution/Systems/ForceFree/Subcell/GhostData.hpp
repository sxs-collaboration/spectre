// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename T>
class Variables;
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace ForceFree::subcell {
/*!
 * \brief Returns \f$\tilde{J}^i\f$, \f$\tilde{E}^i\f$, \f$\tilde{B}^i\f$,
 * \f$\tilde{\psi}\f$, \f$\tilde{\phi}\f$ and \f$\tilde{q}\f$ for FD
 * reconstruction.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
class GhostVariables {
 private:
  using evolved_vars =
      tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                 ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                 ForceFree::Tags::TildeQ>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<evolved_vars>, ForceFree::Tags::TildeJ>;

  static DataVector apply(
      const Variables<evolved_vars>& vars,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j, size_t rdmp_size);
};
}  // namespace ForceFree::subcell
