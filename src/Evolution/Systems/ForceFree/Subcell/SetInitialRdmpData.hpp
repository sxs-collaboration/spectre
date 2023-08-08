// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace ForceFree::subcell {
/*!
 * \brief Sets the initial RDMP data.
 *
 * Used on the subcells after the TCI marked the DG solution as inadmissible.
 */
struct SetInitialRdmpData {
  using argument_tags = tmpl::list<
      ForceFree::Tags::TildeE, ForceFree::Tags::TildeB, ForceFree::Tags::TildeQ,
      evolution::dg::subcell::Tags::ActiveGrid, ::domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>>;
  using return_tags = tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;

  static void apply(
      gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      const Scalar<DataVector>& subcell_tilde_q,
      evolution::dg::subcell::ActiveGrid active_grid, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh);
};
}  // namespace ForceFree::subcell
