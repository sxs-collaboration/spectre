// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/// \brief Sets the initial RDMP data.
///
/// Used on the subcells after the TCI marked the DG solution as inadmissible.
struct SetInitialRdmpData {
  using argument_tags = tmpl::list<
      ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeYe,
      ValenciaDivClean::Tags::TildeTau, ValenciaDivClean::Tags::TildeB<>,
      evolution::dg::subcell::Tags::ActiveGrid, ::domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>>;
  using return_tags = tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;

  static void apply(
      gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
      const Scalar<DataVector>& subcell_tilde_d,
      const Scalar<DataVector>& subcell_tilde_ye,
      const Scalar<DataVector>& subcell_tilde_tau,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      evolution::dg::subcell::ActiveGrid active_grid, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh);
};
}  // namespace grmhd::ValenciaDivClean::subcell
