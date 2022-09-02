// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace evolution::dg::subcell {
class SubcellOptions;
}  // namespace evolution::dg::subcell
template <size_t Dim>
class Mesh;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief The troubled-cell indicator run on the FD grid to check if the
 * corresponding DG solution is admissible.
 *
 * The following checks are done in the order they are listed:
 *
 * - if `grmhd::ValenciaDivClean::Tags::VariablesNeededFixing` is `true` then we
 *   remain on FD. (Note: this could be relaxed in the future if we need to
 *   allow switching from FD to DG in the atmosphere and the current approach
 *   isn't working.)
 * - if `min(tilde_d)` is less than
 *   `tci_options.minimum_rest_mass_density_times_lorentz_factor` or if
 *   `min(tilde_tau)` is less than `tci_options.minimum_tilde_tau` then the we
 *   remain on FD.
 * - apply the Persson TCI to \f$\tilde{D}\f$ and \f$\tilde{\tau}\f$
 * - apply the RDMP TCI to `TildeD`, `TildeTau` and `magnitude(TildeB)`.
 * - apply the Persson TCI to the magnitude of \f$\tilde{B}^{n+1}\f$ if its
 *   magnitude is greater than `tci_options.magnetic_field_cutoff`.
 */
struct TciOnFdGrid {
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                 grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                 grmhd::ValenciaDivClean::Tags::VariablesNeededFixing,
                 domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::DataForRdmpTci, Tags::TciOptions,
                 evolution::dg::subcell::Tags::SubcellOptions>;
  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      const Scalar<DataVector>& subcell_tilde_d,
      const Scalar<DataVector>& subcell_tilde_tau,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      bool vars_needed_fixing, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const TciOptions& tci_options,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      double persson_exponent);
};
}  // namespace grmhd::ValenciaDivClean::subcell
