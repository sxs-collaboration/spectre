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
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
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
 * <table>
 * <caption>List of checks</caption>
 * <tr><th> Description <th> TCI status
 *
 * <tr><td> if `min(tilde_d)` is less than
 *  `tci_options.minimum_rest_mass_density_times_lorentz_factor`, or if
 *  `min(tilde_ye)` is less than
 *  `tci_options.minimum_rest_mass_density_times_lorentz_factor` times
 *  `tci_options.minimum_ye`, or if `min(tilde_tau)` is less than
 *  `tci_options.minimum_tilde_tau`, then the we remain on FD.
 * <td> `+1`
 *
 * <tr><td> if `grmhd::ValenciaDivClean::Tags::VariablesNeededFixing` is `true`
 *  and the maximum of rest mass density on FD grid is greater than
 * `tci_options.atmosphere_density`, then we remain on FD.
 * <td> `+2`
 *
 * <tr><td> apply the Persson TCI to \f$\tilde{D}\f$, \f$\tilde{Y}_e\f$, and
 * pressure if the maximum of rest mass density on FD grid is greater than
 * `tci_options.atmosphere_density`.
 * <td> `+3`
 *
 * <tr><td> apply the Persson TCI to \f$\tilde{Y}_e\f$ if the maximum of rest
 * mass density on FD grid is greater than `tci_options.atmosphere_density`.
 * <td> `+4`
 *
 * <tr><td> apply the Persson TCI to pressure if the maximum of rest mass
 * density on FD grid is greater than `tci_options.atmosphere_density`. <td>
 * `+5`
 *
 * <tr><td> apply the RDMP TCI to `TildeD`
 * <td> `+6`
 *
 * <tr><td> apply the RDMP TCI to `TildeYe`
 * <td> `+7`
 *
 * <tr><td> apply the RDMP TCI to `TildeTau`
 * <td> `+8`
 *
 * <tr><td> apply the RDMP TCI to `TildeB`
 * <td> `+9`
 *
 * <tr><td> apply the Persson TCI to the magnitude of \f$\tilde{B}^{n+1}\f$ if
 * its magnitude is greater than `tci_options.magnetic_field_cutoff`.
 * <td> `+10`
 *
 * </table>
 *
 * The second column of the table above denotes the value of an integer stored
 * as the first element of the returned `std::tuple`, which indicates the
 * particular kind of check that failed. For example, if the fifth check
 * (RDMP TCI to TildeTau) fails and cell is marked as troubled, an integer with
 * value `+5` is stored in the first slot of the returned tuple. Note that this
 * integer is marking only the _first_ check to fail, since checks are done in a
 * particular sequence as listed above. If all checks are passed and cell is not
 * troubled, it is returned with the value `0`.
 *
 * \note We adopt positive integers to mark TCI status from FD grid returned by
 * TciOnFdGrid class. Negative integers are reserved for TCIs on DG grid; see
 * TciOnDgGrid and its documentation.
 *
 */
struct TciOnFdGrid {
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                 grmhd::ValenciaDivClean::Tags::TildeYe,
                 grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                 hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 grmhd::ValenciaDivClean::Tags::VariablesNeededFixing,
                 domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::DataForRdmpTci, Tags::TciOptions,
                 evolution::dg::subcell::Tags::SubcellOptions<3>>;
  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      const Scalar<DataVector>& subcell_tilde_d,
      const Scalar<DataVector>& subcell_tilde_ye,
      const Scalar<DataVector>& subcell_tilde_tau,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      const Scalar<DataVector>& subcell_rest_mass_density,
      const Scalar<DataVector>& subcell_pressure, bool vars_needed_fixing,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const TciOptions& tci_options,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      double persson_exponent, bool need_rdmp_data_only);
};
}  // namespace grmhd::ValenciaDivClean::subcell
