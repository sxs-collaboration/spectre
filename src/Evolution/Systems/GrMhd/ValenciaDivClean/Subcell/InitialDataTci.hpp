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
namespace detail {
std::tuple<int, evolution::dg::subcell::RdmpTciData> initial_data_tci_work(
    const Scalar<DataVector>& dg_tilde_d, const Scalar<DataVector>& dg_tilde_ye,
    const Scalar<DataVector>& dg_tilde_tau,
    const Scalar<DataVector>& dg_tilde_b_magnitude,
    const Scalar<DataVector>& subcell_tilde_d,
    const Scalar<DataVector>& subcell_tilde_ye,
    const Scalar<DataVector>& subcell_tilde_tau,
    const Scalar<DataVector>& subcell_tilde_b_magnitude,
    const double persson_exponent, const Mesh<3>& dg_mesh,
    const TciOptions& tci_options);
}  // namespace detail

/*!
 * \brief The troubled-cell indicator run on DG initial data to see if we need
 * to switch to subcell.
 *
 * The following checks are done in the order they are listed:
 *
 * <table>
 * <caption>List of checks</caption>
 * <tr><th> Description <th> TCI status
 *
 * <tr><td> if `TildeD` on the DG or subcell grid (projected from the DG grid,
 * not initialized to the initial data) is less than
 * `tci_options.minimum_rest_mass_density_times_lorentz_factor` then the element
 * is flagged as troubled.
 * <td> `-1`
 *
 * <tr><td> if `TildeTau` on the DG or subcell grid (projected from
 * the DG grid, not initialized to the initial data) is less than
 * `tci_options.minimum_tilde_tau` then the element is flagged as troubled
 * <td> `-2`
 *
 * <tr><td> apply the Persson TCI to \f$\tilde{D}\f$ and \f$\tilde{\tau}\f$.
 * <td> `-5`
 *
 * <tr><td> apply the Persson TCI to the magnitude of \f$\tilde{B}\f$ if its
 * magnitude on the DG grid is greater than `tci_options.magnetic_field_cutoff`.
 * <td> `-6`
 *
 * <tr><td>
 * apply the two-mesh relaxed discrete maximum principle TCI to `TildeD`,
 * `TildeTau`, `TildeS`, `TildeB`, and `TildePhi`
 * <td> `-7`
 *
 * </table>
 *
 * The second column of the table above denotes the value of an integer stored
 * as the first element of the returned `std::tuple`, which indicates the
 * particular kind of check that failed. For example, if the third check
 * (Persson TCI to TildeD and TildeTau) fails and cell is marked as troubled,
 * an integer with value `-5` is stored in the first slot of the returned tuple.
 * Note that this integer is marking only the _first_ check to fail, since
 * checks are done in a particular sequence as listed above. If all checks are
 * passed and cell is not troubled, it is returned with the value `0`.
 *
 * Somewhat seemingly irregular prescription of TCI status values is due to
 * matching those with TciOnDgGrid as much as possible (see the documentation of
 * TciOnDgGrid).
 *
 */
struct DgInitialDataTci {
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 Tags::TciOptions>;

  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      const Variables<tmpl::list<
          ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeYe,
          ValenciaDivClean::Tags::TildeTau, ValenciaDivClean::Tags::TildeS<>,
          ValenciaDivClean::Tags::TildeB<>, ValenciaDivClean::Tags::TildePhi>>&
          dg_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const TciOptions& tci_options);
};

/// \brief Sets the initial RDMP data.
///
/// Used on the subcells after the TCI marked the DG solution as inadmissible.
struct SetInitialRdmpData {
  using argument_tags = tmpl::list<
      ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeYe,
      ValenciaDivClean::Tags::TildeTau, ValenciaDivClean::Tags::TildeB<>,
      evolution::dg::subcell::Tags::ActiveGrid>;
  using return_tags = tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;

  static void apply(
      gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
      const Scalar<DataVector>& subcell_tilde_d,
      const Scalar<DataVector>& subcell_tilde_ye,
      const Scalar<DataVector>& subcell_tilde_tau,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      const evolution::dg::subcell::ActiveGrid active_grid);
};
}  // namespace grmhd::ValenciaDivClean::subcell
