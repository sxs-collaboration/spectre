// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace ForceFree::subcell {
namespace detail {
std::tuple<int, evolution::dg::subcell::RdmpTciData> initial_data_tci_work(
    const Scalar<DataVector>& dg_tilde_e_mag,
    const Scalar<DataVector>& dg_tilde_b_mag,
    const Scalar<DataVector>& dg_tilde_q,
    const Scalar<DataVector>& subcell_tilde_e_mag,
    const Scalar<DataVector>& subcell_tilde_b_mag,
    const Scalar<DataVector>& subcell_tilde_q, const double persson_exponent,
    const Mesh<3>& dg_mesh);
}

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
 * <tr><td> apply the Persson TCI to magnitude of `TildeE`
 * <td> `-1`
 *
 * <tr><td> apply the Persson TCI to magnitude of `TildeB`
 * <td> `-2`
 *
 * <tr><td> apply the Persson TCI to `TildeQ`
 * <td> `-3`
 *
 * <tr><td> apply the two-mesh relaxed discrete maximum principle TCI to
 * `TildeE`, `TildeB`, `TildePsi`, `TildePhi`, and `TildeQ`.
 * <td> `-4`
 *
 * </table>
 *
 * The second column of the table above denotes the value of an integer stored
 * as the first element of the returned `std::tuple`, which indicates the
 * particular kind of check that failed. For example, if the 3rd check
 * (Persson TCI to TildeQ) fails and cell is marked as troubled, an integer with
 * value `-3` is stored in the first slot of the returned tuple. Note that this
 * integer is marking only the first check to fail since checks are done in a
 * particular sequence as listed above. If all checks are passed and cell is not
 * troubled, it is returned with the value `0`.
 *
 * When computing magnitudes of tensor quantities (TildeE, TildeB) here for TCI
 * checks, we simply use the square root of the sum of spatial components
 * squared, _not_ the square root of the scalar product using the spatial
 * metric.
 *
 * See also the documentation of TciOnDgGrid.
 *
 */
struct DgInitialDataTci {
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>>;

  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      const Variables<tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi,
                                 Tags::TildePhi, Tags::TildeQ>>& dg_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh);
};

/*!
 * \brief Sets the initial RDMP data.
 *
 * Used on the subcells after the TCI marked the DG solution as inadmissible.
 */
struct SetInitialRdmpData {
  using argument_tags =
      tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                 ForceFree::Tags::TildeQ,
                 evolution::dg::subcell::Tags::ActiveGrid>;
  using return_tags = tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;

  static void apply(
      gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      const Scalar<DataVector>& subcell_tilde_q,
      evolution::dg::subcell::ActiveGrid active_grid);
};
}  // namespace ForceFree::subcell
