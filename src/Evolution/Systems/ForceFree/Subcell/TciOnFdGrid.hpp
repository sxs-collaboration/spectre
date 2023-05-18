// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree::subcell {

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
 * <tr><td> apply the Persson TCI to magnitude of `TildeE`
 * <td> `+1`
 *
 * <tr><td> apply the Persson TCI to magnitude of `TildeB`
 * <td> `+2`
 *
 * <tr><td> apply the Persson TCI to `TildeQ`
 * <td> `+3`
 *
 * <tr><td> apply the RDMP TCI to magnitude of `TildeE`
 * <td> `+4`
 *
 * <tr><td> apply the RDMP TCI to magnitude of `TildeB`
 * <td> `+5`
 *
 * <tr><td> apply the RDMP TCI to `TildeQ`
 * <td> `+6`
 *
 * </table>
 *
 * The second column of the table above denotes the value of an integer stored
 * as the first element of the returned `std::tuple`, which indicates the
 * particular kind of check that failed. For example, if the second check
 * (Persson TCI to `mag(TildeB)`) fails and element is marked as troubled, an
 * integer with value `+2` is stored in the first slot of the returned tuple.
 * Note that this integer is marking only the _first_ check to fail since
 * checks are done in a particular sequence as listed above. If all checks are
 * passed and cell is not troubled, it is returned with the value `0`.
 *
 * When computing magnitudes of tensor quantities (TildeE, TildeB) for TCI
 * checks, we simply use the square root of the sum of spatial components
 * squared, _not_ the square root of the scalar product using the spatial
 * metric.
 *
 * \note We adopt positive integers to mark TCI status from FD grid returned by
 * TciOnFdGrid class. Negative integers are reserved for TCIs on DG grid; see
 * TciOnDgGrid and its documentation.
 *
 */
class TciOnFdGrid {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildeQ,
                 domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::DataForRdmpTci,
                 evolution::dg::subcell::Tags::SubcellOptions<3>>;

  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_tilde_b,
      const Scalar<DataVector>& subcell_tilde_q, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      double persson_exponent, bool need_rdmp_data_only);
};

}  // namespace ForceFree::subcell
