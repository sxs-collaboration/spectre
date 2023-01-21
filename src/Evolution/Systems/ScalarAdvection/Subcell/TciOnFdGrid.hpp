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
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace ScalarAdvection::subcell {
/*!
 * \brief Troubled-cell indicator applied to the finite difference subcell
 * solution to check if the corresponding DG solution is admissible.
 *
 * Applies 1) the RDMP TCI to \f$U\f$ and 2) the Persson TCI to \f$U\f$ if the
 * \f$\max(|U|)\f$ on the DG grid is greater than `tci_options.u_cutoff`.
 */
template <size_t Dim>
struct TciOnFdGrid {
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<ScalarAdvection::Tags::U, ::domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::DataForRdmpTci,
                 evolution::dg::subcell::Tags::SubcellOptions<Dim>,
                 Tags::TciOptions>;

  static std::tuple<bool, evolution::dg::subcell::RdmpTciData> apply(
      const Scalar<DataVector>& subcell_u, const Mesh<Dim>& dg_mesh,
      const Mesh<Dim>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      const TciOptions& tci_options, double persson_exponent,
      bool need_rdmp_data_only);
};
}  // namespace ScalarAdvection::subcell
