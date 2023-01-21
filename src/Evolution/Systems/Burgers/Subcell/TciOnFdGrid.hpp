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
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief Troubled-cell indicator applied to the finite difference subcell
 * solution to check if the corresponding DG solution is admissible.
 *
 * Applies the Persson to \f$U\f$ on the DG grid, and the RDMP TCI to \f$U\f$.
 */
struct TciOnFdGrid {
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<Burgers::Tags::U, ::domain::Tags::Mesh<1>,
                 evolution::dg::subcell::Tags::Mesh<1>,
                 evolution::dg::subcell::Tags::DataForRdmpTci,
                 evolution::dg::subcell::Tags::SubcellOptions<1>>;

  static std::tuple<bool, evolution::dg::subcell::RdmpTciData> apply(
      const Scalar<DataVector>& subcell_u, const Mesh<1>& dg_mesh,
      const Mesh<1>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      double persson_exponent, bool need_rdmp_data_only);
};
}  // namespace Burgers::subcell
