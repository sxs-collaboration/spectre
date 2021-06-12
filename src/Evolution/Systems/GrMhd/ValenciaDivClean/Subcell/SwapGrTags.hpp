// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Swaps the inactive and active GR variables.
 *
 * The values on the subcells are at the cell-centers.
 *
 * It should be possible to reduce memory usage by deallocating the GR variables
 * on the DG grid when switching to subcell. However, the opposite case is not
 * true since the GR variables are needed on the subcells if a neighbor is using
 * subcell in order to compute the neighbor's fluxes.
 *
 * \note The `active_grid` is the grid we are swapping to, which may be the same
 * as the current grid. On output the `active_gr_vars` will match the grid that
 * `active_grid` is. This mutator is a no-op if they matched on input.
 */
struct SwapGrTags {
  using return_tags = tmpl::list<typename System::spacetime_variables_tag,
                                 evolution::dg::subcell::Tags::Inactive<
                                     typename System::spacetime_variables_tag>>;
  using argument_tags =
      tmpl::list<::domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::ActiveGrid>;

  static void apply(
      gsl::not_null<
          Variables<typename System::spacetime_variables_tag::tags_list>*>
          active_gr_vars,
      gsl::not_null<typename evolution::dg::subcell::Tags::Inactive<
          typename System::spacetime_variables_tag>::type*>
          inactive_gr_vars,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      evolution::dg::subcell::ActiveGrid active_grid) noexcept;
};
}  // namespace grmhd::ValenciaDivClean::subcell
