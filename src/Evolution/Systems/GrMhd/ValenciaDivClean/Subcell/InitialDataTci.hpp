// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
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
/*!
 * \brief The troubled-cell indicator run on DG initial data to see if we need
 * to switch to subcell.
 *
 * The following checks are done in the order they are listed:
 *
 * - if `TildeD` (`TildeTau`) on the DG or subcell grid (projected from the DG
 *   grid, not initialized to the initial data) is less than
 *   `tci_options.minimum_rest_mass_density_times_lorentz_factor`
 *   (`tci_options.minimum_tilde_tau`), then the element is flagged as troubled.
 * - apply the two-mesh relaxed discrete maximum principle TCI
 * - apply the Persson TCI to \f$\tilde{D}\f$ and \f$\tilde{\tau}\f$.
 * - apply the Persson TCI to the magnitude of \f$\tilde{B}\f$ if its magnitude
 *   on the DG grid is greater than `tci_options.magnetic_field_cutoff`.
 */
struct DgInitialDataTci {
 private:
  template <typename Tag>
  using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

 public:
  using argument_tags = tmpl::list<domain::Tags::Mesh<3>, Tags::TciOptions>;

  static bool apply(
      const Variables<tmpl::list<
          ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeTau,
          ValenciaDivClean::Tags::TildeS<>, ValenciaDivClean::Tags::TildeB<>,
          ValenciaDivClean::Tags::TildePhi>>& dg_vars,
      const Variables<tmpl::list<Inactive<ValenciaDivClean::Tags::TildeD>,
                                 Inactive<ValenciaDivClean::Tags::TildeTau>,
                                 Inactive<ValenciaDivClean::Tags::TildeS<>>,
                                 Inactive<ValenciaDivClean::Tags::TildeB<>>,
                                 Inactive<ValenciaDivClean::Tags::TildePhi>>>&
          subcell_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<3>& dg_mesh, const TciOptions& tci_options);
};
}  // namespace grmhd::ValenciaDivClean::subcell
