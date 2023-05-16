// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::GhValenciaDivClean::subcell {
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
 * - apply the Persson TCI to \f$\tilde{D}\f$ and \f$\tilde{\tau}\f$.
 * - apply the Persson TCI to the magnitude of \f$\tilde{B}\f$ if its magnitude
 *   on the DG grid is greater than `tci_options.magnetic_field_cutoff`.
 * - apply the two-mesh relaxed discrete maximum principle TCI
 */
struct DgInitialDataTci {
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 ValenciaDivClean::subcell::Tags::TciOptions>;

  static std::tuple<bool, evolution::dg::subcell::RdmpTciData> apply(
      const Variables<tmpl::list<
          gr::Tags::SpacetimeMetric<DataVector, 3>, gh::Tags::Pi<DataVector, 3>,
          gh::Tags::Phi<DataVector, 3>, ValenciaDivClean::Tags::TildeD,
          ValenciaDivClean::Tags::TildeYe, ValenciaDivClean::Tags::TildeTau,
          ValenciaDivClean::Tags::TildeS<>, ValenciaDivClean::Tags::TildeB<>,
          ValenciaDivClean::Tags::TildePhi>>& dg_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const ValenciaDivClean::subcell::TciOptions& tci_options);
};
}  // namespace grmhd::GhValenciaDivClean::subcell
