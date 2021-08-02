// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
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
 * \brief The troubled-cell indicator run on the DG grid to check if the
 * solution is admissible.
 *
 * We denote variables at the candidate solution's time level by a superscript
 * \f$n+1\f$ and at the time level where the solution is known to be admissible
 * by a superscript \f$n\f$.
 *
 * The following checks are done in the order they are listed:
 *
 * - if \f$\min(\tilde{D}^{n+1}/\sqrt{\gamma^{n}})\f$ is less than
 *   `tci_options.minimum_rest_mass_density_times_lorentz_factor` then we have a
 *   negative (or extremely small) density and the cell is troubled. Note that
 *   if this `tci_option` is approximately equal to or larger than the
 *   `atmosphere_density`, the atmosphere will be flagged as troubled.
 * - if \f$\max(\tilde{D}^{n+1}/(\sqrt{\gamma^n}W^n))\f$ and \f$\max(\rho^n)\f$
 *   are less than `tci_options.atmosphere_density` then the entire DG element
 *   is in atmosphere and it is _not_ troubled.
 * - if
 *   \f$(\tilde{B}^{n+1})^2>2\sqrt{\gamma^n}(1-\epsilon_B)\tilde{\tau}^{n+1}\f$
 *   at any grid point, then the cell is troubled
 * - attempt a primitive recovery using the `RecoveryScheme` from the template
 *   parameter. The cell is marked as troubled if the primitive recovery fails
 *   at any grid point.
 * - if \f$\max(\rho^{n+1})\f$ is below `tci_options.atmosphere_density` then
 *   the cell is in atmosphere and not marked as troubled. Note that the
 *   magnetic field is still freely evolved.
 * - apply the Persson TCI to \f$\tilde{D}^{n+1}\f$ and \f$\tilde{\tau}^{n+1}\f$
 *
 * If the cell is not flagged as troubled then the primitives are computed at
 * time level `n+1`.
 */
template <typename RecoveryScheme>
class TciOnDgGrid {
 public:
  using return_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>>;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::Inactive<
                     grmhd::ValenciaDivClean::Tags::TildeD>,
                 grmhd::ValenciaDivClean::Tags::TildeD,
                 grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeS<>,
                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                 grmhd::ValenciaDivClean::Tags::TildePhi,
                 gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>,
                 gr::Tags::SqrtDetSpatialMetric<>,
                 hydro::Tags::EquationOfStateBase, domain::Tags::Mesh<3>,
                 Tags::TciOptions>;

  template <size_t ThermodynamicDim>
  static bool apply(
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> dg_prim_vars,
      const Scalar<DataVector>& subcell_tilde_d,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const Mesh<3>& dg_mesh, const TciOptions& tci_options,
      double persson_exponent) noexcept;
};
}  // namespace grmhd::ValenciaDivClean::subcell
