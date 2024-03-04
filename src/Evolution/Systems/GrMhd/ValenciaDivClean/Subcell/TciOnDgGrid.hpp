// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
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
namespace evolution::dg::subcell {
struct RdmpTciData;
class SubcellOptions;
}  // namespace evolution::dg::subcell
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
 * <table>
 * <caption>List of checks</caption>
 * <tr><th> Description <th> TCI status
 *
 * <tr><td> if \f$\min(\tilde{D}^{n+1}/\textrm{avg}(\sqrt{\gamma^{n}}))\f$
 * is less than `tci_options.minimum_rest_mass_density_times_lorentz_factor`
 * or if \f$\min(\tilde{Y}_e^{n+1}/\textrm{avg}(\sqrt{\gamma^{n}}))\f$ is less
 * than `tci_options.minimum_rest_mass_density_times_lorentz_factor` times
 * `tci_options.minimum_ye`, we have a negative (or extremely small) density or
 * electron fraction and the cell is troubled. Note that if this `tci_option` is
 * approximately equal to or larger than the `atmosphere_density`, the
 * atmosphere will be flagged as troubled.
 * <td> `-1`
 *
 * <tr><td> if \f$\tilde{\tau}\f$ is less than `tci_options.minimum_tilde_tau`
 * then we have a negative (or extremely small) energy and the cell is troubled.
 * <td> `-2`
 *
 * <tr><td> if \f$\max(\tilde{D}^{n+1}/(\sqrt{\gamma^n}W^n))\f$ and
 * \f$\max(\rho^n)\f$ are less than `tci_options.atmosphere_density` then the
 * entire DG element is in atmosphere and it is _not_ troubled.
 * <td> `0`
 *
 * <tr><td> if
 * \f$(\tilde{B}^{n+1})^2>2\sqrt{\gamma^n}(1-\epsilon_B)\tilde{\tau}^{n+1}\f$ at
 * any grid point, then the cell is troubled.
 * <td> `-3`
 *
 * <tr><td> attempt a primitive recovery using the `RecoveryScheme` from the
 * template parameter. The cell is marked as troubled if the primitive recovery
 * fails at any grid point.
 * <td> `-4`
 *
 * <tr><td> if \f$\max(\rho^{n+1})\f$ is below `tci_options.atmosphere_density`
 * then the cell is in atmosphere and not marked as troubled. Note that the
 * magnetic field is still freely evolved.
 * <td> `0`
 *
 * <tr><td> apply the Persson TCI to \f$\tilde{D}^{n+1}\f$
 * <td> `-5`
 *
 * <tr><td> apply the Persson TCI to \f$\tilde{Y}_e^{n+1}\f$
 * <td> `-6`
 *
 * <tr><td> apply the Persson TCI to pressure \f$p^{n+1}\f$
 * <td> `-7`
 *
 * <tr><td> apply the Persson TCI to the magnitude of \f$\tilde{B}^{n+1}\f$ if
 * its magnitude is greater than `tci_options.magnetic_field_cutoff`
 * <td> `-8`
 *
 * <tr><td> apply the RDMP TCI to `TildeD`
 * <td> `-9`
 *
 * <tr><td> apply the RDMP TCI to `TildeTau`
 * <td> `-10`
 *
 * <tr><td> apply the RDMP TCI to `TildeB`
 * <td> `-11`
 *
 * </table>
 *
 * If the cell is not flagged as troubled then the primitives are computed at
 * time level `n+1`.
 *
 * The second column of the table above denotes the value of an integer stored
 * as the first element of the returned `std::tuple`, which indicates the
 * particular kind of check that failed. For example, if the fifth check
 * (primitive recovery) fails and cell is marked as troubled, an integer with
 * value `-4` is stored in the first slot of the returned tuple. Note that this
 * integer is marking only the _first_ check to fail, since checks are done in a
 * particular sequence as listed above. If all checks are passed and cell is not
 * troubled, it is returned with the value `0`.
 *
 * \note We adopt negative integers to mark TCI status from DG grid returned by
 * TciOnDgGrid class. Positive integers are used for TCIs on FD grid; see
 * TciOnFdGrid and its documentation.
 *
 */
template <typename RecoveryScheme>
class TciOnDgGrid {
 public:
  using return_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>>;
  using argument_tags = tmpl::list<
      grmhd::ValenciaDivClean::Tags::TildeD,
      grmhd::ValenciaDivClean::Tags::TildeYe,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeS<>,
      grmhd::ValenciaDivClean::Tags::TildeB<>,
      grmhd::ValenciaDivClean::Tags::TildePhi,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      hydro::Tags::GrmhdEquationOfState, domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::DataForRdmpTci, Tags::TciOptions,
      evolution::dg::subcell::Tags::SubcellOptions<3>,
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>;

  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> dg_prim_vars,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
      const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const EquationsOfState::EquationOfState<true, 3>& eos,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const TciOptions& tci_options,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options,
      double persson_exponent, bool element_stays_on_dg);
};
}  // namespace grmhd::ValenciaDivClean::subcell
