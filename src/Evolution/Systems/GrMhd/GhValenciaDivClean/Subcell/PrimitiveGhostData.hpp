// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename T>
class Variables;
/// \endcond

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief Computes the rest mass density \f$\rho\f$, pressure \f$p\f$,
 * Lorentz factor times the spatial velocity \f$W v^i\f$, magnetic field
 * \f$B^i\f$, the divergence cleaning field \f$\Phi\f$, and the generalized
 * harmonic evolved variables \f$g_{ab}\f$, \f$\Phi_{iab}\f$ and \f$\Pi_{ab}\f$
 * on the subcells so they can be used for reconstruction.
 *
 * The computation copies the data from the primitive variables to a new
 * Variables and computes \f$W v^i\f$. In the future we will likely want to
 * elide this copy but that requires support from the actions.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 *
 * \note Only called on elements using FD.
 */
class PrimitiveGhostDataOnSubcells {
 private:
  using tags_for_reconstruction =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 gr::Tags::SpacetimeMetric<3>,
                 GeneralizedHarmonic::Tags::Phi<3>,
                 GeneralizedHarmonic::Tags::Pi<3>>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                 gr::Tags::SpacetimeMetric<3>,
                 GeneralizedHarmonic::Tags::Phi<3>,
                 GeneralizedHarmonic::Tags::Pi<3>>;

  static Variables<tags_for_reconstruction> apply(
      const Variables<hydro::grmhd_tags<DataVector>>& prims,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& pi);
};

/*!
 * \brief Projects the rest mass density \f$\rho\f$, pressure \f$p\f$, Lorentz
 * factor times the spatial velocity \f$W v^i\f$, magnetic field \f$B^i\f$, and
 * the divergence cleaning field \f$\Phi\f$ so they can be sent to neighbors
 * for subcell reconstruction.
 *
 * The computation copies the data from the primitive variables to a new
 * Variables and computes \f$W v^i\f$, then does the projection. In the future
 * we will likely want to elide this copy but that requires support from the
 * actions.
 *
 * This mutator is passed what `Metavars::SubcellOptions::GhostDataToSlice` must
 * be set to.
 *
 * \note We are ultimately projecting the primitive variables rather than
 * computing them on the subcells. This introduces truncation level errors, but
 * from tests so far this seems to be fine and is what is done with local time
 * stepping ADER-DG.
 */
class PrimitiveGhostDataToSlice {
 private:
  using tags_for_reconstruction =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 gr::Tags::SpacetimeMetric<3>,
                 GeneralizedHarmonic::Tags::Phi<3>,
                 GeneralizedHarmonic::Tags::Pi<3>>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                 gr::Tags::SpacetimeMetric<3>,
                 GeneralizedHarmonic::Tags::Phi<3>,
                 GeneralizedHarmonic::Tags::Pi<3>, domain::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::Mesh<3>>;

  static Variables<tags_for_reconstruction> apply(
      const Variables<hydro::grmhd_tags<DataVector>>& prims,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh);
};
}  // namespace grmhd::GhValenciaDivClean::subcell
