// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {

namespace detail {
void fluxes_impl(
    gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_psi_flux,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

    // Temporaries
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_electric_field_one_form,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_magnetic_field_one_form,

    // extra args
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric);
}  // namespace detail

/*!
 * \brief Compute the fluxes of the GRFFE system with divergence cleaning.
 *
 * \f{align*}
 *  F^j(\tilde{E}^i) & = -\beta^j\tilde{E}^i + \alpha (\gamma^{ij}\tilde{\psi}
 *      - \epsilon^{ijk}_{(3)}\tilde{B}_k) \\
 *  F^j(\tilde{B}^i) & = -\beta^j\tilde{B}^i + \alpha (\gamma^{ij}\tilde{\phi}
 *      + \epsilon^{ijk}_{(3)}\tilde{E}_k) \\
 *  F^j(\tilde{\psi}) & = -\beta^j \tilde{\psi} + \alpha \tilde{E}^j \\
 *  F^j(\tilde{\phi}) & = -\beta^j \tilde{\phi} + \alpha \tilde{B}^j \\
 *  F^j(\tilde{q}) & = \alpha \sqrt{\gamma}J^j - \beta^j \tilde{q}
 * \f}
 *
 * where the conserved variables \f$\tilde{E}^i, \tilde{B}^i, \tilde{\psi},
 * \tilde{\phi}, \tilde{q}\f$ are densitized electric field, magnetic field,
 * electric divergence cleaning field, magnetic divergence cleaning field, and
 * electric charge density. \f$J^i\f$ is the spatial electric current density.
 *
 * \f$\epsilon_{(3)}^{ijk}\f$ is the spatial Levi-Civita tensor defined as
 *
 * \f{align*}
 *  \epsilon_{(3)}^{ijk} \equiv n_\mu \epsilon^{\mu ijk}
 *   = -\frac{1}{\sqrt{-g}} n_\mu [\mu ijk] = \frac{1}{\sqrt{\gamma}} [ijk]
 * \f}
 *
 * where $\epsilon^{\mu\nu\rho\sigma}$ is the Levi-Civita tensor, \f$g\f$ is the
 * determinant of spacetime metric, \f$\gamma\f$ is the determinant of spatial
 * metric, \f$n^\mu\f$ is the normal to spatial hypersurface. Also,
 * \f$[abcd]\f$ and \f$[ijk]\f$ are the usual antisymmetric _symbols_ (which
 * only have the value \f$\pm 1\f$) with 4 and 3 indices, respectively, with the
 * sign \f$[0123] = [123] = +1\f$. Note that
 *
 * \f{align*}
 *  \epsilon_{\mu\nu\rho\sigma} = \sqrt{-g} \, [\mu\nu\rho\sigma]
 *  , \quad \text{and} \quad
 *  \epsilon^{\mu\nu\rho\sigma} = -\frac{1}{\sqrt{-g}} [\mu\nu\rho\sigma]
 * \f}
 *
 */
struct Fluxes {
  using return_tags =
      tmpl::list<::Tags::Flux<Tags::TildeE, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<Tags::TildeB, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<Tags::TildePsi, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<Tags::TildePhi, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<Tags::TildeQ, tmpl::size_t<3>, Frame::Inertial>>;

  using argument_tags =
      tmpl::list<Tags::TildeE, Tags::TildeB, Tags::TildePsi, Tags::TildePhi,
                 Tags::TildeQ, Tags::TildeJ, gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>>;

  static void apply(
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_psi_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
      const Scalar<DataVector>& tilde_q,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric);
};
}  // namespace ForceFree
