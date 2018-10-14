// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  //  IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/Hydro/Tags.hpp"
// IWYU pragma: no_include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
class DataVector;

namespace Tags {
template <typename>
struct Source;
}  // namespace Tags
/// \endcond

namespace grmhd {
namespace ValenciaDivClean {

/*!
 * \brief Compute the source terms for the flux-conservative Valencia
 * formulation of GRMHD with divergence cleaning.
 *
 *
 * A flux-conservative system has the generic form:
 * \f[
 * \partial_t U_i + \partial_m F^m(U_i) = S(U_i)
 * \f]
 *
 * where \f$F^a()\f$ denotes the flux of a conserved variable \f$U_i\f$ and
 * \f$S()\f$ denotes the source term for the conserved variable.
 *
 * For the Valencia formulation:
 * \f{align*}
 * S({\tilde D}) = & 0\\
 * S({\tilde S}_i) = & \frac{1}{2} \alpha {\tilde S}^{mn} \partial_i \gamma_{mn}
 * + {\tilde S}_m \partial_i \beta^m - ({\tilde D} + {\tilde \tau}) \partial_i
 * \alpha \\
 * S({\tilde \tau}) = & \alpha {\tilde S}^{mn} K_{mn} - {\tilde S}^m \partial_m
 * \alpha \\
 * S({\tilde B}^i) = & -{\tilde B}^m \partial_m \beta^i + {\tilde \Phi}
 * \gamma^{im} \partial_m \alpha + \alpha {\tilde \Phi} \left( \frac{1}{2}
 * \gamma^{il} \gamma^{jk} - \gamma^{ij} \gamma^{lk} \right) \partial_l
 * \gamma_{jk} \\ S({\tilde \Phi}) = & {\tilde B}^k \partial_k \alpha - \alpha K
 * {\tilde \Phi}
 * - \alpha \kappa {\tilde \Phi}
 * \f}
 *
 * where
 * \f{align*}
 * {\tilde S}^i = & {\tilde S}_m \gamma^{im} \\
 * {\tilde S}^{ij} = & \sqrt{\gamma} \left[ \left(h \rho W^2 + B^n B_n \right)
 * v^i v^j + \left(p + p_m \right) \gamma^{ij} - B^n v_n \left( B^i v^j + B^j
 * v^i \right) - \frac{B^i B^j}{W^2} \right]
 * \f}
 *
 * where \f${\tilde D}\f$, \f${\tilde S}_i\f$, \f${\tilde \tau}\f$, \f${\tilde
 * B}^i\f$, and \f${\tilde \Phi}\f$ are a generalized mass-energy density,
 * momentum density, specific internal energy density, magnetic field, and
 * divergence cleaning field.  Furthermore, \f$\gamma\f$ is the determinant of
 * the spatial metric \f$\gamma_{ij}\f$, \f$\rho\f$ is the rest mass density,
 * \f$W\f$ is the Lorentz factor, \f$h\f$ is the specific enthalpy, \f$v^i\f$ is
 * the spatial velocity, \f$B^k\f$ is the magnetic field, \f$p\f$ is the
 * pressure, \f$p_m = \frac{1}{2} \left[ \left( B^n v_n \right)^2 + B^n B_n /
 * W^2 \right]\f$ is the magnetic pressure, \f$\alpha\f$ is the lapse,
 * \f$\beta^i\f$ is the shift, \f$K\f$ is the trace of the extrinsic curvature
 * \f$K_{ij}\f$, and \f$\kappa\f$ is a damping parameter that damps violations
 * of the divergence-free (no-monopole) condition \f$\Phi = \partial_i {\tilde
 * B}^i = 0\f$ .
 */
struct ComputeSources {
  using return_tags =
      tmpl::list<::Tags::Source<grmhd::ValenciaDivClean::Tags::TildeTau>,
                 ::Tags::Source<grmhd::ValenciaDivClean::Tags::TildeS<>>,
                 ::Tags::Source<grmhd::ValenciaDivClean::Tags::TildeB<>>,
                 ::Tags::Source<grmhd::ValenciaDivClean::Tags::TildePhi>>;

  using argument_tags = tmpl::list<
      grmhd::ValenciaDivClean::Tags::TildeD,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeS<>,
      grmhd::ValenciaDivClean::Tags::TildeB<>,
      grmhd::ValenciaDivClean::Tags::TildePhi,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::MagneticField<DataVector, 3>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificEnthalpy<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>, hydro::Tags::Pressure<DataVector>,
      gr::Tags::Lapse<>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::SpatialMetric<3>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<3>, gr::Tags::SqrtDetSpatialMetric<>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> source_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_b,
      gsl::not_null<Scalar<DataVector>*> source_tilde_phi,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& specific_enthalpy,
      const Scalar<DataVector>& lorentz_factor,
      const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
      double constraint_damping_parameter) noexcept;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
