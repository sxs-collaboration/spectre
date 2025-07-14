// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to general relativistic magnetohydrodynamics (GRMHD)
namespace grmhd {
/// Items related to the Valencia formulation of ideal GRMHD with divergence
/// cleaning coupled with electron fraction.
///
/// References:
/// - Numerical 3+1 General Relativistic Magnetohydrodynamics: A Local
/// Characteristic Approach \cite Anton2006
/// - GRHydro: a new open-source general-relativistic magnetohydrodynamics code
/// for the Einstein toolkit \cite Moesta2014
/// - Black hole-neutron star mergers with a hot nuclear equation of state:
/// outflow and neutrino-cooled disk for a low-mass, high-spin case
/// \cite Deaton2013
///
namespace ValenciaDivClean {

/*!
 * \brief Ideal general relativistic magnetohydrodynamics (GRMHD) system with
 * divergence cleaning coupled with electron fraction.
 *
 * We adopt the standard 3+1 form of metric
 * \f{align*}
 *  ds^2 = -\alpha^2 dt^2 + \gamma_{ij}(dx^i + \beta^i dt)(dx^j + \beta^j dt)
 * \f}
 * where \f$\alpha\f$ is the lapse, \f$\beta^i\f$ is the shift vector, and
 * \f$\gamma_{ij}\f$ is the spatial metric.
 *
 * Primitive variables of the system are
 *  - rest mass density \f$\rho\f$
 *  - electron fraction \f$Y_e\f$
 *  - the spatial velocity \f{align*} v^i =
 * \frac{1}{\alpha}\left(\frac{u^i}{u^0} + \beta^i\right) \f} measured by the
 * Eulerian observer
 *  - specific internal energy density \f$\epsilon\f$
 *  - magnetic field \f$B^i = -^*F^{ia}n_a = -\alpha (^*F^{0i})\f$
 *  - the divergence cleaning field \f$\Phi\f$
 *
 * with corresponding derived physical quantities which frequently appear in
 * equations:
 * - The transport velocity \f$v^i_{tr} = \alpha v^i - \beta^i\f$
 * - The Lorentz factor \f{align*} W = -u^an_a = \alpha u^0 =
 * \frac{1}{\sqrt{1-\gamma_{ij}v^iv^j}} = \sqrt{1+\gamma^{ij}u_iu_j} =
 * \sqrt{1+\gamma^{ij}W^2v_iv_j} \f}
 * - The specific enthalpy \f$h = 1 + \epsilon + p/\rho\f$ where \f$p\f$ is the
 * pressure specified by a particular equation of state (EoS)
 * - The comoving magnetic field \f$b^b = -^*F^{ba} u_a\f$ in the ideal MHD
 * limit
 * \f{align*}
 *  b^0 & = W B^i v_i / \alpha \\
 *  b^i & = (B^i + \alpha b^0 u^i) / W \\
 *  b^2 & = b^ab_a = B^2/W^2 + (B^iv_i)^2
 * \f}
 * - Augmented enthalpy density \f$(\rho h)^* = \rho h + b^2\f$ and augmented
 * pressure \f$p^* = p + b^2/2\f$ which include contributions from magnetic
 * field
 *
 * \note We are using the electromagnetic variables with the scaling convention
 * that the factor \f$4\pi\f$ does not appear in Maxwell equations and the
 * stress-energy tensor of the EM fields (geometrized Heaviside-Lorentz units).
 * To recover the physical value of magnetic field in the usual CGS Gaussian
 * unit, the conversion factor is
 * \f{align*}
 *  \sqrt{4\pi}\frac{c^4}{G^{3/2}M_\odot} \approx 8.35 \times 10^{19}
 *  \,\text{Gauss}
 * \f}
 * For example, magnetic field $10^{-5}$ with the code unit corresponds to the
 * $8.35\times 10^{14}\,\text{G}$ in the CGS Gaussian unit. See also
 * documentation of hydro::units::cgs::gauss_unit for details.
 *
 * The GRMHD equations can be written in a flux-balanced form
 * \f[
 *  \partial_t U+ \partial_i F^i(U) = S(U).
 * \f]
 *
 * Evolved (conserved) variables \f$U\f$ are
 * \f{align*}{
 * U = \sqrt{\gamma}\left[\,\begin{matrix}
 *      D \\
 *      D Y_e \\
 *      S_j \\
 *      \tau \\
 *      B^j \\
 *      \Phi \\
 * \end{matrix}\,\right] \equiv \left[\,\,\begin{matrix}
 *      \tilde{D} \\
 *      \tilde{Y_e} \\
 *      \tilde{S}_j \\
 *      \tilde{\tau} \\
 *      \tilde{B}^j \\
 *      \tilde{\Phi} \\
 * \end{matrix}\,\,\right] = \sqrt{\gamma} \left[\,\,\begin{matrix}
 *      \rho W \\
 *      \rho W Y_e \\
 *      (\rho h)^* W^2 v_j - \alpha b^0 b_j \\
 *      (\rho h)^* W^2 - p^* - (\alpha b^0)^2 - \rho W \\
 *      B^j \\
 *      \Phi \\
 * \end{matrix}\,\,\right]
 * \f}
 *
 * where \f${\tilde D}\f$, \f${\tilde Y}_e\f$,\f${\tilde S}_i\f$, \f${\tilde
 * \tau}\f$, \f${\tilde B}^i\f$, and \f${\tilde \Phi}\f$ are a generalized
 * mass-energy density, electron fraction, momentum density, specific internal
 * energy density, magnetic field, and divergence cleaning field. Also,
 * \f$\gamma\f$ is the determinant of the spatial metric \f$\gamma_{ij}\f$.
 *
 * Corresponding fluxes \f$F^i(U)\f$ are
 *
 * \f{align*}
 * F^i({\tilde D}) &= {\tilde D} v^i_{tr} \\
 * F^i({\tilde Y}_e) &= {\tilde Y}_e v^i_{tr} \\
 * F^i({\tilde S}_j) &= {\tilde S}_j v^i_{tr} + \alpha \sqrt{\gamma} p^*
 *      \delta^i_j - \alpha b_j \tilde{B}^i / W \\
 * F^i({\tilde \tau}) &= {\tilde \tau} v^i_{tr} + \alpha \sqrt{\gamma} p^* v^i
 *      - \alpha^2 b^0 \tilde{B}^i / W \\
 * F^i({\tilde B}^j) &= {\tilde B}^j v^i_{tr} - \alpha v^j {\tilde B}^i
 *      + \alpha \gamma^{ij} {\tilde \Phi} \\
 * F^i({\tilde \Phi}) &= \alpha {\tilde B^i} - {\tilde \Phi} \beta^i
 * \f}
 *
 * and source terms \f$S(U)\f$ are
 *
 * \f{align*}
 * S({\tilde D}) &= 0 \\
 * S({\tilde Y}_e) &= 0 \\
 * S({\tilde S}_j) &= \frac{\alpha}{2} {\tilde S}^{mn} \partial_j \gamma_{mn}
 *      + {\tilde S}_k \partial_j \beta^k - ({\tilde D}
 *      + {\tilde \tau}) \partial_j \alpha \\
 * S({\tilde \tau}) &= \alpha {\tilde S}^{mn} K_{mn}
 *      - {\tilde S}^m \partial_m \alpha \\
 * S({\tilde B}^j) &= -{\tilde B}^m \partial_m \beta^j
 *      + \Phi \partial_k (\alpha \sqrt{\gamma}\gamma^{jk}) \\
 * S({\tilde \Phi}) &= {\tilde B}^k \partial_k \alpha - \alpha K
 *      {\tilde \Phi} - \alpha \kappa {\tilde \Phi}
 * \f}
 *
 * with
 * \f{align*}
 * {\tilde S}^{ij} = & \sqrt{\gamma} \left[ (\rho h)^* W^2 v^i v^j + p^*
 *      \gamma^{ij} - \gamma^{ik}\gamma^{jl}b_kb_l \right]
 * \f}
 *
 * where \f$K\f$ is the trace of the extrinsic curvature \f$K_{ij}\f$, and
 * \f$\kappa\f$ is a damping parameter that damps violations of the
 * divergence-free (no-monopole) condition \f$\Phi = \partial_i {\tilde B}^i =
 * 0\f$.
 *
 * \note On the electron fraction side, the source term is currently set to
 * \f$S(\tilde{Y}_e) = 0\f$. Implementing the source term using neutrino scheme
 * is in progress (Last update : Oct 2022).
 *
 */
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = 3;

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection;

  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
                 Tags::TildeB<>, Tags::TildePhi>>;
  using flux_variables =
      tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
                 Tags::TildeB<>, Tags::TildePhi>;
  using non_conservative_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  using primitive_variables_tag =
      ::Tags::Variables<hydro::grmhd_tags<DataVector>>;
  using spacetime_variables_tag =
      ::Tags::Variables<gr::tags_for_hydro<volume_dim, DataVector>>;
  using flux_spacetime_variables_tag = ::Tags::Variables<
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>>>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms;

  using conservative_from_primitive = ConservativeFromPrimitive;
  template <typename OrderedListOfPrimitiveRecoverySchemes>
  using primitive_from_conservative =
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed;

  using inverse_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim>;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
