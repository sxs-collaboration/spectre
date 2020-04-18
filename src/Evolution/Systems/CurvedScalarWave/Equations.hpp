// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace CurvedScalarWave {
/*!
 * \brief Compute the time derivative of the evolved variables of the
 * first-order scalar wave system on a curved background.
 *
 * The evolution equations for the first-order scalar wave system are given by
 * \cite Holst2004wt :
 *
 * \f{align}
 * \partial_t\Psi = & (1 + \gamma_1) \beta^k \partial_k \Psi - \alpha \Pi -
 * \gamma_1 \beta^k \Phi_k \\
 *
 * \partial_t\Pi = & - \alpha \gamma^{ij}\partial_i\Phi_j + \beta^k \partial_k
 * \Pi + \gamma_1 \gamma_2 \beta^k \partial_k \Psi  + \alpha \Gamma^i -
 * \gamma^{ij} \Phi_i \partial_j \alpha
 *  + \alpha K \Pi - \gamma_1  \gamma_2 \beta^k \Phi_k\\
 *
 * \partial_t\Phi_i = & - \alpha \partial_i \Pi  + \beta^k \partial_k \Phi +
 * \gamma_2 \alpha \partial_i \Psi - \Pi
 * \partial_i \alpha + \Phi_k \partial_i \beta^j - \gamma_2 \alpha \Phi_i\\
 * \f}
 *
 * where \f$\Psi\f$ is the scalar field, \f$\Pi\f$ is the
 * conjugate momentum to \f$\Psi\f$, \f$\Phi_i=\partial_i\Psi\f$ is an
 * auxiliary variable, \f$\alpha\f$ is the lapse, \f$\beta^k\f$ is the shift,
 * \f$ \gamma_{ij} \f$ is the spatial metric, \f$ K \f$ is the trace of the
 * extrinsic curvature, and \f$ \Gamma^i \f$ is the trace of the Christoffel
 * symbol of the second kind. \f$\gamma_1, \gamma_2\f$ are constraint damping
 * parameters.
 */
template <size_t Dim>
struct ComputeDuDt {
 public:
  template <template <class> class StepPrefix>
  using return_tags = tmpl::list<db::add_tag_prefix<StepPrefix, Pi>,
                                 db::add_tag_prefix<StepPrefix, Phi<Dim>>,
                                 db::add_tag_prefix<StepPrefix, Psi>>;

  using argument_tags = tmpl::list<
      Pi, Phi<Dim>, ::Tags::deriv<Psi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Pi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::TraceSpatialChristoffelSecondKind<Dim, Frame::Inertial,
                                                  DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>, Tags::ConstraintGamma1,
      Tags::ConstraintGamma2>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> dt_psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim>& phi,
      const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::i<DataVector, Dim>& d_pi,
      const tnsr::ij<DataVector, Dim>& d_phi, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::i<DataVector, Dim>& deriv_lapse,
      const tnsr::iJ<DataVector, Dim>& deriv_shift,
      const tnsr::II<DataVector, Dim>& upper_spatial_metric,
      const tnsr::I<DataVector, Dim>& trace_spatial_christoffel,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2) noexcept;
};

/*!
 * \brief Compute fluxes for the scalar-wave system in curved spacetime.
 *
 * \details The expressions for fluxes is obtained from \cite Holst2004wt by
 * taking the principal part of equations 15, 23, and 24, and replacing
 * derivatives \f$ \partial_k \f$ with the unit normal \f$ n_k \f$ (c.f. Gauss'
 * theorem). This gives:
 *
 * \f{align*}
 * F(\psi) &= -(1 + \gamma_1) \beta^k n_k \psi \\
 * F(\Pi) &= - \beta^k n_k \Pi + \alpha g^{ki}n_k \Phi_{i}
 *           - \gamma_1 \gamma_2 \beta^k n_k \Psi  \\
 * F(\Phi_{i}) &= - \beta^k n_k \Phi_{i} + \alpha n_i \Pi - \gamma_2 \alpha n_i
 * \psi \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\Pi\f$ is its conjugate
 * momentum, \f$ \Phi_{i} \f$ is an auxiliary field defined as the spatial
 * derivative of \f$\psi\f$, \f$\alpha\f$ is the lapse, \f$ \beta^k \f$ is the
 * shift, \f$ g^{ki} \f$ is the inverse spatial metric, and \f$ \gamma_1,
 * \gamma_2\f$ are constraint damping parameters. Note that the last term in
 * \f$F(\Pi)\f$ will be identically zero, as it is necessary to set
 * \f$\gamma_1\gamma_2=0\f$ in order to make this first-order formulation of the
 * curved scalar wave system symmetric hyperbolic.
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
 public:
  using argument_tags =
      tmpl::list<Pi, Phi<Dim>, Psi, Tags::ConstraintGamma1,
                 Tags::ConstraintGamma2, gr::Tags::Lapse<>,
                 gr::Tags::Shift<Dim>, gr::Tags::InverseSpatialMetric<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
      gsl::not_null<tnsr::i<DataVector, Dim>*> phi_normal_dot_flux,
      gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
      const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
      const Scalar<DataVector>& psi, const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
      const tnsr::i<DataVector, Dim>& interface_unit_normal) noexcept;
};

/*!
 * \ingroup NumericalFluxesGroup
 * \brief Computes the upwind flux for scalar-waves in curved spacetime.
 *
 * \details The upwind flux is given by \cite Teukolsky2015ega :
 *
 * \f[
 * G = S\left(\Lambda^+ S^{-1} U^{\rm int}
 *           + \Lambda^- S^{-1} U^{\rm ext}\right)
 * = S \left(\Lambda^+ \hat{U}^{\rm int}
 *           + \Lambda^- \hat{U}^{\rm ext}\right),
 * \f]
 *
 * where
 *
 *  - \f$G\f$ is the numerical upwind flux dotted with the interface normal;
 *  - \f$U\f$ is a vector of all evolved variables;
 *  - \f$S\f$ is a matrix whose columns are the eigenvectors of the
 *       characteristic matrix for the evolution system. It maps the
 *       evolved variables to characteristic variables \f$\hat{U}\f$, s.t.
 *       \f$\hat{U} := S^{-1}\cdot U\f$; and
 *  - \f$\Lambda^\pm\f$ are diagonal matrices containing
 *       positive / negative eigenvalues of the same matrix as its elements.
 *
 * The superscripts \f${\rm int}\f$ and \f${\rm ext}\f$ on \f$U\f$ indicate
 * that the corresponding set of variables at the element interface have been
 * taken from the _interior_ or _exterior_ of the element. Exterior of the
 * element is naturally the interior of its neighboring element. Therefore,
 * \f$\hat{U}^{\rm int}:= S^{-1}U^{\rm int}\f$ are the characteristic variables
 * at the element interface computed using evolved variables from the interior
 * of the element, and \f$\hat{U}^{\rm ext}= S^{-1}U^{\rm ext}\f$ are the
 * same computed from evolved variables taken from the element exterior, i.e.
 * the neighboring element.
 * The sign of characteristic speeds indicates the direction of propagation of
 * the corresponding characteristic field with respect to the interface normal
 * that the field has been computed along (with negative speeds indicating
 * incoming characteristics, and with positive speeds indicating outgoing
 * characteristics). Therefore, \f$\Lambda^+\f$ contains characterstic speeds
 * for variables that are outgoing from the element at the interface, and
 * \f$\Lambda^-\f$ contains characteristic speeds for variables that are
 * incoming to the element at the interface. An ambiguity naturally arises
 * as to which set of evolved variables (''interior'' or ''exterior'') to use
 * when computing these speeds. We compute both and use their average, i.e.
 * \f$\lambda^{\rm avg} = (\lambda^{\rm int} + \lambda^{\rm ext})/2\f$, to
 * populate \f$\Lambda^\pm\f$.
 *
 *
 * This function computes the upwind flux as follows:
 *  -# Computes internal and external characteristic variables and speeds using
 * evolved variables from
 *     both the element interior and exterior.
 *  -# Computes the average characteristic speeds and constructs
 *      \f$\Lambda^{\pm}\f$.
 *  -# Computes the upwind flux as a weigted sum of external and internal
 *     characteristic variables
 *
 * \f[
 * G = S\left(w_{\rm ext} \hat{U}^{\rm ext}
 *       + w_{\rm int} \hat{U}^{\rm int}\right),
 * \f]
 *
 * with weights \f$w_{\rm ext} = \Theta(-\Lambda)\cdot\Lambda\f$, and
 * \f$w_{\rm int} = \Theta(\Lambda)\cdot\Lambda\f$, where \f$\Theta\f$ is the
 * step function centered at zero, \f$\Lambda = \Lambda^+ + \Lambda^-\f$, and
 * the dot operator \f$(\cdot)\f$ indicates an element-wise product.
 *
 * \warning With the averaging of characteristic speeds, this flux does not
 * satisfy the generalized Rankine-Hugoniot conditions
 * \f${G}^{\rm int}(\hat{U}^{\rm int}, \hat{U}^{\rm ext})
 * = - {G}^{\rm ext}(\hat{U}^{\rm ext}, \hat{U}^{\rm int})\f$,
 * which enforces that the flux leaving the element is equal to the flux
 * entering the neighboring element, and vice versa. This condition is
 * important for a well-balanced scheme, and so please use this flux with
 * caution.
 */
template <size_t Dim>
struct UpwindFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the curved scalar-wave upwind flux."};

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using variables_tags = tmpl::list<Pi, Phi<Dim>, Psi>;

  // This is the data needed to compute the numerical flux.
  // `dg::SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_field_tags = tmpl::list<
      Pi, Phi<Dim>, Psi, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2,
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;
  using package_extra_tags = tmpl::list<>;

  // These tags on the interface of the element are passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags = tmpl::list<
      Pi, Phi<Dim>, Psi, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2,
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // Following the not-null pointer to packaged_data, this function expects as
  // arguments the databox types of the `argument_tags`.
  void package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_psi,
      gsl::not_null<Scalar<DataVector>*> packaged_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
      gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          packaged_inverse_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_gamma1,
      gsl::not_null<Scalar<DataVector>*> packaged_gamma2,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_interface_unit_normal,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const Scalar<DataVector>& psi, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // The arguments are first the system::variables_tag::tags_list wrapped in
  // Tags::NormalDotNumericalFlux as not-null pointers to write the results
  // into, then the package_tags on the interior side of the mortar followed by
  // the package_tags on the exterior side.
  void operator()(
      gsl::not_null<Scalar<DataVector>*> pi_normal_dot_numerical_flux,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_normal_dot_numerical_flux,
      gsl::not_null<Scalar<DataVector>*> psi_normal_dot_numerical_flux,
      const Scalar<DataVector>& pi_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi_int,
      const Scalar<DataVector>& psi_int, const Scalar<DataVector>& lapse_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
      const tnsr::II<DataVector, Dim, Frame::Inertial>&
          inverse_spatial_metric_int,
      const Scalar<DataVector>& gamma1_int,
      const Scalar<DataVector>& gamma2_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_int,
      const Scalar<DataVector>& pi_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi_ext,
      const Scalar<DataVector>& psi_ext, const Scalar<DataVector>& lapse_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
      const tnsr::II<DataVector, Dim, Frame::Inertial>&
          inverse_spatial_metric_ext,
      const Scalar<DataVector>& gamma1_ext,
      const Scalar<DataVector>& gamma2_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_ext) const noexcept;
};
}  // namespace CurvedScalarWave
