// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace sgb {
struct Fluxes;
struct Sources;
}  // namespace sgb
/// \endcond

namespace sgb {

/*!
 * \brief Compute the fluxes $F^i=\left(\psi^{-4} \tilde{\gamma}^{ij}
 * -\alpha^{2} \beta^i \beta^j \right) \partial_j \Psi(x)$ for the scalar
 * equation in sGB gravity on a conformal metric $\tilde{\gamma}_{ij}$.
 */
void curved_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& conformal_factor,
                   const tnsr::i<DataVector, 3>& field_gradient);

/*!
 * \brief Compute the fluxes $F^i=\left(\psi^{-4} \tilde{\gamma}^{ij}
 * -\alpha^{2} \beta^i \beta^j \right) \partial_j \Psi(x)$ for the scalar
 * equation in sGB gravity on a conformal metric $\tilde{\gamma}_{ij}$ on a face
 * normal.
 */
void face_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                 const tnsr::II<DataVector, 3>& inv_conformal_metric,
                 const tnsr::I<DataVector, 3>& shift,
                 const Scalar<DataVector>& lapse,
                 const Scalar<DataVector>& conformal_factor,
                 const tnsr::i<DataVector, 3>& face_normal,
                 const Scalar<DataVector>& field);

/*!
 * \brief Adds the source terms arising from the $\Box \Psi$ term in the
 * equation of motion for the scalar field: $S=-\tilde{\Gamma}^i_{ij}F^j-F^j
 * \alpha^{-1} \partial_j \alpha - 6F^j \psi^{-1} \partial_j \psi$.*/
void add_curved_sources(gsl::not_null<Scalar<DataVector>*> source_for_field,
                        const tnsr::i<DataVector, 3>& christoffel_contracted,
                        const tnsr::I<DataVector, 3>& flux_for_field,
                        const tnsr::i<DataVector, 3>& deriv_lapse,
                        const Scalar<DataVector>& lapse,
                        const Scalar<DataVector>& conformal_factor,
                        const tnsr::i<DataVector, 3>& conformal_factor_deriv);

/*!
 * \brief Add the sGB coupling term $\mathcal{R} f'(\Psi)=2(E-B)(\epsilon_2 \Psi
 * + \epsilon_4 \Psi^3)$.
 */
void add_GB_terms(gsl::not_null<Scalar<DataVector>*> scalar_tensor_equation,
                  double eps2, double eps4,
                  const Scalar<DataVector>& weyl_electric,
                  const Scalar<DataVector>& weyl_magnetic,
                  const Scalar<DataVector>& field);

/*!
 * \brief Add sources arising from linearising the sGB coupling term.
 */
void add_linearized_GB_terms(
    gsl::not_null<Scalar<DataVector>*> linearized_scalar_tensor_equation,
    double eps2, double eps4, const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction);

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the scalar equation in sGB gravity on
 * a spatial metric \f$\gamma_{ij}\f$.
 */
struct Fluxes {
  using argument_tags = tmpl::list<
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      sgb::Tags::RolledOffShift, gr::Tags::Lapse<DataVector>,
      Xcts::Tags::ConformalFactor<DataVector>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                    const tnsr::II<DataVector, 3>& inv_conformal_metric,
                    const tnsr::I<DataVector, 3>& shift,
                    const Scalar<DataVector>& lapse,
                    const Scalar<DataVector>& conformal_factor,
                    const Scalar<DataVector>& field,
                    const tnsr::i<DataVector, 3>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                    const tnsr::II<DataVector, 3>& inv_conformal_metric,
                    const tnsr::I<DataVector, 3>& shift,
                    const Scalar<DataVector>& lapse,
                    const Scalar<DataVector>& conformal_factor,
                    const tnsr::i<DataVector, 3>& face_normal,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
                    const Scalar<DataVector>& field);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the scalar equation in sGB gravity on a
 * spatial metric \f$\gamma_{ij}\f$.
 */
struct Sources {
  using argument_tags = tmpl::list<
      Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                 Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      gr::Tags::Lapse<DataVector>, Xcts::Tags::ConformalFactor<DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      Tags::Epsilon2, Tags::Epsilon4, gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_field,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, 3>& deriv_lapse,
      const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::i<DataVector, 3>& conformal_factor_deriv, const double& eps2,
      const double& eps4, const Scalar<DataVector>& weyl_electric,
      const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
      const tnsr::I<DataVector, 3>& field_flux);
};

/*!
 * \brief Add the linearised sources \f$S_A\f$ for the scalar equation in sGB
 * gravity on a spatial metric \f$\gamma_{ij}\f$.
 */
struct LinearizedSources {
  using argument_tags =
      tmpl::list<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                            Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>,
                 Xcts::Tags::ConformalFactor<DataVector>,
                 ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 ::CurvedScalarWave::Tags::Psi, Tags::Epsilon2, Tags::Epsilon4,
                 gr::Tags::WeylElectricScalar<DataVector>,
                 gr::Tags::WeylMagneticScalar<DataVector>>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> linearized_equation_for_field,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, 3>& deriv_lapse,
      const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::i<DataVector, 3>& conformal_factor_deriv,
      const Scalar<DataVector>& field, const double& eps2, const double& eps4,
      const Scalar<DataVector>& weyl_electric,
      const Scalar<DataVector>& weyl_magnetic,
      const Scalar<DataVector>& field_correction,
      const tnsr::I<DataVector, 3>& field_flux_correction);
};

}  // namespace sgb
