// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce {

/// @{
/// We're working with 4D tensors to represent the 10 independent components
/// we're solving for, but we only take 2D spatial derivatives, so we define
/// these mixed-dimension tensors for gradients and fluxes.
using GradTensorType =
    TensorMetafunctions::prepend_spatial_index<tnsr::aa<ComplexDataVector, 3>,
                                               2, UpLo::Lo, Frame::Inertial>;
using FluxTensorType =
    TensorMetafunctions::prepend_spatial_index<tnsr::aa<ComplexDataVector, 3>,
                                               2, UpLo::Up, Frame::Inertial>;
/// @}

/*!
 * \brief The first-order flux $F^i=\{\partial_{r_\star}, \alpha
 * \partial_\theta\}\Psi_m$.
 */
void fluxes(gsl::not_null<FluxTensorType*> flux,
            const Scalar<ComplexDataVector>& alpha,
            const GradTensorType& field_gradient);

/*!
 * \brief The first-order flux on an element face
 * $F^i=\{n_{r_\star}, \alpha n_\theta\}\Psi_m$.
 */
void fluxes_on_face(gsl::not_null<FluxTensorType*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const tnsr::aa<ComplexDataVector, 3>& field);

/*!
 * \brief The source term $\beta_{ab}^{cd} (\Psi_m)_{cd} + \gamma_{iab}^{cd}
 * F^i_{cd}$.
 */
void add_sources(gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> source,
                 const tnsr::aaBB<ComplexDataVector, 3>& beta,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
                 const tnsr::aa<ComplexDataVector, 3>& field,
                 const FluxTensorType& flux);

/// Fluxes $F^i$ for the gravitational self-force system.
/// \see GrSelfForce::FirstOrderSystem
struct Fluxes {
  using argument_tags = tmpl::list<Tags::Alpha>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<FluxTensorType*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::aa<ComplexDataVector, 3>& /*field*/,
                    const GradTensorType& field_gradient);
  static void apply(gsl::not_null<FluxTensorType*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::i<DataVector, 2>& /*face_normal*/,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const tnsr::aa<ComplexDataVector, 3>& field);
};

/// Source terms for the gravitational self-force system.
/// \see GrSelfForce::FirstOrderSystem
struct Sources {
  using argument_tags =
      tmpl::list<Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> scalar_equation,
      const tnsr::aaBB<ComplexDataVector, 3>& beta,
      const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
      const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
      const tnsr::aa<ComplexDataVector, 3>& field, const FluxTensorType& flux);
};

/*!
 * \brief Adds or subtracts the singular field to/from the received data on
 * element boundaries.
 *
 * In the regularized region we solve for the regularized field
 * \begin{equation}
 *   \Psi_m^R = \Psi_m - \Psi_m^P
 *   \text{,}
 * \end{equation}
 * so we subtract the singular field on the regularized side (where
 * `field_is_regularized` is true) and add it on the other side of the boundary
 * (where `field_is_regularized` is false). We do the same for the received
 * normal dot flux $n_i F^i$, but with an extra minus sign because this quantity
 * is defined with the face normal from the perspective of the sending element
 * (see `elliptic::protocols::FirstOrderSystem`).
 */
struct ModifyBoundaryData {
 private:
  static constexpr size_t Dim = 2;
  using singular_vars_on_mortars_tag =
      ::Tags::Variables<tmpl::list<Tags::SingularField,
                                   ::Tags::NormalDotFlux<Tags::SingularField>>>;

 public:
  using argument_tags =
      tmpl::list<Tags::FieldIsRegularized,
                 ::Tags::Mortars<Tags::FieldIsRegularized, Dim>,
                 ::Tags::Mortars<singular_vars_on_mortars_tag, Dim>>;
  static void apply(
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_flux,
      const DirectionalId<Dim>& mortar_id, bool field_is_regularized,
      const DirectionalIdMap<Dim, bool>& neighbors_field_is_regularized,
      const DirectionalIdMap<Dim, typename singular_vars_on_mortars_tag::type>&
          singular_vars_on_mortars);
};

}  // namespace GrSelfForce
