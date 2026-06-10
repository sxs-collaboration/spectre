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
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce {

/*!
 * \brief The first-order flux
 * $F^i=\{\partial_{r_\star}, \alpha \partial_{\cos\theta}\}\Psi_m$.
 */
void fluxes(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
            const tnsr::I<ComplexDataVector, 2>& alpha,
            const tnsr::i<ComplexDataVector, 2>& field_gradient);

/*!
 * \brief The first-order flux on an element face
 * $F^i=\{n_{r_\star}, \alpha n_{\cos\theta}\}\Psi_m$.
 */
void fluxes_on_face(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const tnsr::I<ComplexDataVector, 2>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field);

/*!
 * \brief The source term $\beta \Psi_m + \gamma_i \partial_i \Psi_m$.
 */
void add_sources(gsl::not_null<Scalar<ComplexDataVector>*> source,
                 const Scalar<ComplexDataVector>& beta,
                 const tnsr::i<ComplexDataVector, 2>& gamma,
                 const Scalar<ComplexDataVector>& field,
                 const tnsr::i<ComplexDataVector, 2>& field_gradient);

/// Fluxes $F^i$ for the scalar self-force system.
/// \see ScalarSelfForce::FirstOrderSystem
struct Fluxes {
  using argument_tags = tmpl::list<Tags::Alpha>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const tnsr::I<ComplexDataVector, 2>& alpha,
                    const Scalar<ComplexDataVector>& /*field*/,
                    const tnsr::i<ComplexDataVector, 2>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const tnsr::I<ComplexDataVector, 2>& alpha,
                    const tnsr::i<DataVector, 2>& /*face_normal*/,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field);
};

/// Source terms for the scalar self-force system.
/// \see ScalarSelfForce::FirstOrderSystem
struct Sources {
  using argument_tags = tmpl::list<Tags::Beta, Tags::Gamma>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(gsl::not_null<Scalar<ComplexDataVector>*> scalar_equation,
                    const Scalar<ComplexDataVector>& beta,
                    const tnsr::i<ComplexDataVector, 2>& gamma,
                    const Scalar<ComplexDataVector>& field,
                    const tnsr::i<ComplexDataVector, 2>& field_gradient,
                    const tnsr::I<ComplexDataVector, 2>& flux);
};

/*!
 * \brief Modifies the received boundary data to account for jump conditions.
 *
 * \par Worldtube jump conditions
 * The `apply` method adds or subtracts the singular field to/from
 * the received data on element boundaries.
 *
 * In the regularized region we solve for the regularized field
 * \begin{equation}
 *   \Psi_m^R = \Psi_m - \Psi_m^P
 *   \text{,}
 * \end{equation}
 * so we subtract the singular field on the regularized side and add it on the
 * other side of the boundary. We do the same for the received normal dot flux
 * $n_i F^i$, but with an extra minus sign because this quantity is defined with
 * the face normal from the perspective of the sending element (see
 * `elliptic::protocols::FirstOrderSystem`).
 *
 * \par VTU-slicing jump conditions
 * The `apply_linearized` method imposes the jump conditions at the
 * interfaces between domains with different slicings (v-t and t-u transitions).
 * At these interfaces the boost function $H(r_*)$ is discontinuous, which
 * produces a jump in the radial derivative of $\Psi_m$
 * [Eqs. (2.36)-(2.37) in \cite Vu:2026ypc ].
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
 public:
  using argument_tags_linearized = tmpl::list<
      domain::Tags::Element<Dim>, Tags::NullSlicingBlocks<Dim>,
      elliptic::Tags::Background<elliptic::analytic_data::Background>>;
  using const_global_cache_tags = tmpl::list<
      Tags::NullSlicingBlocks<Dim>,
      elliptic::Tags::Background<elliptic::analytic_data::Background>>;
  static void apply(
      gsl::not_null<Scalar<ComplexDataVector>*> field,
      gsl::not_null<Scalar<ComplexDataVector>*> n_dot_flux,
      const DirectionalId<Dim>& mortar_id, bool field_is_regularized,
      const DirectionalIdMap<Dim, bool>& neighbors_field_is_regularized,
      const DirectionalIdMap<Dim, typename singular_vars_on_mortars_tag::type>&
          singular_vars_on_mortars);
  static void apply_linearized(
      gsl::not_null<Scalar<ComplexDataVector>*> field_remote,
      gsl::not_null<Scalar<ComplexDataVector>*> n_dot_flux_remote,
      const Scalar<ComplexDataVector>& field_local,
      const Scalar<ComplexDataVector>& n_dot_flux_local,
      const DirectionalId<Dim>& mortar_id, const Element<Dim>& element,
      const std::vector<size_t>& null_slicing_blocks,
      const elliptic::analytic_data::Background& background);
};

}  // namespace ScalarSelfForce
