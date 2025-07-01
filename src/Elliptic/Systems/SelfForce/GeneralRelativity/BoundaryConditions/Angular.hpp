// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

/*!
 * \brief Angular boundary conditions for the m-mode metric perturbations.
 *
 * The angular boundary conditions are determined by enforcing regularity of
 * the metric perturbation at the poles $\theta=0,\pi$:
 * \begin{align}
 *   &\Psi_{m=1}^{0,1,4,7,8,9} = 0 \\
 *   &\partial_\theta \Psi_{m=1}^{2,3,5,6} = 0 \\
 *   &\Psi_{m=2}^{0,1,2,3,4,5,6} = 0 \\
 *   &\partial_\theta \Psi_{m=2}^{7,8,9} = 0 \\
 *   &\Psi_{m>2} = 0 \\
 * \end{align}
 * The ordering of components in $\Psi$ here is:
 * $\{tt,tr,t\theta,t\phi,rr,r\theta,r\phi,\theta\theta,\theta\phi,\phi\phi\}$.
 */
class Angular : public elliptic::BoundaryConditions::BoundaryCondition<2> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<2>;
  using GradTensorType =
      TensorMetafunctions::prepend_spatial_index<tnsr::aa<ComplexDataVector, 3>,
                                                 2, UpLo::Lo, Frame::Inertial>;

 public:
  struct MModeNumber {
    static constexpr Options::String help =
        "Mode number 'm' of the metric perturbation";
    using type = int;
  };

  static constexpr Options::String help = "Angular boundary condition";
  using options = tmpl::list<MModeNumber>;

  Angular() = default;
  Angular(const Angular&) = default;
  Angular& operator=(const Angular&) = default;
  Angular(Angular&&) = default;
  Angular& operator=(Angular&&) = default;
  ~Angular() override = default;

  explicit Angular(int m_mode_number);

  int m_mode_number() const { return m_mode_number_; }

  /// \cond
  explicit Angular(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Angular);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Angular>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    // For m > 0 some tensor components are Neumann-type and others Dirichlet.
    // The boundary condition type is currently only used for the
    // `MinusLaplacian` subdomain preconditioner, so its effectiveness will be
    // lower unless we extend this function to return a type for each tensor
    // component.
    return {m_mode_number_ == 0 ? elliptic::BoundaryConditionType::Neumann
                                : elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_field_gradient,
      const GradTensorType& deriv_field) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field_correction,
      gsl::not_null<tnsr::aa<ComplexDataVector, 3>*>
          n_dot_field_correction_gradient,
      const GradTensorType& deriv_field_correction) const;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const Angular& lhs, const Angular& rhs);

  int m_mode_number_{};
};

bool operator!=(const Angular& lhs, const Angular& rhs);

}  // namespace GrSelfForce::BoundaryConditions
