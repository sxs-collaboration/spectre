// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

/*!
 * \brief Applies no boundary condition at all. Used to impose nothing but
 * regularity at the horizon in horizon-penetrating coordinates.
 */
class None : public elliptic::BoundaryConditions::BoundaryCondition<2> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<2>;
  using GradTensorType =
      TensorMetafunctions::prepend_spatial_index<tnsr::aa<ComplexDataVector, 3>,
                                                 2, UpLo::Lo, Frame::Inertial>;

 public:
  static constexpr Options::String help =
      "Applies no boundary condition at all. Used to impose nothing but "
      "regularity at the horizon in horizon-penetrating coordinates.";
  using options = tmpl::list<>;

  None() = default;
  None(const None&) = default;
  None& operator=(const None&) = default;
  None(None&&) = default;
  None& operator=(None&&) = default;
  ~None() override = default;

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(None);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override;

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Neumann};
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
          n_dot_field_gradient_correction,
      const GradTensorType& deriv_field_correction) const;

 private:
  friend bool operator==(const None& lhs, const None& rhs);
};

bool operator!=(const None& lhs, const None& rhs);

}  // namespace GrSelfForce::BoundaryConditions
