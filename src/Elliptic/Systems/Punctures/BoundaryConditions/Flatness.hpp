// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Punctures::BoundaryConditions {

/// Impose asymptotic flatness boundary conditions $\partial_r(ru)=0$
class Flatness : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help = "Asymptotic flatness d_r(ru)=0";
  using options = tmpl::list<>;

  Flatness() = default;
  Flatness(const Flatness&) = default;
  Flatness& operator=(const Flatness&) = default;
  Flatness(Flatness&&) = default;
  Flatness& operator=(Flatness&&) = default;
  ~Flatness() = default;

  /// \cond
  explicit Flatness(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Flatness);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Flatness>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;

  static void apply(gsl::not_null<Scalar<DataVector>*> field,
                    gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient,
                    const tnsr::I<DataVector, 3>& x);

  using argument_tags_linearized =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> field_correction,
      gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient_correction,
      const tnsr::I<DataVector, 3>& x);
};

bool operator==(const Flatness& lhs, const Flatness& rhs);
bool operator!=(const Flatness& lhs, const Flatness& rhs);

}  // namespace Punctures::BoundaryConditions
