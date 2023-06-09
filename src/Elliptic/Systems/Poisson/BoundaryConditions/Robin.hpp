// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Poisson::BoundaryConditions {

/// Impose Robin boundary conditions \f$a u + b n_i \nabla^i u = c\f$. The
/// boundary condition is imposed as Neumann-type (i.e. on \f$n_i \nabla^i u\f$)
/// if \f$|b| > 0\f$ and as Dirichlet-type (i.e. on \f$u\f$) if \f$b = 0\f$.
template <size_t Dim>
class Robin : public elliptic::BoundaryConditions::BoundaryCondition<Dim> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<Dim>;

 public:
  static constexpr Options::String help =
      "Robin boundary conditions a * u + b * n_i grad(u)^i = c. The boundary "
      "condition is imposed as Neumann-type (i.e. on n_i grad(u)^i) if abs(b) "
      "> 0 and as Dirichlet-type (i.e. on u) if b = 0.";

  struct DirichletWeight {
    using type = double;
    static constexpr Options::String help = "The parameter 'a'";
  };

  struct NeumannWeight {
    using type = double;
    static constexpr Options::String help = "The parameter 'b'";
  };

  struct Constant {
    using type = double;
    static constexpr Options::String help = "The parameter 'c'";
  };

  using options = tmpl::list<DirichletWeight, NeumannWeight, Constant>;

  Robin() = default;
  Robin(const Robin&) = default;
  Robin& operator=(const Robin&) = default;
  Robin(Robin&&) = default;
  Robin& operator=(Robin&&) = default;
  ~Robin() = default;

  /// \cond
  explicit Robin(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Robin);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Robin>(*this);
  }

  Robin(double dirichlet_weight, double neumann_weight, double constant,
        const Options::Context& context = {});

  double dirichlet_weight() const { return dirichlet_weight_; }
  double neumann_weight() const { return neumann_weight_; }
  double constant() const { return constant_; }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {1, neumann_weight_ == 0.
                   ? elliptic::BoundaryConditionType::Dirichlet
                   : elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<DataVector>*> field,
             gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> field_correction,
      gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient_correction) const;

  void pup(PUP::er& p) override;

 private:
  double dirichlet_weight_ = std::numeric_limits<double>::signaling_NaN();
  double neumann_weight_ = std::numeric_limits<double>::signaling_NaN();
  double constant_ = std::numeric_limits<double>::signaling_NaN();
};

template <size_t Dim>
bool operator==(const Robin<Dim>& lhs, const Robin<Dim>& rhs);

template <size_t Dim>
bool operator!=(const Robin<Dim>& lhs, const Robin<Dim>& rhs);

}  // namespace Poisson::BoundaryConditions
