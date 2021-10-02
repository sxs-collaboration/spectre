// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Poisson::BoundaryConditions {
namespace detail {

struct RobinImpl {
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

  RobinImpl() = default;
  RobinImpl(const RobinImpl&) = default;
  RobinImpl& operator=(const RobinImpl&) = default;
  RobinImpl(RobinImpl&&) = default;
  RobinImpl& operator=(RobinImpl&&) = default;
  ~RobinImpl() = default;

  RobinImpl(double dirichlet_weight, double neumann_weight, double constant,
            const Options::Context& context = {});

  double dirichlet_weight() const;
  double neumann_weight() const;
  double constant() const;

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<DataVector>*> field,
             gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> field_correction,
      gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient_correction) const;

  void pup(PUP::er& p);

 private:
  double dirichlet_weight_ = std::numeric_limits<double>::signaling_NaN();
  double neumann_weight_ = std::numeric_limits<double>::signaling_NaN();
  double constant_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator==(const RobinImpl& lhs, const RobinImpl& rhs);
bool operator!=(const RobinImpl& lhs, const RobinImpl& rhs);

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <size_t Dim, typename Registrars>
struct Robin;

namespace Registrars {
template <size_t Dim>
struct Robin {
  template <typename Registrars>
  using f = BoundaryConditions::Robin<Dim, Registrars>;
};
}  // namespace Registrars
/// \endcond

/// Impose Robin boundary conditions \f$a u + b n_i \nabla^i u = c\f$. The
/// boundary condition is imposed as Neumann-type (i.e. on \f$n_i \nabla^i u\f$)
/// if \f$|b| > 0\f$ and as Dirichlet-type (i.e. on \f$u\f$) if \f$b = 0\f$.
template <size_t Dim, typename Registrars = tmpl::list<Registrars::Robin<Dim>>>
class Robin
    : public elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>,
      public detail::RobinImpl {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>;

 public:
  Robin() = default;
  Robin(const Robin&) = default;
  Robin& operator=(const Robin&) = default;
  Robin(Robin&&) = default;
  Robin& operator=(Robin&&) = default;
  ~Robin() = default;
  using RobinImpl::RobinImpl;

  /// \cond
  explicit Robin(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Robin);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Robin>(*this);
  }

  void pup(PUP::er& p) override {
    Base::pup(p);
    RobinImpl::pup(p);
  }
};

/// \cond
template <size_t Dim, typename Registrars>
PUP::able::PUP_ID Robin<Dim,
                        Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Poisson::BoundaryConditions
