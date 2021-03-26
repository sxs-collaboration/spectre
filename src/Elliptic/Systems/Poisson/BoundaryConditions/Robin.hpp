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
  RobinImpl(const RobinImpl&) noexcept = default;
  RobinImpl& operator=(const RobinImpl&) noexcept = default;
  RobinImpl(RobinImpl&&) noexcept = default;
  RobinImpl& operator=(RobinImpl&&) noexcept = default;
  ~RobinImpl() noexcept = default;

  RobinImpl(double dirichlet_weight, double neumann_weight, double constant,
            const Options::Context& context = {});

  double dirichlet_weight() const noexcept;
  double neumann_weight() const noexcept;
  double constant() const noexcept;

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<Scalar<DataVector>*> field,
      gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient) const noexcept;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(gsl::not_null<Scalar<DataVector>*> field_correction,
                        gsl::not_null<Scalar<DataVector>*>
                            n_dot_field_gradient_correction) const noexcept;

  void pup(PUP::er& p) noexcept;

 private:
  double dirichlet_weight_ = std::numeric_limits<double>::signaling_NaN();
  double neumann_weight_ = std::numeric_limits<double>::signaling_NaN();
  double constant_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator==(const RobinImpl& lhs, const RobinImpl& rhs) noexcept;
bool operator!=(const RobinImpl& lhs, const RobinImpl& rhs) noexcept;

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
  Robin(const Robin&) noexcept = default;
  Robin& operator=(const Robin&) noexcept = default;
  Robin(Robin&&) noexcept = default;
  Robin& operator=(Robin&&) noexcept = default;
  ~Robin() noexcept = default;
  using RobinImpl::RobinImpl;

  /// \cond
  explicit Robin(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Robin);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<Robin>(*this);
  }

  void pup(PUP::er& p) noexcept override {
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
