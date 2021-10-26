// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
template <typename>
class Variables;
/// \endcond

namespace Burgers::BoundaryConditions {
/*
 * \brief Dirichlet boundary condition setting the value of U to a
 * time-independent constant.
 */
class Dirichlet final : public BoundaryCondition {
 private:
  using flux_tag =
      ::Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>;

 public:
  struct U {
    using type = double;
    static constexpr Options::String help{"The value for U on the boundary"};
  };

  using options = tmpl::list<U>;
  static constexpr Options::String help{
      "Dirichlet boundary condition setting the value of U to "
      "a time-independent constant."};

  Dirichlet(double u_value);

  Dirichlet() = default;
  Dirichlet(Dirichlet&&) = default;
  Dirichlet& operator=(Dirichlet&&) = default;
  Dirichlet(const Dirichlet&) = default;
  Dirichlet& operator=(const Dirichlet&) = default;
  ~Dirichlet() override = default;

  explicit Dirichlet(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(BoundaryCondition, Dirichlet);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> u,
      gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux_u,
      const std::optional<
          tnsr::I<DataVector, 1, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 1, Frame::Inertial>& /*normal_covector*/) const;

 private:
  void dg_ghost_impl(
      gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux,
      gsl::not_null<Scalar<DataVector>*> u) const;

  double u_value_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Burgers::BoundaryConditions
