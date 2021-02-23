// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace Burgers::BoundaryConditions {
class DirichletAnalytic final : public BoundaryCondition {
 private:
  using flux_tag =
      ::Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions setting the value of U to "
      "the analytic solution or analytic data."};
  static std::string name() noexcept { return "DirichletAnalytic"; }

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) noexcept = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) noexcept = default;
  DirichletAnalytic(const DirichletAnalytic&) = default;
  DirichletAnalytic& operator=(const DirichletAnalytic&) = default;
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<1, Frame::Inertial>>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> u,
      const gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux_u,
      const std::optional<
          tnsr::I<DataVector, 1, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 1, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 1, Frame::Inertial>& coords,
      [[maybe_unused]] const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const noexcept {
    if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                    AnalyticSolutionOrData>) {
      *u = get<Burgers::Tags::U>(analytic_solution_or_data.variables(
          coords, time, tmpl::list<Burgers::Tags::U>{}));
    } else {
      *u = get<Burgers::Tags::U>(analytic_solution_or_data.variables(
          coords, tmpl::list<Burgers::Tags::U>{}));
    }
    flux_impl(flux_u, *u);
    return {};
  }

 private:
  static void flux_impl(
      gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux,
      const Scalar<DataVector>& u_analytic) noexcept;
};
}  // namespace Burgers::BoundaryConditions
