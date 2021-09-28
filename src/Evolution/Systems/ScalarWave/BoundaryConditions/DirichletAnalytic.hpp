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
#include "Evolution/Systems/ScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
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

namespace ScalarWave::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
template <size_t Dim>
class DirichletAnalytic final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions setting the value of Psi, Phi, "
      "and Pi to the analytic solution or analytic data."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&) = default;
  DirichletAnalytic& operator=(const DirichletAnalytic&) = default;
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Tags::ConstraintGamma2>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> pi,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      const gsl::not_null<Scalar<DataVector>*> psi,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& interior_gamma2, const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const {
    *gamma2 = interior_gamma2;
    auto boundary_values = [&analytic_solution_or_data, &coords, &time]() {
      if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                      AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<ScalarWave::Pi, ScalarWave::Phi<Dim>,
                       ScalarWave::Psi>{});

      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords, tmpl::list<ScalarWave::Pi, ScalarWave::Phi<Dim>,
                               ScalarWave::Psi>{});
      }
    }();
    *pi = get<ScalarWave::Pi>(boundary_values);
    *phi = get<ScalarWave::Phi<Dim>>(boundary_values);
    *psi = get<ScalarWave::Psi>(boundary_values);
    return {};
  }
};
}  // namespace ScalarWave::BoundaryConditions
