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
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace GeneralizedHarmonic::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
template <size_t Dim>
class DirichletAnalytic final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions setting the value of the "
      "spacetime metric and its derivatives Phi and Pi to the analytic "
      "solution or analytic data."};

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
  using dg_interior_temporary_tags = tmpl::list<
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*> phi,
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2, const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const noexcept {
    *gamma1 = interior_gamma1;
    *gamma2 = interior_gamma2;
    auto boundary_values = [&analytic_solution_or_data, &coords,
                            &time]() noexcept {
      if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                      AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<
                GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>,
                GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>,
                gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>{});

      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords,
            tmpl::list<
                GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>,
                GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>,
                gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>{});
      }
    }();

    *spacetime_metric =
        get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
            boundary_values);
    *pi = get<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(
        boundary_values);
    *phi = get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(
        boundary_values);

    // Now compute lapse and shift...
    lapse_shift_and_inv_spatial_metric(lapse, shift, inv_spatial_metric,
                                       *spacetime_metric);
    return {};
  }

 private:
  void lapse_shift_and_inv_spatial_metric(
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          inv_spatial_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric)
      const noexcept;
};
}  // namespace GeneralizedHarmonic::BoundaryConditions
