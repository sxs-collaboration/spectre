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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
class DirichletAnalytic final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data."};

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
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> tilde_d,
      const gsl::not_null<Scalar<DataVector>*> tilde_tau,
      const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<DataVector>*> tilde_phi,

      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_d_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_tau_flux,
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          tilde_s_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          tilde_b_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_phi_flux,

      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const {
    auto boundary_values = [&analytic_solution_or_data, &coords, &time]() {
      if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                      AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<
                hydro::Tags::RestMassDensity<DataVector>,
                hydro::Tags::SpecificInternalEnergy<DataVector>,
                hydro::Tags::SpecificEnthalpy<DataVector>,
                hydro::Tags::Pressure<DataVector>,
                hydro::Tags::SpatialVelocity<DataVector, 3>,
                hydro::Tags::LorentzFactor<DataVector>,
                hydro::Tags::MagneticField<DataVector, 3>,
                hydro::Tags::DivergenceCleaningField<DataVector>,
                gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
                gr::Tags::SqrtDetSpatialMetric<DataVector>,
                gr::Tags::Lapse<DataVector>,
                gr::Tags::Shift<3, Frame::Inertial, DataVector>>{});

      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords,
            tmpl::list<
                hydro::Tags::RestMassDensity<DataVector>,
                hydro::Tags::SpecificInternalEnergy<DataVector>,
                hydro::Tags::SpecificEnthalpy<DataVector>,
                hydro::Tags::Pressure<DataVector>,
                hydro::Tags::SpatialVelocity<DataVector, 3>,
                hydro::Tags::LorentzFactor<DataVector>,
                hydro::Tags::MagneticField<DataVector, 3>,
                hydro::Tags::DivergenceCleaningField<DataVector>,
                gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
                gr::Tags::SqrtDetSpatialMetric<DataVector>,
                gr::Tags::Lapse<DataVector>,
                gr::Tags::Shift<3, Frame::Inertial, DataVector>>{});
      }
    }();

    *lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
    *shift =
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(boundary_values);
    *inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
            boundary_values);

    ConservativeFromPrimitive::apply(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
        get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_values),
        get<hydro::Tags::SpecificEnthalpy<DataVector>>(boundary_values),
        get<hydro::Tags::Pressure<DataVector>>(boundary_values),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values),
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values),
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_values),
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            boundary_values),
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_values));

    ComputeFluxes::apply(
        tilde_d_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux,
        tilde_phi_flux, *tilde_d, *tilde_tau, *tilde_s, *tilde_b, *tilde_phi,
        *lapse, *shift,
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_values),
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            boundary_values),
        get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
            boundary_values),
        get<hydro::Tags::Pressure<DataVector>>(boundary_values),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values),
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values),
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values));

    return {};
  }
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
