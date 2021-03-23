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
#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Fluxes.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
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

namespace RelativisticEuler::Valencia::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
template <size_t Dim>
class DirichletAnalytic final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data."};
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
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> tilde_d,
      const gsl::not_null<Scalar<DataVector>*> tilde_tau,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> tilde_s,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_tilde_d,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_tilde_tau,
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
          flux_tilde_s,

      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          spatial_metric,
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,

      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const noexcept {
    auto boundary_values = [&analytic_solution_or_data, &coords,
                            &time]() noexcept {
      if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                      AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::SpecificInternalEnergy<DataVector>,
                       hydro::Tags::SpecificEnthalpy<DataVector>,
                       hydro::Tags::Pressure<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, Dim>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::Lapse<>,
                       gr::Tags::Shift<Dim>, gr::Tags::SpatialMetric<Dim>>{});
      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords,
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::SpecificInternalEnergy<DataVector>,
                       hydro::Tags::SpecificEnthalpy<DataVector>,
                       hydro::Tags::Pressure<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, Dim>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::Lapse<>,
                       gr::Tags::Shift<Dim>, gr::Tags::SpatialMetric<Dim>>{});
      }
    }();

    *lapse = get<gr::Tags::Lapse<>>(boundary_values);
    *shift = get<gr::Tags::Shift<Dim>>(boundary_values);
    *spatial_metric = get<gr::Tags::SpatialMetric<Dim>>(boundary_values);
    *rest_mass_density =
        get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values);
    *specific_internal_energy =
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_values);
    *specific_enthalpy =
        get<hydro::Tags::SpecificEnthalpy<DataVector>>(boundary_values);
    *spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(boundary_values);

    const auto& pressure =
        get<hydro::Tags::Pressure<DataVector>>(boundary_values);
    const auto& lorentz_factor =
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values);
    const auto& sqrt_det_spatial_metric =
        get<gr::Tags::SqrtDetSpatialMetric<>>(boundary_values);

    ConservativeFromPrimitive<Dim>::apply(
        tilde_d, tilde_tau, tilde_s, *rest_mass_density,
        *specific_internal_energy, *specific_enthalpy, pressure,
        *spatial_velocity, lorentz_factor, sqrt_det_spatial_metric,
        *spatial_metric);
    ComputeFluxes<Dim>::apply(flux_tilde_d, flux_tilde_tau, flux_tilde_s,
                              *tilde_d, *tilde_tau, *tilde_s, *lapse, *shift,
                              sqrt_det_spatial_metric, pressure,
                              *spatial_velocity);

    return {};
  }
};
}  // namespace RelativisticEuler::Valencia::BoundaryConditions
