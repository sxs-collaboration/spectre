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
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
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

namespace NewtonianEuler::BoundaryConditions {
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
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          momentum_density,
      const gsl::not_null<Scalar<DataVector>*> energy_density,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_mass_density,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_momentum_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_energy_density,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const {
    auto boundary_values = [&analytic_solution_or_data, &coords, &time]() {
      if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                      AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<Tags::MassDensity<DataVector>,
                       Tags::Velocity<DataVector, Dim>,
                       Tags::Pressure<DataVector>,
                       Tags::SpecificInternalEnergy<DataVector>>{});

      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords, tmpl::list<Tags::MassDensity<DataVector>,
                               Tags::Velocity<DataVector, Dim>,
                               Tags::Pressure<DataVector>,
                               Tags::SpecificInternalEnergy<DataVector>>{});
      }
    }();

    *mass_density = get<Tags::MassDensity<DataVector>>(boundary_values);
    *velocity = get<Tags::Velocity<DataVector, Dim>>(boundary_values);
    *specific_internal_energy =
        get<Tags::SpecificInternalEnergy<DataVector>>(boundary_values);

    ConservativeFromPrimitive<Dim>::apply(mass_density, momentum_density,
                                          energy_density, *mass_density,
                                          *velocity, *specific_internal_energy);
    ComputeFluxes<Dim>::apply(flux_mass_density, flux_momentum_density,
                              flux_energy_density, *momentum_density,
                              *energy_density, *velocity,
                              get<Tags::Pressure<DataVector>>(boundary_values));

    return {};
  }
};
}  // namespace NewtonianEuler::BoundaryConditions
