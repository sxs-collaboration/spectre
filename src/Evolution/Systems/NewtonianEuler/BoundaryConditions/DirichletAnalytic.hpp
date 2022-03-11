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
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
      if constexpr (is_analytic_solution_v<AnalyticSolutionOrData>) {
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

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags = tmpl::list<
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags =
      tmpl::list<::Tags::Time, domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 fd::Tags::Reconstructor<Dim>, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  void fd_ghost(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const Direction<Dim>& direction, const Mesh<Dim> subcell_mesh,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
          subcell_logical_coords,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map,
      const fd::Reconstructor<Dim>& reconstructor,
      const AnalyticSolutionOrData& analytic_solution_or_data) const {
    const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

    // slice logical coordinates and shift it to compute inertial coordinates of
    // ghost points
    auto shifted_logical_coords =
        evolution::dg::subcell::slice_tensor_for_subcell(
            subcell_logical_coords, subcell_mesh.extents(), ghost_zone_size,
            direction);
    // Note: assumes isotropic subcell extents
    ASSERT(
        subcell_mesh == Mesh<Dim>(subcell_mesh.extents(0),
                                  subcell_mesh.basis(0),
                                  subcell_mesh.quadrature(0)),
        "The subcell/FD mesh must be isotropic for the FD Dirichlet analytic "
        "boundary condition but got "
            << subcell_mesh);
    const double delta_x{subcell_logical_coords.get(0)[1] -
                         subcell_logical_coords.get(0)[0]};
    const size_t dim_to_slice{direction.dimension()};
    shifted_logical_coords.get(dim_to_slice) =
        shifted_logical_coords.get(dim_to_slice) +
        direction.sign() * delta_x * ghost_zone_size;
    const auto shifted_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(shifted_logical_coords), time, functions_of_time);

    // Compute FD ghost data (prims) with the analytic data or solution
    auto boundary_values = [&analytic_solution_or_data,
                            &shifted_inertial_coords, &time]() {
      if constexpr (is_analytic_solution_v<AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            shifted_inertial_coords, time,
            tmpl::list<Tags::MassDensity<DataVector>,
                       Tags::Velocity<DataVector, Dim>,
                       Tags::Pressure<DataVector>>{});
      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            shifted_inertial_coords, tmpl::list<Tags::MassDensity<DataVector>,
                                                Tags::Velocity<DataVector, Dim>,
                                                Tags::Pressure<DataVector>>{});
      }
    }();

    *mass_density = get<Tags::MassDensity<DataVector>>(boundary_values);
    *velocity = get<Tags::Velocity<DataVector, Dim>>(boundary_values);
    *pressure = get<Tags::Pressure<DataVector>>(boundary_values);
  }
};
}  // namespace NewtonianEuler::BoundaryConditions
