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
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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

namespace RadiationTransport::M1Grey::BoundaryConditions {

/// \cond
template <typename NeutrinoSpeciesList>
class DirichletAnalytic;
/// \endcond

/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
template <typename... NeutrinoSpecies>
class DirichletAnalytic<tmpl::list<NeutrinoSpecies...>> final
    : public BoundaryCondition<tmpl::list<NeutrinoSpecies...>> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) noexcept = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) noexcept = default;
  DirichletAnalytic(const DirichletAnalytic&) = default;
  DirichletAnalytic& operator=(const DirichletAnalytic&) = default;
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg) noexcept
      : BoundaryCondition<tmpl::list<NeutrinoSpecies...>>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<DirichletAnalytic>(*this);
  }

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override {
    BoundaryCondition<tmpl::list<NeutrinoSpecies...>>::pup(p);
  }

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial, NeutrinoSpecies>::type*>... tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial, NeutrinoSpecies>::type*>... tilde_s,

      const gsl::not_null<typename ::Tags::Flux<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type*>... flux_tilde_e,
      const gsl::not_null<typename ::Tags::Flux<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type*>... flux_tilde_s,

      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const noexcept {
    auto boundary_values = [&analytic_solution_or_data, &coords,
                            &time]() noexcept {
      if constexpr (std::is_base_of_v<MarkAsAnalyticSolution,
                                      AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                       Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       gr::Tags::Lapse<>, gr::Tags::Shift<3>,
                       gr::Tags::SpatialMetric<3>,
                       gr::Tags::InverseSpatialMetric<3>>{});

      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords,
            tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                       Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       gr::Tags::Lapse<>, gr::Tags::Shift<3>,
                       gr::Tags::SpatialMetric<3>,
                       gr::Tags::InverseSpatialMetric<3>>{});
      }
    }();

    *inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<3>>(boundary_values);

    // Allocate the temporary tensors outside the loop over species
    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempII<0, 3>,
                         ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
                         ::Tags::Tempi<0, 3>, ::Tags::TempI<0, 3>>>
        buffer((*inv_spatial_metric)[0].size());

    const auto assign_boundary_values_for_neutrino_species =
        [&boundary_values, &inv_spatial_metric, &buffer](
            const gsl::not_null<Scalar<DataVector>*> local_tilde_e,
            const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
                local_tilde_s,
            const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
                local_flux_tilde_e,
            const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
                local_flux_tilde_s,
            auto species_v) noexcept {
          using species = decltype(species_v);
          using tilde_e_tag = Tags::TildeE<Frame::Inertial, species>;
          using tilde_s_tag = Tags::TildeS<Frame::Inertial, species>;
          *local_tilde_e = get<tilde_e_tag>(boundary_values);
          *local_tilde_s = get<tilde_s_tag>(boundary_values);

          // Compute pressure tensor tilde_p from the M1Closure
          const auto& fluid_velocity =
              get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values);
          const auto& fluid_lorentz_factor =
              get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values);
          const auto& lapse = get<gr::Tags::Lapse<>>(boundary_values);
          const auto& shift = get<gr::Tags::Shift<3>>(boundary_values);
          const auto& spatial_metric =
              get<gr::Tags::SpatialMetric<3>>(boundary_values);
          auto& closure_factor = get<::Tags::TempScalar<0>>(buffer);
          // The M1Closure reads in values from `closure_factor` as an initial
          // guess. We need to specify some value (else code fails in DEBUG
          // mode when the temp Variables are initialized with NaNs)... for now
          // we use 0 because it's easy, but improvements may be possible.
          get(closure_factor) = 0.;
          auto& pressure_tensor = get<::Tags::TempII<0, 3>>(buffer);
          auto& comoving_energy_density = get<::Tags::TempScalar<1>>(buffer);
          auto& comoving_momentum_density_normal =
              get<::Tags::TempScalar<2>>(buffer);
          auto& comoving_momentum_density_spatial =
              get<::Tags::Tempi<0, 3>>(buffer);
          detail::compute_closure_impl(
              make_not_null(&closure_factor), make_not_null(&pressure_tensor),
              make_not_null(&comoving_energy_density),
              make_not_null(&comoving_momentum_density_normal),
              make_not_null(&comoving_momentum_density_spatial), *local_tilde_e,
              *local_tilde_s, fluid_velocity, fluid_lorentz_factor,
              spatial_metric, *inv_spatial_metric);
          // Store det of metric in (otherwise unused) comoving_energy_density
          get(comoving_energy_density) = get(determinant(spatial_metric));
          for (auto& component : pressure_tensor) {
            component *= get(comoving_energy_density);
          }
          const auto& tilde_p = pressure_tensor;

          auto& tilde_s_M = get<::Tags::TempI<0, 3>>(buffer);
          detail::compute_fluxes_impl(local_flux_tilde_e, local_flux_tilde_s,
                                      &tilde_s_M, *local_tilde_e,
                                      *local_tilde_s, tilde_p, lapse, shift,
                                      spatial_metric, *inv_spatial_metric);
          return '0';
        };

    expand_pack(assign_boundary_values_for_neutrino_species(
        tilde_e, tilde_s, flux_tilde_e, flux_tilde_s, NeutrinoSpecies{})...);
    return {};
  }
};

/// \cond
template <typename... NeutrinoSpecies>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::my_PUP_ID =
    0;
/// \endcond

}  // namespace RadiationTransport::M1Grey::BoundaryConditions
