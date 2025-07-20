// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/RadiationTransport/M1Grey/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace RadiationTransport::M1Grey::BoundaryConditions {

template <typename... NeutrinoSpecies>
DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::DirichletAnalytic(
    const DirichletAnalytic& rhs)
    : BoundaryCondition<tmpl::list<NeutrinoSpecies...>>{dynamic_cast<
          const BoundaryCondition<tmpl::list<NeutrinoSpecies...>>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

template <typename... NeutrinoSpecies>
DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>&
DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::operator=(
    const DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

template <typename... NeutrinoSpecies>
DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

template <typename... NeutrinoSpecies>
void DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::pup(PUP::er& p) {
  BoundaryCondition<tmpl::list<NeutrinoSpecies...>>::pup(p);
  p | analytic_prescription_;
}

template <typename... NeutrinoSpecies>
std::optional<std::string>
DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::dg_ghost(
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
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    const double time) const {
  using initial_data_list =
      tmpl::append<RadiationTransport::M1Grey::AnalyticData::all_data,
                   RadiationTransport::M1Grey::Solutions::all_solutions>;
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataVector>,
                          hydro::Tags::SpatialVelocity<DataVector, 3>,
                          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                          gr::Tags::Lapse<DataVector>,
                          gr::Tags::Shift<DataVector, 3>,
                          gr::Tags::SpatialMetric<DataVector, 3>,
                          gr::Tags::InverseSpatialMetric<DataVector, 3>>,
      initial_data_list>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const initial_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(
              coords, time,
              tmpl::list<hydro::Tags::LorentzFactor<DataVector>,
                         hydro::Tags::SpatialVelocity<DataVector, 3>,
                         Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                         Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>>{});

        } else {
          (void)time;
          return initial_data->variables(
              coords,
              tmpl::list<hydro::Tags::LorentzFactor<DataVector>,
                         hydro::Tags::SpatialVelocity<DataVector, 3>,
                         Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                         Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>>{});
        }
      });

  *inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(boundary_values);

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
          auto species_v) {
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
        const auto& lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
        const auto& shift =
            get<gr::Tags::Shift<DataVector, 3>>(boundary_values);
        const auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataVector, 3>>(boundary_values);
        auto& closure_factor = get<::Tags::TempScalar<0>>(buffer);
        // The M1Closure reads in values from `closure_factor` as an
        //   initial
        // guess. We need to specify some value (else code fails in DEBUG
        // mode when the temp Variables are initialized with NaNs)... for
        //   now
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
                                    &tilde_s_M, *local_tilde_e, *local_tilde_s,
                                    tilde_p, lapse, shift, spatial_metric,
                                    *inv_spatial_metric);
        return '0';
      };

  expand_pack(assign_boundary_values_for_neutrino_species(
      tilde_e, tilde_s, flux_tilde_e, flux_tilde_s, NeutrinoSpecies{})...);

  return {};
}

#ifndef __CUDA_ARCH__
template <typename... NeutrinoSpecies>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic<tmpl::list<NeutrinoSpecies...>>::my_PUP_ID =
    0;
#endif  // __CUDA_ARCH__

template class DirichletAnalytic<tmpl::list<neutrinos::ElectronNeutrinos<1>>>;

template class DirichletAnalytic<tmpl::list<
    neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>>;

}  // namespace RadiationTransport::M1Grey::BoundaryConditions
