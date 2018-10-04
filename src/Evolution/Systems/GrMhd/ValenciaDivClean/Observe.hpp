// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd {
namespace ValenciaDivClean {

namespace Actions {
/*!
 * \brief Temporary action for observing volume and reduction data
 *
 * A few notes:
 * - Observation frequency is currently hard-coded and must manually be updated.
 *   Look for `time_by_timestep_value` to update.
 */
struct Observe {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& time = db::get<::Tags::Time>(box);
    // Note: this currently assumes a constant time step that is a power of ten.
    const size_t time_by_timestep_value = static_cast<size_t>(
        std::round(time.value() / db::get<::Tags::TimeStep>(box).value()));

    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';
    // We hard-code the writing frequency to large time values to avoid breaking
    // the tests.
    if (time_by_timestep_value % 1000 == 0 and time_by_timestep_value > 0) {
      const auto& extents = db::get<::Tags::Mesh<Dim>>(box).extents();
      // Retrieve the tensors and compute the solution error.
      const auto& tilde_d = db::get<Tags::TildeD>(box);
      const auto& tilde_tau = db::get<Tags::TildeTau>(box);
      const auto& tilde_phi = db::get<Tags::TildePhi>(box);
      const auto& tilde_s = db::get<Tags::TildeS<>>(box);
      const auto& tilde_b = db::get<Tags::TildeB<>>(box);
      const auto& rest_mass_density =
          db::get<hydro::Tags::RestMassDensity<DataVector>>(box);
      const auto& specific_internal_energy =
          db::get<hydro::Tags::SpecificInternalEnergy<DataVector>>(box);
      const auto& divergence_cleaning_field =
          db::get<hydro::Tags::DivergenceCleaningField<DataVector>>(box);
      const auto& pressure = db::get<hydro::Tags::Pressure<DataVector>>(box);
      const auto& lorentz_factor =
          db::get<hydro::Tags::LorentzFactor<DataVector>>(box);
      const auto& specific_enthalpy =
          db::get<hydro::Tags::SpecificEnthalpy<DataVector>>(box);
      const auto& spatial_velocity =
          db::get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(box);
      const auto& magnetic_field =
          db::get<hydro::Tags::MagneticField<DataVector, Dim>>(box);

      const auto& inertial_coordinates =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Compute the error in the solution, and generate tensor component list.
      using PrimitiveVars =
          typename Metavariables::analytic_variables_tags;
      using solution_tag = OptionTags::AnalyticSolutionBase;
      const auto exact_solution = Parallel::get<solution_tag>(cache).variables(
          inertial_coordinates, time.value(), PrimitiveVars{});

      // Remove tensor types, only storing individual components.
      std::vector<TensorComponent> components;
      components.reserve(34);

      components.emplace_back(element_name + Tags::TildeD::name(),
                              tilde_d.get());
      components.emplace_back(element_name + Tags::TildeTau::name(),
                              tilde_tau.get());
      components.emplace_back(element_name + Tags::TildePhi::name(),
                              tilde_phi.get());
      components.emplace_back(
          element_name + hydro::Tags::RestMassDensity<DataVector>::name(),
          rest_mass_density.get());
      components.emplace_back(
          element_name +
              hydro::Tags::SpecificInternalEnergy<DataVector>::name(),
          specific_internal_energy.get());
      components.emplace_back(
          element_name +
              hydro::Tags::DivergenceCleaningField<DataVector>::name(),
          divergence_cleaning_field.get());
      components.emplace_back(
          element_name + hydro::Tags::Pressure<DataVector>::name(),
          pressure.get());
      components.emplace_back(
          element_name + hydro::Tags::LorentzFactor<DataVector>::name(),
          lorentz_factor.get());
      components.emplace_back(
          element_name + hydro::Tags::SpecificEnthalpy<DataVector>::name(),
          specific_enthalpy.get());

      using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;
      DataVector error =
          tuples::get<hydro::Tags::RestMassDensity<DataVector>>(exact_solution)
              .get() -
          rest_mass_density.get();
      const double rest_mass_density_error =
          alg::accumulate(error, 0.0, PlusSquare{});
      components.emplace_back(
          element_name + "Error" +
              hydro::Tags::RestMassDensity<DataVector>::name(),
          error);
      error = tuples::get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                  exact_solution)
                  .get() -
              specific_internal_energy.get();
      const double specific_internal_energy_error =
          alg::accumulate(error, 0.0, PlusSquare{});
      components.emplace_back(
          element_name + "Error" +
              hydro::Tags::SpecificInternalEnergy<DataVector>::name(),
          error);
      error =
          tuples::get<hydro::Tags::Pressure<DataVector>>(exact_solution).get() -
          pressure.get();
      const double pressure_error =
          alg::accumulate(error, 0.0, PlusSquare{});
      components.emplace_back(
          element_name + "Error" + hydro::Tags::Pressure<DataVector>::name(),
          error);
      for (size_t d = 0; d < Dim; ++d) {
        const std::string component_suffix =
            d == 0 ? "_x" : d == 1 ? "_y" : "_z";
        components.emplace_back(
            element_name + Tags::TildeS<>::name() + component_suffix,
            tilde_s.get(d));
        components.emplace_back(
            element_name + Tags::TildeB<>::name() + component_suffix,
            tilde_b.get(d));
        components.emplace_back(
            element_name +
                hydro::Tags::SpatialVelocity<DataVector, Dim>::name() +
                component_suffix,
            spatial_velocity.get(d));
        components.emplace_back(
            element_name + hydro::Tags::MagneticField<DataVector, Dim>::name() +
                component_suffix,
            magnetic_field.get(d));

        error = tuples::get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(
                    exact_solution)
                    .get(d) -
                spatial_velocity.get(d);
        components.emplace_back(
            element_name + "Error" +
                hydro::Tags::SpatialVelocity<DataVector, Dim>::name() +
                component_suffix,
            error);
        error = tuples::get<hydro::Tags::MagneticField<DataVector, Dim>>(
                    exact_solution)
                    .get(d) -
                magnetic_field.get(d);
        components.emplace_back(
            element_name + "Error" +
                hydro::Tags::MagneticField<DataVector, Dim>::name() +
                component_suffix,
            error);
        components.emplace_back(
            element_name + ::Tags::Coordinates<Dim, Frame::Inertial>::name() +
                component_suffix,
            inertial_coordinates.get(d));
      }

      // Send data to volume observer
      auto& local_observer =
          *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
               cache)
               .ckLocalBranch();
      Parallel::simple_action<observers::Actions::ContributeVolumeData>(
          local_observer, observers::ObservationId(time),
          observers::ArrayComponentId(
              std::add_pointer_t<ParallelComponent>{nullptr},
              Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
          std::move(components), extents);

      // Send data to reduction observer
      using Redum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                             funcl::Sqrt<funcl::Divides<>>,
                                             std::index_sequence<1>>;
      using ReData = Parallel::ReductionData<
          Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
          Parallel::ReductionDatum<size_t, funcl::Plus<>>, Redum, Redum, Redum>;
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, observers::ObservationId(time),
          std::vector<std::string>{
              "Time", "NumberOfPoints", "RestMassDensityError",
              "SpecificInternalEnergyError", "PressureError"},
          ReData{time.value(),
                db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points(),
                 rest_mass_density_error, specific_internal_energy_error,
                 pressure_error});
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace ValenciaDivClean
}  // namespace grmhd
