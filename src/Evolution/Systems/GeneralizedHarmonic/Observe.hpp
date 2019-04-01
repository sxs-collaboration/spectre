// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GeneralizedHarmonic {
namespace Actions {

namespace observe_detail {}  // namespace observe_detail

/*!
 * \brief Temporary action for observing volume and reduction data
 *
 * A few notes:
 * - Writes the solution and error in \f$\Psi_{ab}, \Pi_{ab}\f$, and
 * \f$\Phi_{iab}\f$ to disk as volume data.
 * - The RMS error of \f$\Psi\f$ and \f$\Pi\f$ are written to disk.
 */
struct Observe {
 private:
  using reduction_datum =
      Parallel::ReductionDatum<double, funcl::Plus<>,
                               funcl::Sqrt<funcl::Divides<>>,
                               std::index_sequence<1>>;
  using reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>, reduction_datum,
      reduction_datum, reduction_datum, reduction_datum, reduction_datum,
      reduction_datum, reduction_datum, reduction_datum>;

 public:
  struct ObserveNSlabs {
    using type = size_t;
    static constexpr OptionString help = {"Observe every Nth slab"};
  };
  struct ObserveAtT0 {
    using type = bool;
    static constexpr OptionString help = {"If true observe at t=0"};
  };

  using const_global_cache_tags = tmpl::list<ObserveNSlabs, ObserveAtT0>;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<reduction_data>>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& time_id = db::get<::Tags::TimeId>(box);
    if (time_id.substep() != 0 or (time_id.slab_number() == 0 and
                                   not Parallel::get<ObserveAtT0>(cache))) {
      return std::forward_as_tuple(std::move(box));
    }

    const auto& time = time_id.time();
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';

    if (time_id.slab_number() >= 0 and time_id.time().is_at_slab_start() and
        static_cast<size_t>(time_id.slab_number()) %
                Parallel::get<ObserveNSlabs>(cache) ==
            0) {
      const auto& extents = db::get<::Tags::Mesh<Dim>>(box).extents();
      const auto& num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();
      // Retrieve the tensors and compute the solution error.
      const auto& psi =
          db::get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>>(box);
      const auto& phi =
          db::get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(box);
      const auto& pi =
          db::get<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(box);
      const auto& gauge_constraint = db::get<
          GeneralizedHarmonic::Tags::GaugeConstraint<Dim, Frame::Inertial>>(
          box);
      const auto& three_index_constraint =
          db::get<GeneralizedHarmonic::Tags::ThreeIndexConstraint<
              Dim, Frame::Inertial>>(box);
      const auto& constraint_energy = db::get<
          GeneralizedHarmonic::Tags::ConstraintEnergy<Dim, Frame::Inertial>>(
          box);

      const auto& inertial_coordinates =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      const auto& gauge_H =
          db::get<GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>>(box);

      // Compute the error in the solution, and generate tensor component list.
      using solution_tag = OptionTags::AnalyticSolutionBase;

      const auto& exact_solution = Parallel::get<solution_tag>(cache);

      const auto& exact_solution_variables = exact_solution.variables(
          inertial_coordinates, time.value(),
          gr::analytic_solution_tags<Dim, DataVector>{});
      const auto& exact_lapse =
          get<gr::Tags::Lapse<DataVector>>(exact_solution_variables);
      const auto& exact_dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(
          exact_solution_variables);
      const auto& exact_deriv_lapse =
          get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                            Frame::Inertial>>(exact_solution_variables);
      const auto& exact_shift =
          get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(
              exact_solution_variables);
      const auto& exact_dt_shift =
          get<::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>>(
              exact_solution_variables);
      const auto& exact_deriv_shift =
          get<::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
                            tmpl::size_t<Dim>, Frame::Inertial>>(
              exact_solution_variables);
      const auto& exact_spatial_metric =
          get<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>(
              exact_solution_variables);
      const auto& exact_dt_spatial_metric = get<::Tags::dt<
          gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>>(
          exact_solution_variables);
      const auto& exact_deriv_spatial_metric = get<::Tags::deriv<
          gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
          tmpl::size_t<Dim>, Frame::Inertial>>(exact_solution_variables);

      const auto& exact_psi =
          gr::spacetime_metric(exact_lapse, exact_shift, exact_spatial_metric);
      const auto& exact_phi = GeneralizedHarmonic::phi(
          exact_lapse, exact_deriv_lapse, exact_shift, exact_deriv_shift,
          exact_spatial_metric, exact_deriv_spatial_metric);
      const auto& exact_pi = GeneralizedHarmonic::pi(
          exact_lapse, exact_dt_lapse, exact_shift, exact_dt_shift,
          exact_spatial_metric, exact_dt_spatial_metric, exact_phi);

      // Remove tensor types, only storing individual components.
      std::vector<TensorComponent> components;
      components.reserve(8);  // FIXME

      using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;

      // FIX ME: don't copy, just initialize to the right size
      DataVector error_in_psi_components{num_grid_points, 0.};
      DataVector error_in_phi_components{num_grid_points, 0.};
      DataVector error_in_pi_components{num_grid_points, 0.};
      DataVector gauge_constraint_all_components{num_grid_points, 0.};
      DataVector three_index_constraint_all_components{num_grid_points, 0.};
      for (size_t a = 0; a < Dim + 1; ++a) {
        gauge_constraint_all_components += square(gauge_constraint.get(a));
        for (size_t b = 0; b < Dim + 1; ++b) {
          error_in_psi_components +=
              square(exact_psi.get(a, b) - psi.get(a, b));
          error_in_pi_components += square(exact_pi.get(a, b) - pi.get(a, b));
          for (size_t i = 0; i < Dim; ++i) {
            three_index_constraint_all_components +=
                square(three_index_constraint.get(i, a, b));
            error_in_phi_components +=
                square(exact_phi.get(i, a, b) - phi.get(i, a, b));
          }
        }
      }
      const double psi_error =
          alg::accumulate(error_in_psi_components, 0., PlusSquare{});
      const double phi_error =
          alg::accumulate(error_in_phi_components, 0., PlusSquare{});
      const double pi_error =
          alg::accumulate(error_in_pi_components, 0., PlusSquare{});
      const double gauge_constraint_cumulative =
          alg::accumulate(gauge_constraint_all_components, 0., PlusSquare{});
      const double three_index_constraint_cumulative = alg::accumulate(
          three_index_constraint_all_components, 0., PlusSquare{});
      const double constraint_energy_cumulative =
          alg::accumulate(get(constraint_energy), 0., PlusSquare{});

      const double gauge_H_t =
          alg::accumulate(get<0>(gauge_H), 0., PlusSquare{});
      const double gauge_H_x =
          alg::accumulate(get<1>(gauge_H), 0., PlusSquare{});

      components.emplace_back(
          element_name + "Error" +
              gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>::name(),
          error_in_psi_components);
      components.emplace_back(
          element_name + "Error" +
              GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>::name(),
          error_in_phi_components);
      components.emplace_back(
          element_name + "Error" +
              GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>::name(),
          error_in_pi_components);
      components.emplace_back(element_name + "L2Norm" +
                                  GeneralizedHarmonic::Tags::GaugeConstraint<
                                      Dim, Frame::Inertial>::name(),
                              gauge_constraint_all_components);
      components.emplace_back(
          element_name + "L2Norm" +
              GeneralizedHarmonic::Tags::ThreeIndexConstraint<
                  Dim, Frame::Inertial>::name(),
          three_index_constraint_all_components);

      for (size_t d = 0; d < Dim; ++d) {
        const std::string component_suffix =
            d == 0 ? "_x" : d == 1 ? "_y" : "_z";
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
          local_observer,
          observers::ObservationId(
              time, typename Metavariables::element_observation_type{}),
          std::string{"/element_data"},
          observers::ArrayComponentId(
              std::add_pointer_t<ParallelComponent>{nullptr},
              Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
          std::move(components), extents);

      // Send data to reduction observer
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer,
          observers::ObservationId(
              time, typename Metavariables::element_observation_type{}),
          std::string{"/element_data"},
          std::vector<std::string>{
              "Time", "NumberOfPoints", "PsiError", "PhiError", "PiError",
              "L2NormGaugeConstraint", "L2NormThreeIndexConstraint",
              "L2NormConstraintEnergy", "L2NormHt", "L2NormHx"},
          reduction_data{
              time.value(),
              db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points(),
              psi_error, phi_error, pi_error, gauge_constraint_cumulative,
              three_index_constraint_cumulative, constraint_energy_cumulative,
              gauge_H_t, gauge_H_x});
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
