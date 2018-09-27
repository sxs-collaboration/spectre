// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarWave {
/// %Actions specific to scalar wave evolutions.
namespace Actions {
/*!
 * \brief Temporary action for observing volume and reduction data
 *
 * A few notes:
 * - Observation frequency is currently hard-coded and must manually be updated.
 *   Look for `time_by_timestep_value` to update.
 * - Writes the solution and error in \f$\Psi, \Pi\f$, and \f$\Phi_i\f$ to disk
 *   as volume data.
 * - The RMS error of \f$\Psi\f$ and \f$\Pi\f$ are written to disk.
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
    const auto& time = db::get<Tags::Time>(box);
    // Note: this currently assumes a constant time step that is a power of ten.
    const size_t time_by_timestep_value = static_cast<size_t>(
        std::round(time.value() / db::get<Tags::TimeStep>(box).value()));

    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';
    // We hard-code the writing frequency to large time values to avoid breaking
    // the tests.
    if (time_by_timestep_value % 1000 == 0 and time_by_timestep_value > 0) {
      const auto& extents = db::get<Tags::Mesh<Dim>>(box).extents();
      // Retrieve the tensors and compute the solution error.
      const auto& psi = db::get<ScalarWave::Psi>(box);
      const auto& pi = db::get<ScalarWave::Pi>(box);
      const auto& phi = db::get<ScalarWave::Phi<Dim>>(box);
      const auto& inertial_coordinates =
          db::get<Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Compute the error in the solution, and generate tensor component list.
      using Vars = typename Metavariables::system::variables_tag::type;
      using solution_tag = OptionTags::AnalyticSolutionBase;
      const auto exact_solution = Parallel::get<solution_tag>(cache).variables(
          inertial_coordinates, time.value(), typename Vars::tags_list{});

      // Remove tensor types, only storing individual components.
      std::vector<TensorComponent> components;
      components.reserve(3 * Dim + 4);

      components.emplace_back(element_name + ScalarWave::Psi::name(),
                              psi.get());
      using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;
      DataVector error =
          tuples::get<ScalarWave::Psi>(exact_solution).get() - psi.get();
      const double psi_error = alg::accumulate(error, 0.0, PlusSquare{});
      components.emplace_back(element_name + "Error" + ScalarWave::Psi::name(),
                              error);
      components.emplace_back(element_name + ScalarWave::Pi::name(), pi.get());
      error = tuples::get<ScalarWave::Pi>(exact_solution).get() - pi.get();
      const double pi_error = alg::accumulate(error, 0.0, PlusSquare{});
      components.emplace_back(element_name + "Error" + ScalarWave::Pi::name(),
                              error);
      for (size_t d = 0; d < Dim; ++d) {
        const std::string component_suffix =
            d == 0 ? "_x" : d == 1 ? "_y" : "_z";
        components.emplace_back(
            element_name + ScalarWave::Phi<Dim>::name() + component_suffix,
            phi.get(d));

        error = tuples::get<ScalarWave::Phi<Dim>>(exact_solution).get(d) -
                phi.get(d);
        components.emplace_back(element_name + "Error" +
                                    ScalarWave::Phi<Dim>::name() +
                                    component_suffix,
                                error);
        components.emplace_back(
            element_name + Tags::Coordinates<Dim, Frame::Inertial>::name() +
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
          Parallel::ReductionDatum<size_t, funcl::Plus<>>, Redum, Redum>;
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, observers::ObservationId(time),
          std::vector<std::string>{"Time", "NumberOfPoints", "PsiError",
                                   "PiError"},
          ReData{time.value(),
                 db::get<Tags::Mesh<Dim>>(box).number_of_grid_points(),
                 psi_error, pi_error});
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace ScalarWave
