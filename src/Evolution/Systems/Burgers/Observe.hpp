// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Burgers {
namespace Actions {

/*!
 * \brief Temporary action for observing volume and reduction data
 *
 * A few notes:
 * - Writes the solution and error in \f$U\f$ to disk.
 * - The RMS error of \f$U\f$ is written to disk.
 */
struct Observe {
 private:
  using l2_error_datum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                  funcl::Sqrt<funcl::Divides<>>,
                                                  std::index_sequence<1>>;
  using reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>, l2_error_datum>;

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
      // Retrieve the tensors and compute the solution error.
      const auto& u = db::get<Burgers::Tags::U>(box);
      const auto& inertial_coordinates =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Compute the error in the solution, and generate tensor component list.
      using Vars = typename Metavariables::system::variables_tag::type;
      using solution_tag = OptionTags::AnalyticSolutionBase;
      const auto exact_solution = Parallel::get<solution_tag>(cache).variables(
          inertial_coordinates, time.value(), typename Vars::tags_list{});

      // Remove tensor types, only storing individual components.
      std::vector<TensorComponent> components;
      // U, UError, x
      components.reserve(3);

      components.emplace_back(element_name + Burgers::Tags::U::name(), u.get());
      using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;
      DataVector error =
          tuples::get<Burgers::Tags::U>(exact_solution).get() - u.get();
      const double u_error = alg::accumulate(error, 0.0, PlusSquare{});
      components.emplace_back(element_name + "Error" + Burgers::Tags::U::name(),
                              error);
      components.emplace_back(
          element_name + ::Tags::Coordinates<1, Frame::Inertial>::name() + "_x",
          get<0>(inertial_coordinates));

      // Send data to volume observer
      auto& local_observer =
          *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
               cache)
               .ckLocalBranch();
      Parallel::simple_action<observers::Actions::ContributeVolumeData>(
          local_observer, observers::ObservationId(time),
          std::string{"/element_data"},
          observers::ArrayComponentId(
              std::add_pointer_t<ParallelComponent>{nullptr},
              Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
          std::move(components), extents);

      // Send data to reduction observer
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, observers::ObservationId(time),
          std::string{"/element_data"},
          std::vector<std::string>{"Time", "NumberOfPoints", "UError"},
          reduction_data{
              time.value(),
              db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points(),
              u_error});
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Burgers
