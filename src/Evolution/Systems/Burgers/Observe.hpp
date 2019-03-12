// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare Tensor
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace Burgers {
namespace Events {
template <typename EventRegistrars>
class Observe;

namespace Registrars {
using Observe = Registration::Registrar<Events::Observe>;
}  // namespace Registrars

/*!
 * \brief Observe the Burgers system
 *
 * Writes volume quantities:
 * - `InertialCoordinates`
 * - `U`
 * - `ErrorU` = \f$U - \text{analytic solution}\f$
 *
 * Writes reduction quantities:
 * - `Time`
 * - `NumberOfPoints` = total number of points in the domain
 * - `ErrorU` = \f$\operatorname{RMS}(U - \text{analytic solution})\f$
 *   over all points
 *
 * \warning Currently, only one observation event can be triggered at
 * a given time.  Causing multiple events to run at once will produce
 * unpredictable results.
 */
template <typename EventRegistrars = tmpl::list<Registrars::Observe>>
class Observe : public Event<EventRegistrars> {
 private:
  using L2ErrorDatum =
      Parallel::ReductionDatum<double, funcl::Plus<>,
                               funcl::Sqrt<funcl::Divides<>>,
                               std::index_sequence<1>>;
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>, L2ErrorDatum>;

 public:
  /// \cond
  explicit Observe(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Observe);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help =
      "Observes the Burgers field U and its error.\n"
      "\n"
      "Writes volume quantities:\n"
      " * InertialCoordinates\n"
      " * U\n"
      " * ErrorU = U - analytic solution\n"
      "\n"
      "Writes reduction quantities:\n"
      " * Time\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * ErrorU = RMS(U - analytic solution) over all points";

  Observe() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags =
      tmpl::list<::Tags::Time, ::Tags::Mesh<1>,
                 ::Tags::Coordinates<1, Frame::Inertial>, Tags::U>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const Time& time, const Mesh<1>& mesh,
      const tnsr::I<DataVector, 1, Frame::Inertial>& inertial_coordinates,
      const Scalar<DataVector>& u,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<1>& array_index,
      const ParallelComponent* const /*meta*/) const noexcept {
    const std::string element_name = MakeString{} << ElementId<1>(array_index)
                                                  << '/';
    // Compute the error in the solution, and generate tensor component list.
    using Vars = typename Metavariables::system::variables_tag::type;
    using solution_tag = OptionTags::AnalyticSolutionBase;
    const auto exact_solution = Parallel::get<solution_tag>(cache).variables(
        inertial_coordinates, time.value(), typename Vars::tags_list{});

    // Remove tensor types, only storing individual components.
    std::vector<TensorComponent> components;
    // U, ErrorU, x
    components.reserve(3);

    components.emplace_back(element_name + Tags::U::name(), get(u));
    using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;
    DataVector error = get(u) - get(tuples::get<Tags::U>(exact_solution));
    const double u_error = alg::accumulate(error, 0.0, PlusSquare{});
    components.emplace_back(element_name + "Error" + Tags::U::name(),
                            std::move(error));
    components.emplace_back(
        element_name + ::Tags::Coordinates<1, Frame::Inertial>::name() + "_x",
        get<0>(inertial_coordinates));

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(
            time.value(), typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<1>>(array_index)),
        std::move(components), mesh.extents());

    // Send data to reduction observer
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            time.value(), typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Time", "NumberOfPoints", "ErrorU"},
        ReductionData{time.value(), mesh.number_of_grid_points(), u_error});
  }
};

/// \cond
template <typename EventRegistrars>
PUP::able::PUP_ID Observe<EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace Burgers
