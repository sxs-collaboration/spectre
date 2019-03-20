// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace dg {
namespace Events {
template <size_t VolumeDim, typename Tensors, typename EventRegistrars>
class ObserveErrorNorms;

namespace Registrars {
template <size_t VolumeDim, typename Tensors>
struct ObserveErrorNorms {
  template <typename RegistrarList>
  using f = Events::ObserveErrorNorms<VolumeDim, Tensors, RegistrarList>;
};
}  // namespace Registrars

template <size_t VolumeDim, typename Tensors,
          typename EventRegistrars =
              tmpl::list<Registrars::ObserveErrorNorms<VolumeDim, Tensors>>>
class ObserveErrorNorms;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe the RMS errors in the tensors compared to their
 * analytic solution.
 *
 * Writes reduction quantities:
 * - `Time`
 * - `NumberOfPoints` = total number of points in the domain
 * - `Error(*)` = RMS errors in `Tensors` =
 *   \f$\operatorname{RMS}\left(\sqrt{\sum_{\text{independent components}}\left[
 *   \text{value} - \text{analytic solution}\right]^2}\right)\f$
 *   over all points
 *
 * \warning Currently, only one reduction observation event can be
 * triggered at a given time.  Causing multiple events to run at once
 * will produce unpredictable results.
 */
template <size_t VolumeDim, typename... Tensors, typename EventRegistrars>
class ObserveErrorNorms<VolumeDim, tmpl::list<Tensors...>, EventRegistrars>
    : public Event<EventRegistrars> {
 private:
  using coordinates_tag = ::Tags::Coordinates<VolumeDim, Frame::Inertial>;

  template <typename Tag>
  struct LocalSquareError {
    using type = double;
  };

  using L2ErrorDatum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                funcl::Sqrt<funcl::Divides<>>,
                                                std::index_sequence<1>>;
  using ReductionData = tmpl::wrap<
      tmpl::append<
          tmpl::list<Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
                     Parallel::ReductionDatum<size_t, funcl::Plus<>>>,
          tmpl::filled_list<L2ErrorDatum, sizeof...(Tensors)>>,
      Parallel::ReductionData>;

 public:
  /// \cond
  explicit ObserveErrorNorms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveErrorNorms);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help =
      "Observe the RMS errors in the tensors compared to their analytic\n"
      "solution.\n"
      "\n"
      "Writes reduction quantities:\n"
      " * Time\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * Error(*) = RMS errors in Tensors (see online help details)\n"
      "\n"
      "Warning: Currently, only one reduction observation event can be\n"
      "triggered at a given time.  Causing multiple events to run at once\n"
      "will produce unpredictable results.";

  ObserveErrorNorms() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags = tmpl::list<::Tags::Time, coordinates_tag, Tensors...>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const Time& time,
                  const db::item_type<coordinates_tag>& inertial_coordinates,
                  const db::item_type<Tensors>&... tensors,
                  Parallel::ConstGlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*meta*/) const noexcept {
    const auto analytic_solution =
        Parallel::get<OptionTags::AnalyticSolutionBase>(cache).variables(
            inertial_coordinates, time.value(), tmpl::list<Tensors...>{});

    tuples::TaggedTuple<LocalSquareError<Tensors>...> local_square_errors;
    const auto record_errors = [&analytic_solution, &local_square_errors](
        const auto tensor_tag_v, const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      double local_square_error = 0.0;
      for (size_t i = 0; i < tensor.size(); ++i) {
        const auto error = tensor[i] - get<tensor_tag>(analytic_solution)[i];
        local_square_error += alg::accumulate(square(error), 0.0);
      }
      get<LocalSquareError<tensor_tag>>(local_square_errors) =
          local_square_error;
      return 0;
    };
    expand_pack(record_errors(tmpl::type_<Tensors>{}, tensors)...);
    const size_t num_points = get_first_argument(tensors...).begin()->size();

    // Send data to reduction observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            time.value(), typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Time", "NumberOfPoints",
                                 ("Error(" + Tensors::name() + ")")...},
        ReductionData{
            time.value(), num_points,
            std::move(get<LocalSquareError<Tensors>>(local_square_errors))...});
  }
};

/// \cond
template <size_t VolumeDim, typename... Tensors, typename EventRegistrars>
PUP::able::PUP_ID ObserveErrorNorms<VolumeDim, tmpl::list<Tensors...>,
                                    EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
