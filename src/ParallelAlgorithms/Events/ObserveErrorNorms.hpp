// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace dg {
namespace Events {
template <typename ObservationValueTag, typename Tensors,
          typename EventRegistrars>
class ObserveErrorNorms;

namespace Registrars {
template <typename ObservationValueTag, typename Tensors>
using ObserveErrorNorms =
    ::Registration::Registrar<Events::ObserveErrorNorms, ObservationValueTag,
                              Tensors>;
}  // namespace Registrars

template <typename ObservationValueTag, typename Tensors,
          typename EventRegistrars = tmpl::list<
              Registrars::ObserveErrorNorms<ObservationValueTag, Tensors>>>
class ObserveErrorNorms;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe the RMS errors in the tensors compared to their
 * analytic solution.
 *
 * Writes reduction quantities:
 * - `ObservationValueTag`
 * - `NumberOfPoints` = total number of points in the domain
 * - `Error(*)` = RMS errors in `Tensors` =
 *   \f$\operatorname{RMS}\left(\sqrt{\sum_{\text{independent components}}\left[
 *   \text{value} - \text{analytic solution}\right]^2}\right)\f$
 *   over all points
 */
template <typename ObservationValueTag, typename... Tensors,
          typename EventRegistrars>
class ObserveErrorNorms<ObservationValueTag, tmpl::list<Tensors...>,
                        EventRegistrars> : public Event<EventRegistrars> {
 private:
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
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveErrorNorms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveErrorNorms);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName>;
  static constexpr Options::String help =
      "Observe the RMS errors in the tensors compared to their analytic\n"
      "solution (if one is available).\n"
      "\n"
      "Writes reduction quantities:\n"
      " * ObservationValueTag\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * Error(*) = RMS errors in Tensors (see online help details)\n"
      "\n"
      "Warning: Currently, only one reduction observation event can be\n"
      "triggered at a given observation value.  Causing multiple events to\n"
      "run at once will produce unpredictable results.";

  ObserveErrorNorms() = default;
  explicit ObserveErrorNorms(const std::string& subfile_name) noexcept;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags = tmpl::list<ObservationValueTag, Tensors...,
                                   ::Tags::AnalyticSolutionsBase>;

  template <typename OptionalAnalyticSolutions, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const typename Tensors::type&... tensors,
                  const OptionalAnalyticSolutions& optional_analytic_solutions,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const noexcept {
    if constexpr (tt::is_a_v<std::optional, OptionalAnalyticSolutions>) {
      if (not optional_analytic_solutions.has_value()) {
        // Nothing to do if we don't have analytic solutions. We may generalize
        // this event to observe norms of other quantities.
        return;
      }
    }
    const auto& analytic_solutions =
        [&optional_analytic_solutions]() noexcept -> decltype(auto) {
      // If we generalize this event to do observations of non-solution
      // quantities then we can return a std::optional<std::reference_wrapper>
      // here (using std::cref).
      if constexpr (tt::is_a_v<std::optional, OptionalAnalyticSolutions>) {
        return *optional_analytic_solutions;
      } else {
        return optional_analytic_solutions;
      }
    }();

    tuples::TaggedTuple<LocalSquareError<Tensors>...> local_square_errors;
    const auto record_errors = [&local_square_errors, &analytic_solutions](
                                   const auto tensor_tag_v,
                                   const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      double local_square_error = 0.0;
      for (size_t i = 0; i < tensor.size(); ++i) {
        const auto error = tensor[i] - get<::Tags::Analytic<tensor_tag>>(
                                           analytic_solutions)[i];
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
    Parallel::simple_action<observers::Actions::ContributeReductionData<observers::ThreadedActions::WriteReductionData>>(
        local_observer,
        observers::ObservationId(observation_value, subfile_path_ + ".dat"),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ArrayIndex>(array_index)},
        subfile_path_,
        std::vector<std::string>{db::tag_name<ObservationValueTag>(),
                                 "NumberOfPoints",
                                 ("Error(" + db::tag_name<Tensors>() + ")")...},
        ReductionData{
            static_cast<double>(observation_value), num_points,
            std::move(get<LocalSquareError<Tensors>>(local_square_errors))...});
  }

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey(subfile_path_ + ".dat")};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event<EventRegistrars>::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_;
};

template <typename ObservationValueTag, typename... Tensors,
          typename EventRegistrars>
ObserveErrorNorms<ObservationValueTag, tmpl::list<Tensors...>,
                  EventRegistrars>::ObserveErrorNorms(const std::string&
                                                          subfile_name) noexcept
    : subfile_path_("/" + subfile_name) {}

/// \cond
template <typename ObservationValueTag, typename... Tensors,
          typename EventRegistrars>
PUP::able::PUP_ID ObserveErrorNorms<ObservationValueTag, tmpl::list<Tensors...>,
                                    EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
