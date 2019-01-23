// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace StrahlkorperTags {
template <typename Frame>
struct Jacobian;
template <typename Frame>
struct NormalOneForm;
template <typename Frame>
struct Radius;
template <typename Frame>
struct Rhat;
template <typename Frame>
struct Strahlkorper;
}  // namespace StrahlkorperTags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace observers {
namespace ThreadedActions {
struct WriteReductionData;
}  // namespace ThreadedActions
template <class Metavariables>
struct ObserverWriter;
}  // namespace observers
/// \endcond

namespace intrp {
namespace callbacks {

namespace detail {
template <typename T>
struct reduction_data_type;

template <typename... Ts>
struct reduction_data_type<tmpl::list<Ts...>> {
  // We use ReductionData because that is what is expected by the
  // ObserverWriter.  We do a "reduction" that involves only one
  // processing element (often equivalent to a core),
  // so AssertEqual is used here as a no-op.
  //
  // The first argument is for Time, the others are for
  // the list of scalars being integrated.
  using type = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<typename Ts::type::type::value_type,
                               funcl::AssertEqual<>>...>;
};

template <typename T>
struct reduction_data_tag_type;

template <typename... Ts>
struct reduction_data_tag_type<tmpl::list<Ts...>> {
  // The first argument is for Time, the others are for
  // the list of scalars being integrated.
  using type = observers::Tags::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<typename Ts::type::type::value_type,
                               funcl::AssertEqual<>>...>;
};

template <typename T>
using reduction_data_tag_type_t = typename reduction_data_tag_type<T>::type;

template <typename List, size_t... Is>
auto make_reduction_data(const std::array<double, sizeof...(Is)>& a,
                         std::index_sequence<Is...> /* meta */) noexcept {
  return typename reduction_data_type<List>::type(gsl::at(a, Is)...);
}

template <typename List, size_t N>
auto make_reduction_data(const std::array<double, N>& a) noexcept {
  return make_reduction_data<List>(a, std::make_index_sequence<N>{});
}
}  // namespace detail

/// \brief post_interpolation_callback that outputs
/// surface integrals on a Strahlkorper.
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `StrahlkorperTags::items_tags<Frame>`
///   - `StrahlkorperTags::compute_items_tags<Frame>`
///   - `TagsToObserve`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see InterpolationTarget for a description of InterpolationTargetTag.
template <typename TagsToObserve, typename InterpolationTargetTag,
          typename Frame>
struct ObserveSurfaceIntegrals {
  using reduction_data_tags = detail::reduction_data_tag_type_t<TagsToObserve>;

  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    // Do the integrals and construct the legend.
    const auto& strahlkorper = get<StrahlkorperTags::Strahlkorper<Frame>>(box);
    const auto area_element = StrahlkorperGr::area_element(
        get<gr::Tags::SpatialMetric<3, Frame>>(box),
        get<StrahlkorperTags::Jacobian<Frame>>(box),
        get<StrahlkorperTags::NormalOneForm<Frame>>(box),
        get<StrahlkorperTags::Radius<Frame>>(box),
        get<StrahlkorperTags::Rhat<Frame>>(box));

    // The +1 in the sizes below is because `TagsToObserve` contains
    // only the integrals, but both `legend` and `time_and_integrals` contain
    // time in addition to the integrals.
    std::vector<std::string> legend(tmpl::size<TagsToObserve>::value + 1);
    std::array<double, tmpl::size<TagsToObserve>::value + 1>
        time_and_integrals{};
    time_and_integrals[0] = temporal_id.time().value();
    legend[0] = "Time";
    size_t s = 1;
    tmpl::for_each<TagsToObserve>([&](auto tag_v) noexcept {
      using Tag = typename decltype(tag_v)::type;
      const auto& scalar = get<Tag>(box);
      gsl::at(time_and_integrals, s) =
          StrahlkorperGr::surface_integral_of_scalar(area_element, scalar,
                                                     strahlkorper);
      legend[s] = Tag::name();
      ++s;
    });

    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
        proxy[0], observers::ObservationId(temporal_id.time()),
        std::string{"/" + pretty_type::short_name<InterpolationTargetTag>() +
                    "_integrals"},
        legend, detail::make_reduction_data<TagsToObserve>(time_and_integrals));
  }
};
}  // namespace callbacks
}  // namespace intrp
