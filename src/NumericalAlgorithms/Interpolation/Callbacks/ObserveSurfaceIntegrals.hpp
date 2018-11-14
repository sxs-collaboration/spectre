// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
/// \endcond

namespace intrp {
namespace callbacks {

/// \brief post_interpolation_callback that outputs
/// surface integrals on a Strahlkorper.
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - ConstGlobalCache:
///   - `FileNameTag`
/// - DataBox:
///   - `StrahlkorperTags::items_tags<Frame>`
///   - `StrahlkorperTags::compute_items_tags<Frame>`
///   - `TagsToObserve`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see InterpolationTarget for a description of InterpolationTargetTag.
template <typename TagsToObserve, typename FileNameTag, typename Frame>
struct ObserveSurfaceIntegrals {
  using const_global_cache_tags = tmpl::list<FileNameTag>;
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    // Do the integrals and construct the legend.
    const auto& strahlkorper = get<StrahlkorperTags::Strahlkorper<Frame>>(box);
    const auto area_element = StrahlkorperGr::area_element(
        get<gr::Tags::SpatialMetric<3, Frame>>(box),
        get<StrahlkorperTags::Jacobian<Frame>>(box),
        get<StrahlkorperTags::NormalOneForm<Frame>>(box),
        get<StrahlkorperTags::Radius<Frame>>(box),
        get<StrahlkorperTags::Rhat<Frame>>(box));
    std::vector<double> time_and_integrals(tmpl::size<TagsToObserve>::value +
                                           1);
    std::vector<std::string> legend(tmpl::size<TagsToObserve>::value + 1);
    time_and_integrals[0] = temporal_id.time().value();
    legend[0] = "Time";
    size_t s = 1;
    tmpl::for_each<TagsToObserve>([&](auto tag) noexcept {
      using Tag = typename decltype(tag)::type;
      const auto& scalar = get<Tag>(box);
      time_and_integrals[s] = StrahlkorperGr::surface_integral_of_scalar(
          area_element, scalar, strahlkorper);
      legend[s] = Tag::name();
      ++s;
    });

    // Write to a file.
    // Currently there is no file locking.
    // When issue #1198 is resolved we will replace the file-writing here with
    // calls to the observer infrastructure, which takes care of the file
    // locking.
    const auto& file_prefix = Parallel::get<FileNameTag>(cache);
    h5::H5File<h5::AccessType::ReadWrite> h5file(file_prefix + ".h5", true);
    constexpr size_t version_number = 0;
    auto& time_series_file = h5file.try_insert<h5::Dat>(
        "/surface_integrals", std::move(legend), version_number);
    time_series_file.append(time_and_integrals);
  }
};
}  // namespace callbacks
}  // namespace intrp
