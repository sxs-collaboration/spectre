// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/IdPair.hpp" // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BlockId.hpp" // IWYU pragma: keep

/// \cond
template <size_t VolumeDim>
class ElementId;
template <typename TagsList>
class Variables;
/// \endcond

namespace intrp {

/// Data structures holding quantities that are interpolated by
/// `Interpolator` for use by `InterpolationTarget`s
namespace Vars {
/// \brief Holds a `Variables` interpolated onto a list of points, and
/// information about those points, for a local `Interpolator`.
///
/// `TagList` is a `tmpl::list` of tags that go into the `Variables`.
template <size_t VolumeDim, typename TagList>
struct Info {
  /// `block_coord_holders` holds the list of all points (in block
  /// logical coordinates) that need to be interpolated onto for a
  /// given `InterpolationTarget`.
  ///
  /// The number of interpolated points for which results are stored
  /// in this `Info` (in `vars` and `global_offsets` below)
  /// corresponds to only the subset of the points in
  /// `block_coord_holders` that are contained in local `Element`s.
  /// Moreover, the number of interpolated points stored in this
  /// `Info` will change as more `Elements` send data to this
  /// `Interpolator`, and will be less than or equal to the size of
  /// `block_coord_holders` even after all `Element`s have sent their
  /// data (this is because this `Info` lives only on a single core,
  /// and this core will have access only to the local `Element`s).
  std::vector<IdPair<domain::BlockId,
                     tnsr::I<double, VolumeDim, typename ::Frame::Logical>>>
      block_coord_holders;
  /// `vars` holds the interpolated `Variables` on some subset of the
  /// points in `block_coord_holders`.  The grid points inside vars
  /// are indexed according to `global_offsets` below.  The size of
  /// `vars` changes as more `Element`s send data to this `Interpolator`.
  std::vector<Variables<TagList>> vars{};
  /// `global_offsets[j][i]` is the index into `block_coord_holders` that
  /// corresponds to the index `i` of the `DataVector` held in `vars[j]`.
  /// The size of `global_offsets` changes as more `Element`s
  /// send data to this `Interpolator`.
  std::vector<std::vector<size_t>> global_offsets{};
  /// Holds the `ElementId`s of `Element`s for which interpolation has
  /// already been done for this `Info`.
  std::unordered_set<ElementId<VolumeDim>>
      interpolation_is_done_for_these_elements{};
};

/// Holds `Info`s at all `temporal_id`s for a given
/// `InterpolationTargetTag`.  Also holds `temporal_id`s when data has
/// been interpolated; this is used for cleanup purposes.  All
/// `Holder`s for all `InterpolationTargetTags` are held in a single
/// `TaggedTuple` that is in the `Interpolator`'s `DataBox` with the
/// tag `Tags::InterpolatedVarsHolders`.
template <typename Metavariables,
          typename InterpolationTargetTag, typename TagList>
struct Holder {
  std::unordered_map<
      typename Metavariables::temporal_id::type,
      Info<Metavariables::domain_dim, TagList>>
      infos;
  std::unordered_set<typename Metavariables::temporal_id::type>
      temporal_ids_when_data_has_been_interpolated;
};

/// Indexes a particular `Holder` in the `TaggedTuple` that is
/// accessed from the `Interpolator`'s `DataBox` with tag
/// `Tags::InterpolatedVarsHolders`.
template <typename InterpolationTargetTag, typename Metavariables>
struct HolderTag {
  using type =
      Holder<Metavariables, InterpolationTargetTag,
             typename InterpolationTargetTag::vars_to_interpolate_to_target>;
};

}  // namespace Vars
}  // namespace intrp
