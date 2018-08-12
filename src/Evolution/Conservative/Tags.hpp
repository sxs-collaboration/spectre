// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tags::Flux
/// \endcond

namespace Tags {

/// \ingroup ConservativeGroup
/// \ingroup DataBoxTagsGroup
/// \brief Prefix computing a boundary unit normal vector dotted into
/// the flux from a flux on the boundary.
template <typename Tag, size_t VolumeDim, typename Fr>
struct ComputeNormalDotFlux : db::add_tag_prefix<NormalDotFlux, Tag>,
                              db::ComputeTag {
  using base = db::add_tag_prefix<NormalDotFlux, Tag>;

 private:
  using flux_tag = db::add_tag_prefix<Flux, Tag, tmpl::size_t<VolumeDim>, Fr>;
  using normal_tag =
      Tags::Normalized<Tags::UnnormalizedFaceNormal<VolumeDim, Fr>>;

 public:
  static auto function(const db::item_type<flux_tag>& flux,
                       const db::item_type<normal_tag>& normal) noexcept {
    using tags_list = typename db::item_type<Tag>::tags_list;
    auto result = make_with_value<
        ::Variables<db::wrap_tags_in<NormalDotFlux, tags_list>>>(flux, 0.);

    tmpl::for_each<tags_list>([&result, &flux,
                               &normal ](auto local_tag) noexcept {
      using tensor_tag = tmpl::type_from<decltype(local_tag)>;
      auto& result_tensor = get<NormalDotFlux<tensor_tag>>(result);
      const auto& flux_tensor =
          get<Flux<tensor_tag, tmpl::size_t<VolumeDim>, Fr>>(flux);
      for (auto it = result_tensor.begin(); it != result_tensor.end(); ++it) {
        const auto result_indices = result_tensor.get_tensor_index(it);
        for (size_t d = 0; d < VolumeDim; ++d) {
          *it += normal.get(d) * flux_tensor.get(prepend(result_indices, d));
        }
      }
    });
    return result;
  }
  using argument_tags = tmpl::list<flux_tag, normal_tag>;
};
}  // namespace Tags
