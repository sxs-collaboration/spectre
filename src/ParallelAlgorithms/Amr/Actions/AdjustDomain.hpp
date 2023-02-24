// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// #include "Parallel/Printf.hpp"

namespace amr {
template <size_t Dim, typename System>
struct Projector;
}

namespace amr::Actions {
struct AdjustDomain {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    const auto& my_amr_flags = db::get<amr::Tags::Flags<volume_dim>>(box);
    auto& element_array =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    auto my_proxy = element_array[array_index];

    if (alg::any_of(my_amr_flags,
                    [](amr::Flag flag) { return flag == amr::Flag::Split; })) {
      ERROR("h-refinement not supported yet\n");
    } else if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
                 return flag == amr::Flag::Join;
               })) {
      ERROR("h-refinement not supported yet\n");
    } else if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
                 return (flag == amr::Flag::IncreaseResolution or
                         flag == amr::Flag::DecreaseResolution);
               })) {
      db::mutate_apply<
          amr::Projector<volume_dim, typename Metavariables::system>>(
          make_not_null(&box));
    }
    db::mutate<amr::Tags::Flags<volume_dim>,
               amr::Tags::NeighborFlags<volume_dim>>(
        make_not_null(&box),
        [](const gsl::not_null<std::array<amr::Flag, volume_dim>*> amr_flags,
           const gsl::not_null<std::unordered_map<
               ElementId<volume_dim>, std::array<amr::Flag, volume_dim>>*>
               amr_flags_of_neighbors) {
          amr_flags_of_neighbors->clear();
          for (size_t d = 0; d < volume_dim; ++d) {
            (*amr_flags)[d] = amr::Flag::Undefined;
          }
        });
  }
};
}  // namespace amr::Actions
