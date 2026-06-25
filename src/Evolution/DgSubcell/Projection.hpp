// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/ParityFromSymmetry.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t>
class Index;
template <size_t>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void project_impl(gsl::span<double> subcell_u, gsl::span<const double> dg_u,
                  const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
                  Spectral::Parity parity);
template <size_t Dim>
void project_to_faces_impl(gsl::span<double> subcell_u,
                           gsl::span<const double> dg_u,
                           const Mesh<Dim>& dg_mesh,
                           const Index<Dim>& subcell_extents,
                           const size_t& face_direction,
                           Spectral::Parity parity);

/*!
 * \brief Project `dg_u` onto the subcell grid, sorting even- and odd-parity
 * components into separate batches when a ZernikeB1 basis is present.
 *
 * For non-ZernikeB1 meshes this falls through to the default implementation.
 * The `TagList` must be the full Variables tag list.
 */
template <typename TagList, size_t Dim>
void project_impl_with_tag_list(gsl::span<double> subcell_u,
                                gsl::span<const double> dg_u,
                                const Mesh<Dim>& dg_mesh,
                                const Index<Dim>& subcell_extents) {
  if (dg_mesh.basis(0) == Spectral::Basis::ZernikeB1) {
    ASSERT(Variables<TagList>::number_of_independent_components *
                   dg_mesh.number_of_grid_points() ==
               dg_u.size(),
           "Passed TagList does not have the same components, "
               << Variables<TagList>::number_of_independent_components
               << ", as dg_u holds, "
               << dg_u.size() / dg_mesh.number_of_grid_points());
    constexpr auto parity_info = Spectral::compute_parity_list<TagList>();
    constexpr auto parity_list = std::get<0>(parity_info);
    constexpr size_t num_even = std::get<1>(parity_info);
    constexpr size_t num_odd = std::get<2>(parity_info);

    const size_t num_dg_pts = dg_mesh.number_of_grid_points();
    const size_t num_subcell_pts = subcell_extents.product();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    auto buffer = cpp20::make_unique_for_overwrite<double[]>(
        (num_even + num_odd) * (num_dg_pts + num_subcell_pts));
    DataVector even_input{&buffer[0], num_even * num_dg_pts};
    DataVector odd_input(&buffer[num_even * num_dg_pts], num_odd * num_dg_pts);
    DataVector even_output(&buffer[(num_even + num_odd) * num_dg_pts],
                           num_even * num_subcell_pts);
    DataVector odd_output(
        &buffer[(num_even + num_odd) * num_dg_pts + num_even * num_subcell_pts],
        num_odd * num_subcell_pts);

    // Sort input components into even/odd parity buffers
    const double* p_in = dg_u.data();
    double* p_even_in = even_input.data();
    double* p_odd_in = odd_input.data();
    bool is_even = true;
    for (const size_t seg_size : parity_list) {
      if (seg_size == 0) {
        if (is_even) {
          is_even = false;
          continue;
        } else {
          break;
        }
      }
      if (is_even) {
        std::copy(p_in, p_in + seg_size * num_dg_pts, p_even_in);  // NOLINT
        p_even_in += seg_size * num_dg_pts;                        // NOLINT
      } else {
        std::copy(p_in, p_in + seg_size * num_dg_pts, p_odd_in);  // NOLINT
        p_odd_in += seg_size * num_dg_pts;                        // NOLINT
      }
      p_in += seg_size * num_dg_pts;  // NOLINT
      is_even = not is_even;
    }

    // Project each parity batch with the appropriate projection matrix
    if constexpr (num_even > 0) {
      project_impl(
          gsl::span<double>{even_output.data(), even_output.size()},
          gsl::span<const double>{even_input.data(), even_input.size()},
          dg_mesh, subcell_extents, Spectral::Parity::Even);
    }
    if constexpr (num_odd > 0) {
      project_impl(gsl::span<double>{odd_output.data(), odd_output.size()},
                   gsl::span<const double>{odd_input.data(), odd_input.size()},
                   dg_mesh, subcell_extents, Spectral::Parity::Odd);
    }

    // Reassemble output in original component order
    double* p_out = subcell_u.data();
    const double* p_even_out = even_output.data();
    const double* p_odd_out = odd_output.data();
    is_even = true;
    for (const size_t seg_size : parity_list) {
      if (seg_size == 0) {
        if (is_even) {
          is_even = false;
          continue;
        } else {
          break;
        }
      }
      if (is_even) {
        // NOLINTNEXTLINE
        std::copy(p_even_out, p_even_out + seg_size * num_subcell_pts, p_out);
        p_even_out += seg_size * num_subcell_pts;  // NOLINT
      } else {
        // NOLINTNEXTLINE
        std::copy(p_odd_out, p_odd_out + seg_size * num_subcell_pts, p_out);
        p_odd_out += seg_size * num_subcell_pts;  // NOLINT
      }
      p_out += seg_size * num_subcell_pts;  // NOLINT
      is_even = not is_even;
    }
    return;
  }
  project_impl(subcell_u, dg_u, dg_mesh, subcell_extents,
               Spectral::Parity::Uninitialized);
}
}  // namespace detail

/// @{
/*!
 * \ingroup DgSubcellGroup
 * \brief Project the variable `dg_u` onto the subcell grid with extents
 * `subcell_extents`.
 *
 * When the DG mesh uses a ZernikeB1 basis the Variables overloads deduce
 * per-component parity from the tag list automatically.  The raw `DataVector`
 * overloads accepting a `Spectral::Parity` are for single-component data
 * where the caller already knows the parity; the overloads accepting a
 * `tmpl::list<TagList>` meta-parameter project a multi-component DataVector
 * whose tensor-parity structure is encoded in `TagList`.
 *
 * \note In the return-by-`gsl::not_null` with `Variables` interface, the
 * `SubcellTagList` and the `DgTagList` must be the same when all tag prefixes
 * are removed.
 */
template <size_t Dim>
DataVector project(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                   const Index<Dim>& subcell_extents,
                   Spectral::Parity parity = Spectral::Parity::Uninitialized);

template <size_t Dim>
void project(gsl::not_null<DataVector*> subcell_u, const DataVector& dg_u,
             const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
             Spectral::Parity parity = Spectral::Parity::Uninitialized);

template <typename TagList, size_t Dim>
void project(const gsl::not_null<DataVector*> subcell_u, const DataVector& dg_u,
             const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
             TagList /*meta*/) {
  ASSERT(dg_u.size() % dg_mesh.number_of_grid_points() == 0,
         "The vector dg_u must have size that is a multiple of the number of "
         "grid points "
             << dg_mesh.number_of_grid_points() << " but got " << dg_u.size());
  subcell_u->destructive_resize(subcell_extents.product() * dg_u.size() /
                                dg_mesh.number_of_grid_points());
  detail::project_impl_with_tag_list<TagList>(
      gsl::span<double>{subcell_u->data(), subcell_u->size()},
      gsl::span<const double>{dg_u.data(), dg_u.size()}, dg_mesh,
      subcell_extents);
}

template <typename TagList, size_t Dim>
DataVector project(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                   const Index<Dim>& subcell_extents, TagList /*meta*/) {
  ASSERT(dg_u.size() % dg_mesh.number_of_grid_points() == 0,
         "The vector dg_u must have size that is a multiple of the number of "
         "grid points "
             << dg_mesh.number_of_grid_points() << " but got " << dg_u.size());
  DataVector subcell_u(subcell_extents.product() * dg_u.size() /
                       dg_mesh.number_of_grid_points());
  project(make_not_null(&subcell_u), dg_u, dg_mesh, subcell_extents, TagList{});
  return subcell_u;
}

template <typename SubcellTagList, typename DgTagList, size_t Dim>
void project(const gsl::not_null<Variables<SubcellTagList>*> subcell_u,
             const Variables<DgTagList>& dg_u, const Mesh<Dim>& dg_mesh,
             const Index<Dim>& subcell_extents) {
  static_assert(
      std::is_same_v<
          tmpl::transform<SubcellTagList,
                          tmpl::bind<db::remove_all_prefixes, tmpl::_1>>,
          tmpl::transform<DgTagList,
                          tmpl::bind<db::remove_all_prefixes, tmpl::_1>>>,
      "DG and subcell tag lists must be the same once prefix tags "
      "are removed.");
  ASSERT(dg_u.number_of_grid_points() == dg_mesh.number_of_grid_points(),
         "dg_u has incorrect size " << dg_u.number_of_grid_points()
                                    << " since the mesh is size "
                                    << dg_mesh.number_of_grid_points());
  if (UNLIKELY(subcell_u->number_of_grid_points() !=
               subcell_extents.product())) {
    subcell_u->initialize(subcell_extents.product());
  }
  detail::project_impl_with_tag_list<DgTagList>(
      gsl::span<double>{subcell_u->data(), subcell_u->size()},
      gsl::span<const double>{dg_u.data(), dg_u.size()}, dg_mesh,
      subcell_extents);
}

template <typename TagList, size_t Dim>
Variables<TagList> project(const Variables<TagList>& dg_u,
                           const Mesh<Dim>& dg_mesh,
                           const Index<Dim>& subcell_extents) {
  Variables<TagList> subcell_u(subcell_extents.product());
  project(make_not_null(&subcell_u), dg_u, dg_mesh, subcell_extents);
  return subcell_u;
}

template <size_t Dim>
DataVector project_to_faces(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                            const Index<Dim>& subcell_extents,
                            const size_t& face_direction,
                            Spectral::Parity parity);

template <size_t Dim>
void project_to_faces(gsl::not_null<DataVector*> subcell_u,
                      const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                      const Index<Dim>& subcell_extents,
                      const size_t& face_direction, Spectral::Parity parity);

template <typename SubcellTagList, typename DgTagList, size_t Dim>
void project_to_faces(const gsl::not_null<Variables<SubcellTagList>*> subcell_u,
                      const Variables<DgTagList>& dg_u,
                      const Mesh<Dim>& dg_mesh,
                      const Index<Dim>& subcell_extents,
                      const size_t& face_direction, Spectral::Parity parity) {
  static_assert(
      std::is_same_v<
          tmpl::transform<SubcellTagList,
                          tmpl::bind<db::remove_all_prefixes, tmpl::_1>>,
          tmpl::transform<DgTagList,
                          tmpl::bind<db::remove_all_prefixes, tmpl::_1>>>,
      "DG and subcell tag lists must be the same once prefix tags "
      "are removed.");
  ASSERT(dg_u.number_of_grid_points() == dg_mesh.number_of_grid_points(),
         "dg_u has incorrect size " << dg_u.number_of_grid_points()
                                    << " since the mesh is size "
                                    << dg_mesh.number_of_grid_points());
  if (UNLIKELY(subcell_u->number_of_grid_points() !=
               subcell_extents.product())) {
    subcell_u->initialize(subcell_extents.product());
  }
  detail::project_to_faces_impl(
      gsl::span<double>{subcell_u->data(), subcell_u->size()},
      gsl::span<const double>{dg_u.data(), dg_u.size()}, dg_mesh,
      subcell_extents, face_direction, parity);
}

template <typename TagList, size_t Dim>
Variables<TagList> project_to_faces(const Variables<TagList>& dg_u,
                                    const Mesh<Dim>& dg_mesh,
                                    const Index<Dim>& subcell_extents,
                                    const size_t& face_direction,
                                    Spectral::Parity parity) {
  Variables<TagList> subcell_u(subcell_extents.product());
  project_to_faces(make_not_null(&subcell_u), dg_u, dg_mesh, subcell_extents,
                   face_direction, parity);
  return subcell_u;
}
/// @}
}  // namespace evolution::dg::subcell::fd
