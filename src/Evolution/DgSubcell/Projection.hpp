// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t>
class Index;
template <size_t>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void project_impl(gsl::span<double> subcell_u, gsl::span<const double> dg_u,
                  const Mesh<Dim>& dg_mesh,
                  const Index<Dim>& subcell_extents) noexcept;
}  // namespace detail

/// @{
/*!
 * \ingroup DgSubcellGroup
 * \brief Project the variable `dg_u` onto the subcell grid with extents
 * `subcell_extents`.
 *
 * \note In the return-by-`gsl::not_null` with `Variables` interface, the
 * `SubcellTagList` and the `DgtagList` must be the same when all tag prefixes
 * are removed. Typically the `Tags::Inactive` prefix will be used.
 */
template <size_t Dim>
DataVector project(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                   const Index<Dim>& subcell_extents) noexcept;

template <size_t Dim>
void project(gsl::not_null<DataVector*> subcell_u, const DataVector& dg_u,
             const Mesh<Dim>& dg_mesh,
             const Index<Dim>& subcell_extents) noexcept;

template <typename SubcellTagList, typename DgTagList, size_t Dim>
void project(const gsl::not_null<Variables<SubcellTagList>*> subcell_u,
             const Variables<DgTagList>& dg_u, const Mesh<Dim>& dg_mesh,
             const Index<Dim>& subcell_extents) noexcept {
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
  detail::project_impl(gsl::span<double>{subcell_u->data(), subcell_u->size()},
                       gsl::span<const double>{dg_u.data(), dg_u.size()},
                       dg_mesh, subcell_extents);
}

template <typename TagList, size_t Dim>
Variables<TagList> project(const Variables<TagList>& dg_u,
                           const Mesh<Dim>& dg_mesh,
                           const Index<Dim>& subcell_extents) noexcept {
  Variables<TagList> subcell_u(subcell_extents.product());
  project(make_not_null(&subcell_u), dg_u, dg_mesh, subcell_extents);
  return subcell_u;
}
/// @}
}  // namespace evolution::dg::subcell::fd
