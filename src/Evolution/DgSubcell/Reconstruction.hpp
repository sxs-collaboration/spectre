// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
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
void reconstruct_impl(gsl::span<double> dg_u,
                      gsl::span<const double> subcell_u_times_projected_det_jac,
                      const Mesh<Dim>& dg_mesh,
                      const Index<Dim>& subcell_extents,
                      ReconstructionMethod reconstruction_method,
                      Spectral::Parity parity);

/*!
 * \brief Reconstruct `subcell_u` onto the DG grid, sorting even- and
 * odd-parity components into separate batches when a ZernikeB1 basis is
 * present.
 *
 * For non-ZernikeB1 meshes this falls through to `reconstruct_impl` with
 * `Parity::Uninitialized`. The `TagList` must be the full Variables tag list
 * so that `Spectral::compute_parity_list` can determine per-component parity.
 */
template <typename TagList, size_t Dim>
void reconstruct_impl_with_tag_list(
    gsl::span<double> dg_u, gsl::span<const double> subcell_u,
    const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
    const ReconstructionMethod reconstruction_method) {
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
        (num_even + num_odd) * (num_subcell_pts + num_dg_pts));
    DataVector even_subcell_input{&buffer[0], num_even * num_subcell_pts};
    DataVector odd_subcell_input{&buffer[num_even * num_subcell_pts],
                                 num_odd * num_subcell_pts};
    DataVector even_dg_output{&buffer[(num_even + num_odd) * num_subcell_pts],
                              num_even * num_dg_pts};
    DataVector odd_dg_output{
        &buffer[(num_even + num_odd) * num_subcell_pts + num_even * num_dg_pts],
        num_odd * num_dg_pts};

    // Sort subcell input into even/odd parity buffers
    const double* p_in = subcell_u.data();
    double* p_even_in = even_subcell_input.data();
    double* p_odd_in = odd_subcell_input.data();
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
        std::copy(p_in, p_in + seg_size * num_subcell_pts,  // NOLINT
                  p_even_in);
        p_even_in += seg_size * num_subcell_pts;  // NOLINT
      } else {
        std::copy(p_in, p_in + seg_size * num_subcell_pts,  // NOLINT
                  p_odd_in);
        p_odd_in += seg_size * num_subcell_pts;  // NOLINT
      }
      p_in += seg_size * num_subcell_pts;  // NOLINT
      is_even = not is_even;
    }

    // Reconstruct each parity batch with the appropriate reconstruction
    // matrix
    if constexpr (num_even > 0) {
      reconstruct_impl(
          gsl::span<double>{even_dg_output.data(), even_dg_output.size()},
          gsl::span<const double>{even_subcell_input.data(),
                                  even_subcell_input.size()},
          dg_mesh, subcell_extents, reconstruction_method,
          Spectral::Parity::Even);
    }
    if constexpr (num_odd > 0) {
      reconstruct_impl(
          gsl::span<double>{odd_dg_output.data(), odd_dg_output.size()},
          gsl::span<const double>{odd_subcell_input.data(),
                                  odd_subcell_input.size()},
          dg_mesh, subcell_extents, reconstruction_method,
          Spectral::Parity::Odd);
    }

    // Reassemble output in original component order
    double* p_out = dg_u.data();
    const double* p_even_out = even_dg_output.data();
    const double* p_odd_out = odd_dg_output.data();
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
        std::copy(p_even_out, p_even_out + seg_size * num_dg_pts, p_out);
        p_even_out += seg_size * num_dg_pts;  // NOLINT
      } else {
        // NOLINTNEXTLINE
        std::copy(p_odd_out, p_odd_out + seg_size * num_dg_pts, p_out);
        p_odd_out += seg_size * num_dg_pts;  // NOLINT
      }
      p_out += seg_size * num_dg_pts;  // NOLINT
      is_even = not is_even;
    }
    return;
  }
  reconstruct_impl(dg_u, subcell_u, dg_mesh, subcell_extents,
                   reconstruction_method, Spectral::Parity::Uninitialized);
}
}  // namespace detail

/// @{
/*!
 * \ingroup DgSubcellGroup
 * \brief reconstruct the variable `subcell_u_times_projected_det_jac` onto the
 * DG grid `dg_mesh`.
 *
 * In general we wish that the reconstruction operator is the pseudo-inverse of
 * the projection operator. On curved meshes this means we either need to
 * compute a (time-dependent) reconstruction and projection matrix on each DG
 * element, or we expand the determinant of the Jacobian on the basis, accepting
 * the aliasing errors from that. We accept the aliasing errors in favor of the
 * significantly reduced computational overhead. This means that the projection
 * and reconstruction operators are only inverses of each other if both operate
 * on \f$u J\f$ where \f$u\f$ is the variable being projected and \f$J\f$ is the
 * determinant of the Jacobian. That is, the matrices are guaranteed to satisfy
 * \f$\mathcal{R}(\mathcal{P}(u J))=u J\f$. If the mesh is regular Cartesian,
 * then this isn't an issue. Furthermore, if we reconstruct
 * \f$uJ/\mathcal{P}(J)\f$ we again recover the exact DG solution. Doing the
 * latter has the advantage that, in general, we are ideally projecting to the
 * subcells much more often than reconstructing from them (a statement that we
 * would rather use DG more than the subcells).
 *
 * When the DG mesh uses a ZernikeB1 basis the Variables overloads deduce
 * per-component parity from the tag list automatically. The raw `DataVector`
 * overloads accepting a `Spectral::Parity` are for single-component data where
 * the caller already knows the parity. Only `DimByDim` reconstruction is
 * supported for ZernikeB1 meshes.
 */
template <size_t Dim>
DataVector reconstruct(
    const DataVector& subcell_u_times_projected_det_jac,
    const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
    ReconstructionMethod reconstruction_method,
    Spectral::Parity parity = Spectral::Parity::Uninitialized);

template <size_t Dim>
void reconstruct(gsl::not_null<DataVector*> dg_u,
                 const DataVector& subcell_u_times_projected_det_jac,
                 const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
                 ReconstructionMethod reconstruction_method,
                 Spectral::Parity parity = Spectral::Parity::Uninitialized);

template <typename SubcellTagList, typename DgTagList, size_t Dim>
void reconstruct(const gsl::not_null<Variables<DgTagList>*> dg_u,
                 const Variables<SubcellTagList>& subcell_u,
                 const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
                 const ReconstructionMethod reconstruction_method) {
  ASSERT(subcell_u.number_of_grid_points() == subcell_extents.product(),
         "Incorrect subcell size of u: " << subcell_u.number_of_grid_points()
                                         << " but should be "
                                         << subcell_extents.product());
  if (UNLIKELY(dg_u->number_of_grid_points() !=
               dg_mesh.number_of_grid_points())) {
    dg_u->initialize(dg_mesh.number_of_grid_points(), 0.0);
  }
  detail::reconstruct_impl_with_tag_list<DgTagList>(
      gsl::span<double>{dg_u->data(), dg_u->size()},
      gsl::span<const double>{subcell_u.data(), subcell_u.size()}, dg_mesh,
      subcell_extents, reconstruction_method);
}

template <typename TagList, size_t Dim>
Variables<TagList> reconstruct(
    const Variables<TagList>& subcell_u, const Mesh<Dim>& dg_mesh,
    const Index<Dim>& subcell_extents,
    const ReconstructionMethod reconstruction_method) {
  Variables<TagList> dg_u(dg_mesh.number_of_grid_points());
  reconstruct(make_not_null(&dg_u), subcell_u, dg_mesh, subcell_extents,
              reconstruction_method);
  return dg_u;
}
/// @}
}  // namespace evolution::dg::subcell::fd
