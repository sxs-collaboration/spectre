// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace detail {
template <typename TagsList>
struct NumberOfIndependentComponents;

template <typename... Tags>
struct NumberOfIndependentComponents<tmpl::list<Tags...>> {
  static constexpr size_t value = (... + Tags::type::size());
};
}  // namespace detail

/*!
 * \brief Projects a `Variables` of volume data to a contiguous subset of
 * a boundary `Variables`
 *
 * The `volume_fields` are all projected into the `face_fields` in the direction
 * `direction`. The tags in `VolumeVarsTagsList` must be a contiguous subset of
 * the tags in `FaceVarsTagsList`. That is, `FaceVarsTagsList` must be
 * equivalent to `tmpl::append<Before, VolumeVarsTagsList, After>` where
 * `Before` and `After` are `tmpl::list`s of arbitrary size. This is because the
 * projection is applied on all of the tensor components of the volume variables
 * and is written into contiguous memory on the boundary.
 *
 * In general, this function will be used for projecting all the evolved
 * variables or all the volume fluxes to the faces. The function
 * `evolution::dg::project_tensors_to_boundary()` should be used for projecting
 * individual tensors to the face.
 *
 * \note This function works for both Gauss and Gauss-Lobatto uniform meshes.
 */
template <typename VolumeVarsTagsList, typename FaceVarsTagsList, size_t Dim>
void project_contiguous_data_to_boundary(
    const gsl::not_null<Variables<FaceVarsTagsList>*> face_fields,
    const Variables<VolumeVarsTagsList>& volume_fields,
    const Mesh<Dim>& volume_mesh, const Direction<Dim>& direction) noexcept {
  static_assert(tmpl::size<VolumeVarsTagsList>::value != 0,
                "Must have non-zero number of volume fields");
  static_assert(tmpl::size<FaceVarsTagsList>::value >=
                    tmpl::size<VolumeVarsTagsList>::value,
                "There must not be more volume tags than there are face tags.");
  static_assert(
      tmpl::list_contains_v<FaceVarsTagsList, tmpl::front<VolumeVarsTagsList>>,
      "The first tag of VolumeVarsTagsList is not in the face tags. The "
      "VolumeVarsTagsList must be a subset of the FaceVarsTagsList");
  static_assert(
      tmpl::list_contains_v<FaceVarsTagsList, tmpl::back<VolumeVarsTagsList>>,
      "The last tag of VolumeVarsTagsList is not in the face tags. The "
      "VolumeVarsTagsList must be a subset of the FaceVarsTagsList");
  using face_vars_excluding_extras_at_end = tmpl::front<
      tmpl::split_at<FaceVarsTagsList,
                     tmpl::next<tmpl::index_of<
                         FaceVarsTagsList, tmpl::back<VolumeVarsTagsList>>>>>;
  using front_face_vars_split = tmpl::split_at<
      face_vars_excluding_extras_at_end,
      tmpl::index_of<FaceVarsTagsList, tmpl::front<VolumeVarsTagsList>>>;
  using volume_vars_face_subset_list = tmpl::back<front_face_vars_split>;
  static_assert(
      std::is_same_v<volume_vars_face_subset_list, VolumeVarsTagsList>,
      "The VolumeVarsTagsList must be a subset of the FaceVarsTagsList.");
  constexpr const size_t number_of_independent_components =
      Variables<VolumeVarsTagsList>::number_of_independent_components;
  using first_volume_tag = tmpl::front<VolumeVarsTagsList>;

  const Mesh<Dim> uniform_gauss_mesh(volume_mesh.extents(0),
                                     volume_mesh.basis(0),
                                     Spectral::Quadrature::Gauss);
  if (volume_mesh == uniform_gauss_mesh) {
    const Matrix identity{};
    auto interpolation_matrices = make_array<Dim>(std::cref(identity));
    const auto& matrix = Spectral::boundary_interpolation_matrices(
        volume_mesh.slice_through(direction.dimension()));
    gsl::at(interpolation_matrices, direction.dimension()) =
        direction.side() == Side::Upper ? matrix.second : matrix.first;

    auto& first_face_field = get<first_volume_tag>(*face_fields);
    auto& first_volume_field = get<first_volume_tag>(volume_fields);

    // The size is the number of tensor components we are projecting times the
    // number of grid points on the face. Note that this is _not_ equal to the
    // size of face_fields->size() since face_fields is a superset of the
    // volume variables.
    DataVector face_view{
        first_face_field[0].data(),
        first_face_field[0].size() * number_of_independent_components};

    apply_matrices(make_not_null(&face_view), interpolation_matrices,
                   // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                   DataVector{const_cast<double*>(first_volume_field[0].data()),
                              first_volume_field[0].size() *
                                  number_of_independent_components},
                   volume_mesh.extents());
  } else {
    ASSERT(Mesh<Dim>(volume_mesh.extents(0), volume_mesh.basis(0),
                     Spectral::Quadrature::GaussLobatto) == volume_mesh,
           "The current implementation assumes the mesh be either a uniform "
           "Gauss or Gauss-Lobatto mesh, but got "
               << volume_mesh << ".");
    const size_t sliced_dim = direction.dimension();
    const size_t fixed_index = direction.side() == Side::Upper
                                   ? volume_mesh.extents(sliced_dim) - 1
                                   : 0;

    const size_t interface_grid_points =
        volume_mesh.extents().slice_away(sliced_dim).product();
    const size_t volume_grid_points = volume_mesh.number_of_grid_points();

    const double* vars_data = volume_fields.data();
    // Since the face fields are a superset of the volume tags we need to find
    // the first volume tag on the face and get the pointer for that.
    double* interface_vars_data = get<first_volume_tag>(*face_fields)[0].data();

    // The reason we can't use data_on_slice is because the volume and face tags
    // are not the same, but data_on_slice assumes they are. In general, this
    // function should replace data_on_slice in the long term since in
    // additional to supporting different volume and face tags, it also supports
    // Gauss and Gauss-Lobatto points.
    //
    // Run the SliceIterator as the outer-most loop since incrementing the slice
    // iterator is surprisingly expensive.
    for (SliceIterator si(volume_mesh.extents(), sliced_dim, fixed_index); si;
         ++si) {
      for (size_t i = 0; i < number_of_independent_components; ++i) {
        // clang-tidy: do not use pointer arithmetic
        interface_vars_data[si.slice_offset() +                      // NOLINT
                            i * interface_grid_points] =             // NOLINT
            vars_data[si.volume_offset() + i * volume_grid_points];  // NOLINT
      }
    }
  }
}

/*!
 * \brief Projects a subset of the tensors in the `volume_fields` onto the face
 *
 * The tensors to project are listed in the `TagsToProjectList`.
 *
 * \note This function works for both Gauss and Gauss-Lobatto uniform meshes.
 */
template <typename TagsToProjectList, typename VolumeVarsTagsList,
          typename FaceVarsTagsList, size_t Dim>
void project_tensors_to_boundary(
    const gsl::not_null<Variables<FaceVarsTagsList>*> face_fields,
    const Variables<VolumeVarsTagsList>& volume_fields,
    const Mesh<Dim>& volume_mesh, const Direction<Dim>& direction) noexcept {
  static_assert(tmpl::size<VolumeVarsTagsList>::value != 0,
                "Must have non-zero number of volume fields");
  static_assert(tmpl::size<FaceVarsTagsList>::value >=
                    tmpl::size<VolumeVarsTagsList>::value,
                "There must not be more volume tags than there are face tags.");
  static_assert(
      tmpl::size<
          tmpl::list_difference<TagsToProjectList, FaceVarsTagsList>>::value ==
          0,
      "All of the tags in TagsToProjectList must be in FaceVarsTagsList");
  static_assert(
      tmpl::size<tmpl::list_difference<TagsToProjectList,
                                       VolumeVarsTagsList>>::value == 0,
      "All of the tags in TagsToProjectList must be in VolumeVarsTagsList");
  const Mesh<Dim> uniform_gauss_mesh(volume_mesh.extents(0),
                                     volume_mesh.basis(0),
                                     Spectral::Quadrature::Gauss);
  if (volume_mesh.quadrature() == uniform_gauss_mesh.quadrature()) {
    const Matrix identity{};
    auto interpolation_matrices = make_array<Dim>(std::cref(identity));
    const std::pair<Matrix, Matrix>& matrices =
        Spectral::boundary_interpolation_matrices(
            volume_mesh.slice_through(direction.dimension()));
    gsl::at(interpolation_matrices, direction.dimension()) =
        direction.side() == Side::Upper ? matrices.second : matrices.first;
    tmpl::for_each<TagsToProjectList>([&face_fields, &interpolation_matrices,
                                       &volume_fields,
                                       &volume_mesh](auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;
      auto& face_field = get<tag>(*face_fields);
      const auto& volume_field = get<tag>(volume_fields);
      DataVector face_view{face_field[0].data(),
                           face_field[0].size() * face_field.size()};
      apply_matrices(make_not_null(&face_view), interpolation_matrices,
                     // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                     DataVector{const_cast<double*>(volume_field[0].data()),
                                volume_field[0].size() * volume_field.size()},
                     volume_mesh.extents());
    });
  } else {
    ASSERT(Mesh<Dim>(volume_mesh.extents(0), volume_mesh.basis(0),
                     Spectral::Quadrature::GaussLobatto) == volume_mesh,
           "The current implementation assumes the mesh be either a uniform "
           "Gauss or Gauss-Lobatto mesh, but got "
               << volume_mesh);

    const size_t sliced_dim = direction.dimension();
    const size_t fixed_index = direction.side() == Side::Upper
                                   ? volume_mesh.extents(sliced_dim) - 1
                                   : 0;

    const size_t interface_grid_points =
        volume_mesh.extents().slice_away(sliced_dim).product();
    const size_t volume_grid_points = volume_mesh.number_of_grid_points();

    // Run the SliceIterator as the outer-most loop since incrementing the slice
    // iterator is surprisingly expensive.
    for (SliceIterator si(volume_mesh.extents(), sliced_dim, fixed_index); si;
         ++si) {
      tmpl::for_each<TagsToProjectList>([&face_fields, interface_grid_points,
                                         &si, &volume_fields,
                                         volume_grid_points](
                                            auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;

        const double* vars_data = get<tag>(volume_fields)[0].data();
        double* interface_vars_data = get<tag>(*face_fields)[0].data();
        static constexpr size_t number_of_independent_components_in_tensor =
            std::decay_t<decltype(get<tag>(volume_fields))>::size();

        for (size_t i = 0; i < number_of_independent_components_in_tensor;
             ++i) {
          // clang-tidy: do not use pointer arithmetic
          interface_vars_data[si.slice_offset() +                      // NOLINT
                              i * interface_grid_points] =             // NOLINT
              vars_data[si.volume_offset() + i * volume_grid_points];  // NOLINT
        }
      });
    }
  }
}
}  // namespace evolution::dg
