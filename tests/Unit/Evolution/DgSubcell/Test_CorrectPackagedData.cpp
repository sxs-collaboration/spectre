// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim, bool SkipFirstVar>
void test() {
  CAPTURE(Dim);
  CAPTURE(SkipFirstVar);
  using Vars = Variables<tmpl::list<Var1, Var2<Dim>>>;
  using SubcellFaceVars =
      tmpl::conditional_t<SkipFirstVar, Variables<tmpl::list<Var2<Dim>>>, Vars>;
  const Mesh<Dim> volume_dg_mesh{5, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> volume_subcell_mesh =
      evolution::dg::subcell::fd::mesh(volume_dg_mesh);
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
        Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
  }
  const Element<Dim> element{ElementId<Dim>{0, {}}, neighbors};
  const TimeStepId time_step_id{true, 1, Time{Slab{1.1, 4.4}, {3, 10}}};

  CAPTURE(volume_dg_mesh);
  CAPTURE(volume_subcell_mesh);
  for (size_t direction_to_check = 0; direction_to_check < Dim;
       ++direction_to_check) {
    CAPTURE(direction_to_check);
    Index<Dim> volume_face_extents = volume_subcell_mesh.extents();
    ++volume_face_extents[direction_to_check];
    SubcellFaceVars lower_packaged_data{volume_face_extents.product()};
    SubcellFaceVars upper_packaged_data{volume_face_extents.product()};
    const auto set_volume_data = [&lower_packaged_data,
                                  &upper_packaged_data]() {
      std::iota(lower_packaged_data.data(),
                lower_packaged_data.data() + lower_packaged_data.size(), 1.0);
      std::iota(upper_packaged_data.data(),
                upper_packaged_data.data() + upper_packaged_data.size(),
                1.0 + upper_packaged_data.size());
    };
    set_volume_data();
    // Save a copy of the lower/upper data so we can compare points in the
    // volume/away from the interfaces.
    const SubcellFaceVars interior_lower_packaged_data = lower_packaged_data;
    const SubcellFaceVars interior_upper_packaged_data = upper_packaged_data;

    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::MortarData<Dim>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        mortar_data{};
    const Direction<Dim> upper{direction_to_check, Side::Upper};
    const Direction<Dim> lower{direction_to_check, Side::Lower};
    const std::pair upper_neighbor{upper,
                                   *element.neighbors().at(upper).begin()};
    const std::pair lower_neighbor{lower,
                                   *element.neighbors().at(lower).begin()};
    evolution::dg::MortarData<Dim>& upper_mortar_data =
        mortar_data[upper_neighbor] = {};
    evolution::dg::MortarData<Dim>& lower_mortar_data =
        mortar_data[lower_neighbor] = {};

    const Mesh<Dim - 1> dg_face_mesh =
        volume_dg_mesh.slice_away(direction_to_check);
    const size_t dg_number_of_independent_components =
        Vars::number_of_independent_components;
    const auto upper_neighbor_data = [&dg_face_mesh, &upper_packaged_data]() {
      DataVector result{dg_number_of_independent_components *
                        dg_face_mesh.number_of_grid_points()};
      std::iota(result.begin(), result.end(),
                1.0 + 2.0 * upper_packaged_data.size());
      return result;
    }();
    const auto lower_neighbor_data = [&dg_face_mesh, &upper_packaged_data,
                                      &upper_neighbor_data]() {
      DataVector result{dg_number_of_independent_components *
                        dg_face_mesh.number_of_grid_points()};
      std::iota(
          result.begin(), result.end(),
          1.0 + 2.0 * upper_packaged_data.size() + upper_neighbor_data.size());
      return result;
    }();

    // Insert neighbor DG data.
    upper_mortar_data.insert_neighbor_mortar_data(time_step_id, dg_face_mesh,
                                                  upper_neighbor_data);
    lower_mortar_data.insert_neighbor_mortar_data(time_step_id, dg_face_mesh,
                                                  lower_neighbor_data);

    const Mesh<Dim - 1> subcell_face_mesh =
        volume_subcell_mesh.slice_away(direction_to_check);
    const auto perform_face_check = [&dg_face_mesh, direction_to_check,
                                     &subcell_face_mesh, &volume_face_extents](
                                        const auto& face_dg_data,
                                        const auto& volume_fd_packaged_data,
                                        const size_t subcell_index) {
      SubcellFaceVars dg_face_vars_data{dg_face_mesh.number_of_grid_points()};
      std::copy(
          std::next(
              face_dg_data.begin(),
              static_cast<std::ptrdiff_t>(
                  SkipFirstVar ? dg_face_mesh.number_of_grid_points() : 0_st)),
          face_dg_data.end(), dg_face_vars_data.data());
      SubcellFaceVars subcell_face_vars_data{};
      if constexpr (Dim > 1) {
        subcell_face_vars_data = evolution::dg::subcell::fd::project(
            dg_face_vars_data, dg_face_mesh, subcell_face_mesh.extents());
      } else {
        (void)subcell_face_mesh;
        subcell_face_vars_data = dg_face_vars_data;
      }
      for (SliceIterator si(volume_face_extents, direction_to_check,
                            subcell_index);
           si; ++si) {
        CAPTURE(si.volume_offset());
        CAPTURE(si.slice_offset());
        tmpl::for_each<typename SubcellFaceVars::tags_list>(
            [&si, &subcell_face_vars_data,
             &volume_fd_packaged_data](auto tag_v) {
              using tag = tmpl::type_from<decltype(tag_v)>;
              CAPTURE(pretty_type::name<tag>());
              for (size_t storage_index = 0;
                   storage_index < get<tag>(volume_fd_packaged_data).size();
                   ++storage_index) {
                CAPTURE(storage_index);
                CHECK(get<tag>(volume_fd_packaged_data)[storage_index]
                                                       [si.volume_offset()] ==
                      approx(
                          get<tag>(subcell_face_vars_data)[storage_index]
                                                          [si.slice_offset()]));
              }
            });
      }
    };
    const auto perform_interior_check =
        [direction_to_check, &volume_face_extents](
            const auto& expected_volume_packaged_data,
            const SubcellFaceVars& volume_packaged_data,
            const size_t start_index, const size_t one_past_end_index) {
          for (size_t slice_index = start_index;
               slice_index < one_past_end_index; ++slice_index) {
            for (SliceIterator si(volume_face_extents, direction_to_check,
                                  slice_index);
                 si; ++si) {
              tmpl::for_each<typename SubcellFaceVars::tags_list>(
                  [&si, &expected_volume_packaged_data,
                   &volume_packaged_data](auto tag_v) {
                    using tag = tmpl::type_from<decltype(tag_v)>;
                    CAPTURE(pretty_type::name<tag>());
                    for (size_t storage_index = 0;
                         storage_index < get<tag>(volume_packaged_data).size();
                         ++storage_index) {
                      CAPTURE(storage_index);
                      CHECK(
                          get<tag>(volume_packaged_data)[storage_index]
                                                        [si.volume_offset()] ==
                          approx(get<tag>(expected_volume_packaged_data)
                                     [storage_index][si.volume_offset()]));
                    }
                  });
            }
          }
        };

    // Check with only remote data
    evolution::dg::subcell::correct_package_data<false>(
        make_not_null(&lower_packaged_data),
        make_not_null(&upper_packaged_data), direction_to_check, element,
        volume_subcell_mesh, mortar_data, SkipFirstVar ? 1_st : 0_st);

    perform_face_check(upper_neighbor_data, upper_packaged_data,
                       volume_face_extents[direction_to_check] - 1);
    perform_face_check(lower_neighbor_data, lower_packaged_data, 0);
    perform_interior_check(interior_upper_packaged_data, upper_packaged_data, 0,
                           volume_face_extents[direction_to_check] - 1);
    perform_interior_check(interior_lower_packaged_data, lower_packaged_data, 1,
                           volume_face_extents[direction_to_check]);

    // Reset volume data and then check that we can project both local and
    // remote data (note the `true` template parameter to
    // `correct_package_data`)
    set_volume_data();

    {
      DataVector upper_local_data{dg_number_of_independent_components *
                                  dg_face_mesh.number_of_grid_points()};
      std::iota(upper_local_data.begin(), upper_local_data.end(), 1.0e6);
      DataVector lower_local_data{dg_number_of_independent_components *
                                  dg_face_mesh.number_of_grid_points()};
      std::iota(lower_local_data.begin(), lower_local_data.end(), 1.0e7);
      upper_mortar_data.insert_local_mortar_data(time_step_id, dg_face_mesh,
                                                 upper_local_data);
      lower_mortar_data.insert_local_mortar_data(time_step_id, dg_face_mesh,
                                                 lower_local_data);

      evolution::dg::subcell::correct_package_data<true>(
          make_not_null(&lower_packaged_data),
          make_not_null(&upper_packaged_data), direction_to_check, element,
          volume_subcell_mesh, mortar_data, SkipFirstVar ? 1_st : 0_st);

      perform_face_check(upper_neighbor_data, upper_packaged_data,
                         volume_face_extents[direction_to_check] - 1);
      perform_face_check(lower_neighbor_data, lower_packaged_data, 0);
      perform_face_check(upper_local_data, lower_packaged_data,
                         volume_face_extents[direction_to_check] - 1);
      perform_face_check(lower_local_data, upper_packaged_data, 0);

      perform_interior_check(interior_upper_packaged_data, upper_packaged_data,
                             1, volume_face_extents[direction_to_check] - 1);
      perform_interior_check(interior_lower_packaged_data, lower_packaged_data,
                             1, volume_face_extents[direction_to_check] - 1);
    }

    // Check that if the local data is in the map but not requested to be
    // overwritten then we don't overwrite it.
    set_volume_data();

    evolution::dg::subcell::correct_package_data<false>(
        make_not_null(&lower_packaged_data),
        make_not_null(&upper_packaged_data), direction_to_check, element,
        volume_subcell_mesh, mortar_data, SkipFirstVar ? 1_st : 0_st);

    perform_face_check(upper_neighbor_data, upper_packaged_data,
                       volume_face_extents[direction_to_check] - 1);
    perform_face_check(lower_neighbor_data, lower_packaged_data, 0);

    perform_interior_check(interior_upper_packaged_data, upper_packaged_data, 0,
                           volume_face_extents[direction_to_check] - 1);
    perform_interior_check(interior_lower_packaged_data, lower_packaged_data, 1,
                           volume_face_extents[direction_to_check]);
  }  // for (size_t direction_to_check = 0; direction_to_check < Dim;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.CorrectPackagedData",
                  "[Evolution][Unit]") {
  test<1, false>();
  test<1, true>();
  test<2, false>();
  test<2, true>();
  test<3, false>();
  test<3, true>();
}
