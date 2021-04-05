// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
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

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  using Vars = Variables<tmpl::list<Var1, Var2<Dim>>>;
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
    Vars lower_packaged_data{volume_face_extents.product()};
    Vars upper_packaged_data{volume_face_extents.product()};
    const size_t number_of_independent_components = Variables<
        tmpl::list<Var1, Var2<Dim>>>::number_of_independent_components;
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
    const Vars interior_lower_packaged_data = lower_packaged_data;
    const Vars interior_upper_packaged_data = upper_packaged_data;

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
    std::vector<double> upper_neighbor_data(
        number_of_independent_components *
        dg_face_mesh.number_of_grid_points());
    std::iota(upper_neighbor_data.begin(), upper_neighbor_data.end(),
              1.0 + 2.0 * upper_packaged_data.size());
    std::vector<double> lower_neighbor_data(
        number_of_independent_components *
        dg_face_mesh.number_of_grid_points());
    std::iota(
        lower_neighbor_data.begin(), lower_neighbor_data.end(),
        1.0 + 2.0 * upper_packaged_data.size() + upper_neighbor_data.size());

    upper_mortar_data.insert_neighbor_mortar_data(time_step_id, dg_face_mesh,
                                                  upper_neighbor_data);
    lower_mortar_data.insert_neighbor_mortar_data(time_step_id, dg_face_mesh,
                                                  lower_neighbor_data);

    // Check with only remote data
    evolution::dg::subcell::correct_package_data<false>(
        make_not_null(&lower_packaged_data),
        make_not_null(&upper_packaged_data), direction_to_check, element,
        volume_subcell_mesh, mortar_data);

    const size_t volume_grid_points = volume_face_extents.product();
    const size_t slice_grid_points =
        volume_face_extents.slice_away(direction_to_check).product();
    const Mesh<Dim - 1> subcell_face_mesh =
        volume_subcell_mesh.slice_away(direction_to_check);
    const auto perform_face_check = [&dg_face_mesh, direction_to_check,
                                     slice_grid_points, &subcell_face_mesh,
                                     &volume_face_extents, volume_grid_points](
                                        const auto& face_dg_data,
                                        const auto& volume_packaged_data,
                                        const size_t subcell_index) {
      Vars face_vars_data{dg_face_mesh.number_of_grid_points()};
      std::copy(face_dg_data.begin(), face_dg_data.end(),
                face_vars_data.data());
      if constexpr (Dim > 1) {
        face_vars_data = evolution::dg::subcell::fd::project(
            face_vars_data, dg_face_mesh, subcell_face_mesh.extents());
      } else {
        (void)subcell_face_mesh;
      }
      for (SliceIterator si(volume_face_extents, direction_to_check,
                            subcell_index);
           si; ++si) {
        CAPTURE(si.volume_offset());
        CAPTURE(si.slice_offset());
        for (size_t i = 0; i < number_of_independent_components; ++i) {
          CAPTURE(i);
          CHECK(volume_packaged_data
                    .data()[i * volume_grid_points + si.volume_offset()] ==
                approx(face_vars_data
                           .data()[i * slice_grid_points + si.slice_offset()]));
        }
      }
    };
    const auto perform_interior_check =
        [direction_to_check, &volume_face_extents, volume_grid_points](
            const auto& expected_volume_packaged_data,
            const auto& volume_packaged_data, const size_t start_index,
            const size_t one_past_end_index) {
          for (size_t slice_index = start_index;
               slice_index < one_past_end_index; ++slice_index) {
            for (SliceIterator si(volume_face_extents, direction_to_check,
                                  slice_index);
                 si; ++si) {
              for (size_t i = 0; i < number_of_independent_components; ++i) {
                CHECK(volume_packaged_data.data()[i * volume_grid_points +
                                                  si.volume_offset()] ==
                      approx(expected_volume_packaged_data
                                 .data()[i * volume_grid_points +
                                         si.volume_offset()]));
              }
            }
          }
        };

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

    std::vector<double> upper_local_data(number_of_independent_components *
                                         dg_face_mesh.number_of_grid_points());
    std::iota(upper_local_data.begin(), upper_local_data.end(), 1.0e6);
    std::vector<double> lower_local_data(number_of_independent_components *
                                         dg_face_mesh.number_of_grid_points());
    std::iota(lower_local_data.begin(), lower_local_data.end(), 1.0e7);
    upper_mortar_data.insert_local_mortar_data(time_step_id, dg_face_mesh,
                                               upper_local_data);
    lower_mortar_data.insert_local_mortar_data(time_step_id, dg_face_mesh,
                                               lower_local_data);

    evolution::dg::subcell::correct_package_data<true>(
        make_not_null(&lower_packaged_data),
        make_not_null(&upper_packaged_data), direction_to_check, element,
        volume_subcell_mesh, mortar_data);

    perform_face_check(upper_neighbor_data, upper_packaged_data,
                       volume_face_extents[direction_to_check] - 1);
    perform_face_check(lower_neighbor_data, lower_packaged_data, 0);
    perform_face_check(upper_local_data, lower_packaged_data,
                       volume_face_extents[direction_to_check] - 1);
    perform_face_check(lower_local_data, upper_packaged_data, 0);

    perform_interior_check(interior_upper_packaged_data, upper_packaged_data, 1,
                           volume_face_extents[direction_to_check] - 1);
    perform_interior_check(interior_lower_packaged_data, lower_packaged_data, 1,
                           volume_face_extents[direction_to_check] - 1);

    // Check that if the local data is in the map but not requested to be
    // overwritten then we don't overwrite it.
    set_volume_data();

    evolution::dg::subcell::correct_package_data<false>(
        make_not_null(&lower_packaged_data),
        make_not_null(&upper_packaged_data), direction_to_check, element,
        volume_subcell_mesh, mortar_data);

    perform_face_check(upper_neighbor_data, upper_packaged_data,
                       volume_face_extents[direction_to_check] - 1);
    perform_face_check(lower_neighbor_data, lower_packaged_data, 0);

    perform_interior_check(interior_upper_packaged_data, upper_packaged_data, 0,
                           volume_face_extents[direction_to_check] - 1);
    perform_interior_check(interior_lower_packaged_data, lower_packaged_data, 1,
                           volume_face_extents[direction_to_check]);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.CorrectPackagedData",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
