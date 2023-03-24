// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

namespace evolution::dg::subcell {
namespace {
void test(const size_t number_of_buffers) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  MAKE_GENERATOR(gen);

  constexpr size_t number_of_grid_points = 8;

  CAPTURE(number_of_buffers);
  CAPTURE(number_of_grid_points);

  GhostData ghost_data{number_of_buffers};
  std::vector<DataVector> all_local_data{number_of_buffers,
                                         DataVector{number_of_grid_points}};
  std::vector<DataVector> all_neighbor_data{number_of_buffers,
                                            DataVector{number_of_grid_points}};

  CHECK(ghost_data.total_number_of_buffers() == number_of_buffers);

  for (size_t i = 0; i < number_of_buffers; i++) {
    DataVector& local_data = all_local_data[i];
    DataVector& neighbor_data = all_neighbor_data[i];
    fill_with_random_values(make_not_null(&local_data), make_not_null(&gen),
                            make_not_null(&dist));
    fill_with_random_values(make_not_null(&neighbor_data), make_not_null(&gen),
                            make_not_null(&dist));

    std::string expected_output =
        MakeString{} << "LocalGhostData: " << local_data << "\n"
                     << "NeighborGhostDataForReconstruction: " << neighbor_data
                     << "\n";

    CHECK(ghost_data.local_ghost_data().size() == 0);
    CHECK(ghost_data.neighbor_ghost_data_for_reconstruction().size() == 0);

    ghost_data.local_ghost_data() = local_data;
    ghost_data.neighbor_ghost_data_for_reconstruction() = neighbor_data;

    CHECK(ghost_data.local_ghost_data() == local_data);
    CHECK(ghost_data.neighbor_ghost_data_for_reconstruction() == neighbor_data);

    CHECK(get_output(ghost_data) == expected_output);

    CHECK(ghost_data.current_buffer_index() == i);
    ghost_data.next_buffer();
    // If we only have one buffer then the index will stay the same. Otherwise
    // it'll change
    if (number_of_buffers == 1) {
      CHECK(ghost_data.current_buffer_index() == i);
    } else {
      CHECK_FALSE(ghost_data.current_buffer_index() == i);
    }
  }

  // Make sure we're back at the beginning
  CHECK(ghost_data.current_buffer_index() == 0);

  for (size_t i = 0; i < number_of_buffers; i++) {
    DataVector previous_local_data = ghost_data.local_ghost_data();
    DataVector previous_neighbor_data =
        ghost_data.neighbor_ghost_data_for_reconstruction();

    ghost_data.next_buffer();

    // If we only have one buffer, make sure the data is the same. Otherwise
    // check that the data is different
    if (number_of_buffers == 1) {
      CHECK(ghost_data.local_ghost_data() == previous_local_data);
      CHECK(ghost_data.neighbor_ghost_data_for_reconstruction() ==
            previous_neighbor_data);
    } else {
      CHECK_FALSE(ghost_data.local_ghost_data() == previous_local_data);
      CHECK_FALSE(ghost_data.neighbor_ghost_data_for_reconstruction() ==
                  previous_neighbor_data);
    }
  }

  test_serialization(ghost_data);
}

void test_errors() {
  CHECK_THROWS_WITH(
      GhostData{0},
      Catch::Contains(
          "The GhostData class must be constructed with at least one buffer."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.GhostData", "[Unit][Evolution]") {
  for (size_t i = 1; i < 5; i++) {
    test(i);
  }
  test_errors();
}
}  // namespace evolution::dg::subcell
