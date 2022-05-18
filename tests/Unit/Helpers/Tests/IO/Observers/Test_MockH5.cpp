// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Helpers/IO/Observers/MockH5.hpp"

namespace TestHelpers::observers {
namespace {
void test_mock_dat() {
  MockDat my_dat{};

  const std::vector<std::string> legend{"This", "is", "SPARTA"};
  const std::vector<double> data{{0.0, 1.3, 6.7}};

  my_dat.append(legend, data);

  const auto& dat_legend = my_dat.get_legend();
  CHECK(dat_legend == legend);

  const auto& matrix_data = my_dat.get_data();
  CHECK(matrix_data == Matrix{{0.0, 1.3, 6.7}});

  CHECK_THROWS_WITH(
      ([&my_dat, &legend]() {
        const std::vector<double> bad_data(1, 0.0);
        my_dat.append(legend, bad_data);
      })(),
      Catch::Contains("Size of supplied data does not match number of columns "
                      "in the existing matrix."));
  CHECK_THROWS_WITH(
      ([&my_dat, &data]() {
        const std::vector<std::string> bad_legend{"Bad"};
        my_dat.append(bad_legend, data);
      })(),
      Catch::Contains(
          "Supplied legend is not the same as the existing legend."));

  CHECK_THROWS_WITH(
      ([]() {
        MockDat dat{};
        auto& error_data = dat.get_data();
        (void)error_data;
      })(),
      Catch::Contains("Cannot get data. Append some data first."));
  CHECK_THROWS_WITH(
      ([]() {
        MockDat dat{};
        auto& error_legend = dat.get_legend();
        (void)error_legend;
      })(),
      Catch::Contains("Cannot get legend. Append some data first."));
}

void test_mock_h5() {
  MockH5File file{};

  const std::string path{"/my/path"};
  file.try_insert(path);

  auto& gotten_dat = file.get_dat(path);
  const std::vector<std::string> legend{"Time"};
  const std::vector<double> data{0.0};
  gotten_dat.append(legend, data);

  const auto& const_dat_ref = file.get_dat(path);
  CHECK(const_dat_ref.get_legend() == legend);
  CHECK(const_dat_ref.get_data() == Matrix{{0.0}});

  CHECK_THROWS_WITH(
      ([&file]() {
        auto& nonexistent_dat = file.get_dat("/bad/path");
        (void)nonexistent_dat;
      })(),
      Catch::Contains(
          "Cannot get /bad/path from MockH5File. Path does not exist."));
  CHECK_THROWS_WITH(
      ([&file]() {
        const auto& nonexistent_dat = file.get_dat("/bad/path");
        (void)nonexistent_dat;
      })(),
      Catch::Contains(
          "Cannot get /bad/path from MockH5File. Path does not exist."));
}

SPECTRE_TEST_CASE("Test.TestHelpers.IO.Observers.MockH5", "[IO][Unit]") {
  test_mock_dat();
  test_mock_h5();
}
}  // namespace
}  // namespace TestHelpers::observers
