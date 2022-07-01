// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Formaline.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test() {
  const DataVector expected_pressure{
      3.2731420000000011e-12, 3.1893989140625013e-12, 3.1999981000000014e-12,
      1.2155360915010773e-9,  8.3016422052875601e-10, 2.1627258654874539e-9,
      3.2099012875311601e-5,  2.4468017331204212e-4,  6.1349624337516350e-4,
      2210.6053143615190,     2354.6115554098865,     2538.9119061115266,
      52.861468710971884,     52.861468705720966,     52.861468710893689,
      52.861692578286011,     52.861692577931898,     52.861692577582062,
      52.861449400505300,     52.861440607424719,     52.861463344038555,
      5535.2247866186935,     6793.9338434046485,     6544.9524934507153};
  const DataVector expected_specific_internal_energy{
      1.1372398000000006e-2,  3.5346163132812542e-3,  3.3070626000000045e-3,
      -1.0169032790344480e-4, -7.7663669049909921e-3, -7.6041533773714091e-3,
      1.2234741042710821e-4,  -2.9897660752421736e-3, 4.4546614168404601e-3,
      0.77921157405081587,    0.86947269803694438,    1.0634124481653968,
      193541104778.35413,     193541104733.22195,     193541104777.67715,
      15626334.678079225,     15626334.678358778,     15626334.912313508,
      1260.6482389767325,     1260.6475515928282,     1260.6487573789184,
      1.1740182149625666,     -7.8404943805046665,    -6.7725747751703373};

  const std::vector<std::string> expected_independent_variable_names{
      "number density", "temperature", "electron fraction"};
  const std::vector<std::array<double, 2>> expected_independent_variable_bounds{
      {0.10000000000000001, 158.0}, {1.0e-12, 1.9}, {0.01, 0.6}};
  const std::vector<size_t> expected_independent_variable_number_of_points{2, 4,
                                                                           3};
  const std::vector<bool> expected_independent_variable_uses_log_spacing{
      true, true, false};
  const bool expected_beta_equilibrium = false;

  const std::string h5_file_name{"Unit.IO.H5.EosTable.h5"};
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  const auto check_table =
      [&expected_independent_variable_names,
       &expected_independent_variable_number_of_points,
       &expected_specific_internal_energy, &expected_pressure,
       &expected_beta_equilibrium,
       &expected_independent_variable_uses_log_spacing,
       &expected_independent_variable_bounds](const auto& eos_table) {
        CHECK(eos_table.available_quantities() ==
              std::vector<std::string>{"pressure", "specific internal energy"});
        CHECK(eos_table.number_of_independent_variables() ==
              expected_independent_variable_names.size());
        CHECK(eos_table.independent_variable_names() ==
              expected_independent_variable_names);
        CHECK(eos_table.independent_variable_bounds() ==
              expected_independent_variable_bounds);
        CHECK(eos_table.independent_variable_number_of_points() ==
              expected_independent_variable_number_of_points);
        CHECK(eos_table.independent_variable_uses_log_spacing() ==
              expected_independent_variable_uses_log_spacing);
        CHECK(eos_table.beta_equilibrium() == expected_beta_equilibrium);
        CHECK(eos_table.read_quantity("pressure") == expected_pressure);
        CHECK(eos_table.read_quantity("specific internal energy") ==
              expected_specific_internal_energy);
        CHECK(eos_table.subfile_path() == "/sfo");
      };

  {
    h5::H5File<h5::AccessType::ReadWrite> eos_file{h5_file_name};

    auto& eos_table_written = eos_file.insert<h5::EosTable>(
        "/sfo", expected_independent_variable_names,
        expected_independent_variable_bounds,
        expected_independent_variable_number_of_points,
        expected_independent_variable_uses_log_spacing,
        expected_beta_equilibrium, version_number);
    eos_table_written.write_quantity("pressure", expected_pressure);
    eos_table_written.write_quantity("specific internal energy",
                                     expected_specific_internal_energy);
    check_table(eos_table_written);
    eos_file.close_current_object();

    CHECK_THROWS_WITH(eos_file.get<h5::EosTable>(
                          "/sfo", expected_independent_variable_names,
                          expected_independent_variable_bounds,
                          expected_independent_variable_number_of_points,
                          expected_independent_variable_uses_log_spacing,
                          expected_beta_equilibrium, version_number),
                      Catch::Matchers::Contains(
                          "Opening an equation of state table with the "
                          "constructor for writing a table, but the subfile "));
  }
  {
    h5::H5File<h5::AccessType::ReadOnly> eos_file{h5_file_name};
    check_table(eos_file.get<h5::EosTable>("/sfo"));
    eos_file.close_current_object();
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.H5.EosTable", "[Unit][IO][H5]") { test(); }
