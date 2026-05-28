// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/ComposeTable.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void test_table(const io::ComposeTable& compose_table) {
  CHECK(compose_table.temperature_bounds()[0] ==
        approx(0.10000000000000001));
  CHECK(compose_table.temperature_bounds()[1] == approx(100.0));
  CHECK(compose_table.number_density_bounds()[0] ==
        approx(1.0000000000000000e-12));
  CHECK(compose_table.number_density_bounds()[1] == approx(1.0));
  CHECK(compose_table.electron_fraction_bounds()[0] == approx(0.01));
  CHECK(compose_table.electron_fraction_bounds()[1] == approx(0.6));
  CHECK(compose_table.number_density_number_of_points() == 4);
  CHECK(compose_table.temperature_number_of_points() == 2);
  CHECK(compose_table.electron_fraction_number_of_points() == 3);
  CHECK(compose_table.beta_equilibrium() == false);
  CHECK(compose_table.number_density_log_spacing() == true);
  CHECK(compose_table.temperature_log_spacing() == true);
  CHECK(compose_table.electron_fraction_log_spacing() == false);

  CHECK(compose_table.available_quantities() ==
        std::vector<std::string>{
            "pressure", "specific entropy", "baryon chemical potential",
            "charge chemical potential", "lepton chemical potential",
            "specific internal energy", "kappa", "sound speed squared",
            "free energy"});

  CHECK(compose_table.data("pressure") ==
        DataVector{3.2731399999999999e-012, 3.1893743437500001e-012,
                   3.1999140000000001e-012, 9.8280259999999892e-010,
                   6.6536454570312435e-010, 1.6219799999999979e-009,
                   1.8132359999999956e-005, 1.3813110312499967e-004,
                   3.4519099999999909e-004, 761.66969999999537,
                   754.13948124999547,      786.49059999999554,
                   11.920849999999991,      11.920849999999991,
                   11.920849999999991,      11.920849999999996,
                   11.920849999999996,      11.920849999999996,
                   11.925039999999964,      11.923575117187472,
                   11.923359999999979,      806.44919999999570,
                   809.93654999999592,      850.58989999999551});
  CHECK(compose_table.data("specific entropy") ==
        DataVector{
            158.12330000000000,    142.34600156249999,    143.91770000000000,
            9.6043219999999998,    1.4114397488281254,    2.4095170000000006,
            1.0688820000000012,    0.21825321992187507,   0.48016910000000024,
            1.8227950000000021e-3, 2.9249653632812526e-3, 3.4582940000000037e-3,
            493829299999.99982,    493829299999.99982,    493829299999.99982,
            49382930.000000007,    49382930.000000007,    49382930.000000015,
            4942.4760000000097,    4941.0258125000055,    4940.8130000000147,
            2.2287440000000029,    2.9170417968750031,    3.4335300000000046});
  CHECK(compose_table.data("baryon chemical potential") ==
        DataVector{
            -1.6556899335297999,  -1.8748185728422662,  -15.563139803025999,
            -0.73480252994256023, -1.0117381462964543,  -16.546742633324001,
            0.33851395092211950,  -0.20758187638742570, -18.461849392489999,
            1240.6791985227956,   1170.9186293072507,   1082.7297986941956,
            -939.56511813038003,  -939.56511813038003,  -939.56511813038003,
            -939.55478291097984,  -939.55788347679982,  -939.56098404261991,
            -849.16727603942002,  -872.21899159621978,  -898.84144104163954,
            1189.1064537167954,   1114.8139862189616,   1015.0566611937958});
  CHECK(
      compose_table.data("charge chemical potential") ==
      DataVector{
          -19.869797540044001,    -19.216893616987544,    12.440202930851999,
          -22.626830062841996,    -21.776062291462299,    14.309524668076000,
          -26.450654536453996,    -24.943883377251055,    17.073115357365996,
          -424.70150649913938,    -125.49445906295799,    62.547469999855920,
          -3.1025079016817068e-7, 2.9990496689180154e-7,  9.1005120791595476e-7,
          -1.0191343750298123e-2, -4.0899226662531657e-3, 2.0115371748642553e-3,
          -89.096600385884059,    -36.225786192595038,    17.849196377765999,
          -544.44319746711949,    -143.53021040829975,    71.169194584221955});
  CHECK(compose_table.data("lepton chemical potential") ==
        DataVector{-19.869619022618000,    -19.207807909464226,
                   12.458421103957999,     -22.205491354865995,
                   -20.800266405497513,    15.492381132752000,
                   -20.403123048046002,    -6.1058114290921406,
                   40.764717305411978,     -299.37137448849950,
                   208.99160948902875,     477.92760453951956,
                   -1.5417366231215918e-2, -1.5790172202279767e-3,
                   -1.0050221027217989e-3, -1.0224435243685657e-2,
                   -4.0895217801222954e-3, 2.0168513567666549e-3,
                   -89.095350763902104,    -36.187747079937012,
                   17.924013970567970,     -531.99451965635978,
                   92.987166740877740,     406.90134603959950});
  CHECK(compose_table.data("specific internal energy") ==
        DataVector{1.1372070000000000e-2,  3.5250635976562497e-3,
                   3.3034129999999998e-3,  -1.0079700000000000e-4,
                   -7.7495453945312506e-3, -7.6339050000000007e-3,
                   6.3908559999999740e-5,  -3.6514169687500032e-3,
                   2.7598749999999937e-3,  0.50663449999999799,
                   0.51142999609374795,    0.62049549999999798,
                   39871719999.999977,     39871719999.999977,
                   39871719999.999977,     3987171.0000000019,
                   3987171.0000000019,     3987171.0000000047,
                   398.21300000000116,     398.03899218750104,
                   398.01340000000152,     0.63881809999999817,
                   0.66514036484374817,    0.80032829999999799});
  CHECK(compose_table.data("kappa") ==
        DataVector{2.7890353489065712e-13, 2.7853376746820703e-13,
                   2.7733183475823816e-13, 6.5656686230048955e-9,
                   3.2497568301750098e-9,  2.7920445778786932e-9,
                   6.6649651409208873e-5,  1.0749245332014952e-5,
                   6.2044697222226870e-5,  -8.5383301117496231,
                   4.1061722738276361,     5.9839392440592016,
                   3.2071765752752263e-13, 3.2071765752752263e-13,
                   3.2071765752752263e-13, 3.2071765752752591e-9,
                   3.2071765752752591e-9,  3.2071656449480975e-9,
                   3.2129741711975296e-5,  3.2103317185309919e-5,
                   3.2099230427214770e-5,  0.33398579841295001,
                   0.37618823039543148,    0.37385352163718705});
  CHECK(compose_table.data("sound speed squared") ==
        DataVector{
            4.2178831483341683e-3, 4.1294695809557195e-3, 4.1280533859347213e-3,
            1.7355929131896138e-4, 1.0006022548304284e-4, 2.4363005364282908e-4,
            3.0645745055249087e-4, 1.9630783688857974e-3, 4.8613463610028725e-3,
            0.77844150571462700,   0.77756279313059307,   0.75109701269292695,
            0.32095563758493878,   0.32095563758493878,   0.32095563758493878,
            0.32076132569347848,   0.32076132569347848,   0.32076023274848281,
            0.32094559748152984,   0.32083359530461925,   0.32081207401951156,
            0.74441726301472821,   0.73846586011452453,   0.71126493464760321});
  CHECK(
      compose_table.data("free energy") ==
      DataVector{934.43787403909471,  928.64283077883169,  928.27740491047587,
                 938.51026241840748,  932.14305151369217,  932.15189450871753,
                 939.51855804214824,  936.11282961067434,  942.11046656855785,
                 1415.5814587332179,  1420.0870458811471,  1522.5612207860779,
                 -11920848759408.428, -11920848759408.428, -11920848759408.428,
                 -1192084876.0347986, -1192084876.0347998, -1192084876.0348027,
                 -119160.94559424007, -119179.46310701042, -119182.17977228011,
                 1316.9022872066578,  1272.8041173087074,  1348.1731847190777});

  for (const auto& quantity_name : compose_table.available_quantities()) {
    CHECK(compose_table.data(quantity_name) ==
          compose_table.data().at(quantity_name));
  }
}

void test() {
  const io::ComposeTable compose_table(unit_test_src_path() + "/IO");
  test_table(compose_table);
  test_table(serialize_and_deserialize(compose_table));
}

void test_error_messages() {
  const std::string directory = unit_test_build_path() + "/TestComposeTable";
  if (file_system::check_if_dir_exists(directory)) {
    file_system::rm(directory, true);
  }

  const auto replace_line = [&directory](const std::string& filename,
                                         const std::string& old,
                                         const std::string& new_str) {
    std::ifstream f;
    f.open(directory + filename, std::ios::ate);
    const auto size = f.tellg();
    std::string str(static_cast<size_t>(size), '\0');
    f.seekg(0);
    REQUIRE(f.read(&str[0], size));
    f.close();
    str.replace(str.find(old.c_str()), old.size(), new_str);
    file_system::rm(directory + filename, false);
    std::ofstream(directory + filename) << str;
  };

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring("eos.quantities' does not exist."));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities",
               " # number of regular, additional and derivative ", "");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring("Read unexpected comment line: "));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities",
               " # indices of regular, additional and derivative quantities",
               "");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring("Read unexpected comment line: "));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities", "1 2 3 4 5 7 11 12 1",
               "1 2 3 4 5 7000 11 12 1");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Read in unknown quantity with number 7000"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities", "1 2 3 4 5 7 11 12 1",
               "1 2 3 4 5 7 11 12 10000");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Read in unknown quantity with number 10000"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities", "1 2 3 4 5 7 11 12 1", "1 2 3 4 5 1 11 12 1");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Found quantity 'pressure' more than once."));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring("eos.parameters' does not exist."));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  replace_line("/eos.parameters",
               " # order of interpolation in first, second and third index",
               " # blah");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Read unexpected comment line: ' # blah'"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  replace_line("/eos.parameters",
               " # calculation of beta-equilibrium (1: yes, else: no) and for "
               "given entropy (1: yes, else: no)",
               " # herp");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Read unexpected comment line: ' # herp'"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  replace_line(
      "/eos.parameters",
      " # tabulation scheme (0 = explicit listing, 1 = loops, see manual)",
      " # derp");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Read unexpected comment line: ' # derp'"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  replace_line("/eos.parameters",
               " # parameter values (first, second and third index) depending "
               "on tabulation scheme",
               " # blee");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring(
          "Read unexpected comment line: ' # blee'"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Matchers::ContainsSubstring("eos.table' does not exist."));
  file_system::rm(directory, true);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.ComposeTable", "[Unit][IO]") {
  test();
  test_error_messages();
}
