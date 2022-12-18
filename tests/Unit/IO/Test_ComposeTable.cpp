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

namespace {
void test_table(const io::ComposeTable& compose_table) {
  CHECK(compose_table.temperature_bounds()[0] ==
        approx(0.10000000000000001));
  CHECK(compose_table.temperature_bounds()[1] == approx(158.0));
  CHECK(compose_table.number_density_bounds()[0] ==
        approx(1.0000000000000000e-12));
  CHECK(compose_table.number_density_bounds()[1] == approx(1.9));
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
            "specific internal energy", "sound speed squared", "free energy"});

  CHECK(compose_table.data("pressure") ==
        DataVector{3.2731420000000011e-12, 3.1893989140625013e-12,
                   3.1999981000000014e-12, 1.2155360915010773e-9,
                   8.3016422052875601e-10, 2.1627258654874539e-9,
                   3.2099012875311601e-5,  2.4468017331204212e-4,
                   6.1349624337516350e-4,  2210.6053143615190,
                   2354.6115554098865,     2538.9119061115266,
                   52.861468710971884,     52.861468705720966,
                   52.861468710893689,     52.861692578286011,
                   52.861692577931898,     52.861692577582062,
                   52.861449400505300,     52.861440607424719,
                   52.861463344038555,     5535.2247866186935,
                   6793.9338434046485,     6544.9524934507153});
  CHECK(compose_table.data("specific entropy") ==
        DataVector{
            158.12359000000009,    142.35139851562508,    143.93577000000008,
            9.3829711464065095,    0.99124464633440812,   2.4034713737830846,
            0.83152863391388587,   0.15564581028786784,   0.43268025604719001,
            2.6550771478559753e-3, 3.4795982667156005e-3, 3.3019006656817246e-3,
            1485445157219.0459,    1485445156918.3232,    1485445157214.1091,
            119933515.67157803,    119933515.67303930,    119933517.06181532,
            9683.2483710455308,    9683.2451082927018,    9683.2512877997542,
            5.0965517306845944,    6.2536132566462319,    6.5582247883215183});
  CHECK(compose_table.data("baryon chemical potential") ==
        DataVector{
            -1.6556880602462847,  -1.8745694607276671,  -15.559631577364950,
            -0.71347327189280818, -1.0456573497444996,  -16.605664215691274,
            0.43805416019382620,  -0.19600929781720619, -18.714060368760126,
            1895.9497406444118,   1909.6037477324869,   1873.0882690980980,
            -939.56534999255268,  -939.56534999359724,  -939.56534999464191,
            -939.56525776035880,  -939.56527069940967,  -939.56528363845996,
            -938.41518441013659,  -938.58050505591427,  -938.74176940754023,
            3214.4054109629301,   -4889.2134102447071,  -4340.8030471113798});
  CHECK(
      compose_table.data("charge chemical potential") ==
      DataVector{
          -19.842156349184702,    -19.190385749183669,    12.436817954442299,
          -22.670750207296898,    -21.646087184271057,    14.390424834977958,
          -26.755199814299104,    -25.385280320216147,    17.192821769637060,
          -198.22942467070547,    -28.325443917779911,    132.82602736094074,
          -3.4007065215488760e-9, 7.4961466524481554e-10, 4.8998992671017037e-9,
          -4.2118705904313029e-5, 9.2845115074833722e-6,  6.0687726877748120e-5,
          -0.53075097669568272,   0.11399145866263270,    0.75613580630043975,
          -137.87674894108289,    11.805107240332074,     131.01280476673062});
  CHECK(
      compose_table.data("lepton chemical potential") ==
      DataVector{
          -19.841845353053849,    -19.180888622625869,   12.455422287937649,
          -22.222766207600028,    -20.601886764725815,   15.658889679099508,
          -19.700822293861712,    -3.3834089107048677,   44.759109646262324,
          -35.332465886409203,    480.62811399608114,    770.54386620803041,
          -3.3915057215320162e-9, 1.0312528875660281e-9, 5.4538797241667128e-9,
          -4.2004349433156367e-5, 1.2772383912169461e-5, 6.7549115267934927e-5,
          -0.52933460049891867,   0.15719093234876486,   0.84111837567640779,
          -120.35548478633407,    366.21296279624414,    642.02667219177931});
  CHECK(compose_table.data("specific internal energy") ==
        DataVector{1.1372398000000006e-2,  3.5346163132812542e-3,
                   3.3070626000000045e-3,  -1.0169032790344480e-4,
                   -7.7663669049909921e-3, -7.6041533773714091e-3,
                   1.2234741042710821e-4,  -2.9897660752421736e-3,
                   4.4546614168404601e-3,  0.77921157405081587,
                   0.86947269803694438,    1.0634124481653968,
                   193541104778.35413,     193541104733.22195,
                   193541104777.67715,     15626334.678079225,
                   15626334.678358778,     15626334.912313508,
                   1260.6482389767325,     1260.6475515928282,
                   1260.6487573789184,     1.1740182149625666,
                   -7.8404943805046665,    -6.7725747751703373});
  CHECK(compose_table.data("sound speed squared") ==
        DataVector{
            4.2176792400813656e-3, 4.1456131080695067e-3, 4.1172877390651744e-3,
            1.7336626487566469e-4, 9.2710083433018563e-5, 2.6630342167846473e-4,
            3.4802175676683546e-4, 2.2675519460182546e-3, 5.6319393106579359e-3,
            1.2305841985543764,    1.2418223794193011,    1.2195142464169275,
            0.25395113490394422,   0.25395113613208825,   0.25395113559980687,
            0.25368373066295680,   0.25368373062477667,   0.25368372026896685,
            0.25365338177418945,   0.25365477067007891,   0.25365467228324629,
            -28.866914059936207,   66.325790590411600,    -194.71183243952410});
  CHECK(compose_table.data("free energy") ==
        DataVector{
            934.43810148373984,  928.65121304890044,  928.27897438227694,
            938.53150819022767,  932.16921623914311,  932.18040376414547,
            939.59715052645186,  936.74070481647925,  943.70752751624718,
            1671.6852726725444,  1756.4914144088082,  1938.7105006111608,
            -52861469826937.250, -52861469821686.242, -52861469826858.977,
            -4267988168.9958286, -4267988168.9672489, -4267988168.9389825,
            -344589.38828202756, -344589.44304620347, -344589.29579705157,
            1239.4909811346472,  -7413.7119665909195, -6460.7335245463910});

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
      Catch::Contains("eos.quantities' does not exist."));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities",
               " # number of regular, additional and derivative ", "");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("Read unexpected comment line: "));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities",
               " # indices of regular, additional and derivative quantities",
               "");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("Read unexpected comment line: "));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities", "1 2 3 4 5 7 12 1", "1 2 3 4 5 7000 12 1");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("Read in unknown quantity with number 7000"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities", "1 2 3 4 5 7 12 1", "1 2 3 4 5 7 12 10000");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("Read in unknown quantity with number 10000"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  replace_line("/eos.quantities", "1 2 3 4 5 7 12 1", "1 2 3 4 5 1 12 1");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("Found quantity 'pressure' more than once."));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("eos.parameters' does not exist."));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  replace_line("/eos.parameters",
               " # order of interpolation in first, second and third index",
               " # blah");
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("Read unexpected comment line: ' # blah'"));
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
      Catch::Contains("Read unexpected comment line: ' # herp'"));
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
      Catch::Contains("Read unexpected comment line: ' # derp'"));
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
      Catch::Contains("Read unexpected comment line: ' # blee'"));
  file_system::rm(directory, true);

  file_system::create_directory(directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.quantities", directory);
  file_system::copy(unit_test_src_path() + "/IO/eos.parameters", directory);
  CHECK_THROWS_WITH(
      ([&directory]() { const io::ComposeTable compose_table(directory); })(),
      Catch::Contains("eos.table' does not exist."));
  file_system::rm(directory, true);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.ComposeTable", "[Unit][IO]") {
  test();
  test_error_messages();
}
