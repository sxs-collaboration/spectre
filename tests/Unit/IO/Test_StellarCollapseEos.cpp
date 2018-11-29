// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp" // IWYU pragma: keep
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"  // IWYU pragma: keep
#include "IO/H5/StellarCollapseEos.hpp"
#include "Informer/InfoFromBuild.hpp"

// IWYU pragma: no_include <boost/iterator/iterator_facade.hpp>
// IWYU pragma: no_include <boost/multi_array.hpp>
// IWYU pragma: no_include <boost/multi_array/base.hpp>
// IWYU pragma: no_include <boost/multi_array/extent_gen.hpp>
// IWYU pragma: no_include <boost/multi_array/multi_array_ref.hpp>
// IWYU pragma: no_include <boost/multi_array/subarray.hpp>

namespace {

// For 1D arrays the numbers (in, for example, logrho_expected) are the first
// and second numbers in the corresponding dataset of the HDF5 file.
// For non-zero 3D arrays with no repeating numbers they are the first 2x2x2
// block (all elements with indices [0][0][0] -> [1][1][1]) in the HDF5 file.
// For 3D arrays with every element equal to zero (or with repeating numbers),
// they are random numbers in the range [-10.0, 10.0]
void test_tabulated(const std::string& file_path,
                    const std::string& subgroup_path) noexcept {
  h5::H5File<h5::AccessType::ReadOnly> sample_file(file_path);
  const auto& sample_data =
      sample_file.get<h5::StellarCollapseEos>(subgroup_path);

  // The group /sample_data has the additional dataset "test_data" which
  // is read to verify that we are not in "/"
  if (subgroup_path != "/") {
     CHECK(sample_data.get_scalar_dataset<double>("test_data") == 317);
  }

  CHECK(sample_data.get_scalar_dataset<double>("energy_shift") == 317);
  CHECK(sample_data.get_scalar_dataset<int>("have_rel_cs2") == 1);
  CHECK(sample_data.get_scalar_dataset<int>("pointsrho") == 391);
  CHECK(sample_data.get_scalar_dataset<int>("pointstemp") == 163);
  CHECK(sample_data.get_scalar_dataset<int>("pointsye") == 66);

  const std::vector<double> logrho_expected = {3.0239960056064277,
                                               3.0573293389397609};
  const auto logrho_from_file = sample_data.get_rank1_dataset("logrho");
  CHECK(logrho_from_file == logrho_expected);

  const std::vector<double> logtemp_expected = {-3.0, -2.9666666666666668};
  const auto logtemp_from_file = sample_data.get_rank1_dataset("logtemp");
  CHECK(logtemp_from_file == logtemp_expected);

  const std::vector<double> ye_expected = {0.005, 0.015};
  const auto ye_from_file = sample_data.get_rank1_dataset("ye");
  CHECK(ye_from_file == ye_expected);

  boost::multi_array<double, 3> Abar_expected(boost::extents[2][2][2]);
  Abar_expected[0][0][0] = 7.9795458306812677;
  Abar_expected[0][0][1] = -0.3599944438171061;
  Abar_expected[0][1][0] = 1.9845567482330431;
  Abar_expected[0][1][1] = -7.3785193097452417;
  Abar_expected[1][0][0] = -0.72424107655448644;
  Abar_expected[1][0][1] = -5.1501202601984524;
  Abar_expected[1][1][0] = 1.1758669985969643;
  Abar_expected[1][1][1] = 3.0731761169814753;
  CHECK(sample_data.get_rank3_dataset("Abar") == Abar_expected);

  boost::multi_array<double, 3> Albar_expected(boost::extents[2][2][2]);
  Albar_expected[0][0][0] = 7.5184888339990898;
  Albar_expected[0][0][1] = -0.94214490529767225;
  Albar_expected[0][1][0] = -5.4890512683931973;
  Albar_expected[0][1][1] = 7.1321598088866125;
  Albar_expected[1][0][0] = 7.4871335289327519;
  Albar_expected[1][0][1] = 4.872583973086309;
  Albar_expected[1][1][0] = -0.11058300302998703;
  Albar_expected[1][1][1] = -0.42315478232475989;
  CHECK(sample_data.get_rank3_dataset("Albar") == Albar_expected);

  boost::multi_array<double, 3> Xa_expected(boost::extents[2][2][2]);
  Xa_expected[0][0][0] = 0.89395326431166033;
  Xa_expected[0][0][1] = 5.2981489245483004;
  Xa_expected[0][1][0] = -8.2187962228781259;
  Xa_expected[0][1][1] = 9.1349240360021753;
  Xa_expected[1][0][0] = 6.8179106625235164;
  Xa_expected[1][0][1] = -9.0687734224053429;
  Xa_expected[1][1][0] = -0.026251730557362407;
  Xa_expected[1][1][1] = -6.9882245067320836;
  CHECK(sample_data.get_rank3_dataset("Xa") == Xa_expected);

  boost::multi_array<double, 3> Xh_expected(boost::extents[2][2][2]);
  Xh_expected[0][0][0] = 0.015416666479933495;
  Xh_expected[0][0][1] = 0.015416666761171319;
  Xh_expected[0][1][0] = 0.01541666668806504;
  Xh_expected[0][1][1] = 0.015416666496697482;
  Xh_expected[1][0][0] = 0.046249999646911247;
  Xh_expected[1][0][1] = 0.046249999756318716;
  Xh_expected[1][1][0] = 0.046250000104935601;
  Xh_expected[1][1][1] = 0.046250000039872653;
  CHECK(sample_data.get_rank3_dataset("Xh") == Xh_expected);

  boost::multi_array<double, 3> Xl_expected(boost::extents[2][2][2]);
  Xl_expected[0][0][0] = 7.3034402855828588;
  Xl_expected[0][0][1] = 9.0348910681793164;
  Xl_expected[0][1][0] = 7.535836352018535;
  Xl_expected[0][1][1] = 1.9984514947941445;
  Xl_expected[1][0][0] = 8.4566220303429489;
  Xl_expected[1][0][1] = 4.4478998287751121;
  Xl_expected[1][1][0] = 2.1875936108101861;
  Xl_expected[1][1][1] = 6.9377000841409249;
  CHECK(sample_data.get_rank3_dataset("Xl") == Xl_expected);

  boost::multi_array<double, 3> Xn_expected(boost::extents[2][2][2]);
  Xn_expected[0][0][0] = 0.98458333330164904;
  Xn_expected[0][0][1] = 0.98458333334956705;
  Xn_expected[0][1][0] = 0.98458333333767578;
  Xn_expected[0][1][1] = 0.98458333328657666;
  Xn_expected[1][0][0] = 0.95375000000630239;
  Xn_expected[1][0][1] = 0.95375000001765753;
  Xn_expected[1][1][0] = 0.95375000002310728;
  Xn_expected[1][1][1] = 0.95375000003230248;
  CHECK(sample_data.get_rank3_dataset("Xn") == Xn_expected);

  boost::multi_array<double, 3> Xp_expected(boost::extents[2][2][2]);
  Xp_expected[0][0][0] = -3.796313749453379;
  Xp_expected[0][0][1] = -6.1582001413425598;
  Xp_expected[0][1][0] = -6.8187240867986798;
  Xp_expected[0][1][1] = -2.3102245214369166;
  Xp_expected[1][0][0] = 3.1004045095839672;
  Xp_expected[1][0][1] = 8.603539081758413;
  Xp_expected[1][1][0] = 9.2598581731307092;
  Xp_expected[1][1][1] = -5.5441516922376906;
  CHECK(sample_data.get_rank3_dataset("Xp") == Xp_expected);

  boost::multi_array<double, 3> Zbar_expected(boost::extents[2][2][2]);
  Zbar_expected[0][0][0] = 4.559131217700795;
  Zbar_expected[0][0][1] = -4.2397342651521726;
  Zbar_expected[0][1][0] = 2.393481280087542;
  Zbar_expected[0][1][1] = -5.1014197051952941;
  Zbar_expected[1][0][0] = -1.4240946022305412;
  Zbar_expected[1][0][1] = 7.27848817794737;
  Zbar_expected[1][1][0] = 9.6764051264858288;
  Zbar_expected[1][1][1] = -6.0292384378110402;
  CHECK(sample_data.get_rank3_dataset("Zbar") == Zbar_expected);

  boost::multi_array<double, 3> Zlbar_expected(boost::extents[2][2][2]);
  Zlbar_expected[0][0][0] = 0.16485165186190187;
  Zlbar_expected[0][0][1] = 6.7850358862748159;
  Zlbar_expected[0][1][0] = 0.84432527731156881;
  Zlbar_expected[0][1][1] = 8.5313505248650365;
  Zlbar_expected[1][0][0] = 2.5371093297378415;
  Zlbar_expected[1][0][1] = -8.0091698604208634;
  Zlbar_expected[1][1][0] = 0.081681565040991444;
  Zlbar_expected[1][1][1] = 6.6878158037490429;
  CHECK(sample_data.get_rank3_dataset("Zlbar") == Zlbar_expected);

  boost::multi_array<double, 3> cs2_expected(boost::extents[2][2][2]);
  cs2_expected[0][0][0] = 1577678648549693.8;
  cs2_expected[0][0][1] = 1577667221164363.8;
  cs2_expected[0][1][0] = 1703576033357332.5;
  cs2_expected[0][1][1] = 1703564429793273.2;
  cs2_expected[1][0][0] = 1543882126488769.0;
  cs2_expected[1][0][1] = 1543838596546058.0;
  cs2_expected[1][1][0] = 1667202534427290.2;
  cs2_expected[1][1][1] = 1667158615466813.5;
  CHECK(sample_data.get_rank3_dataset("cs2") == cs2_expected);

  boost::multi_array<double, 3> dedt_expected(boost::extents[2][2][2]);
  dedt_expected[0][0][0] = 1.4207387553108902e+18;
  dedt_expected[0][0][1] = 1.4206993403324593e+18;
  dedt_expected[0][1][0] = 1.4208772178581018e+18;
  dedt_expected[0][1][1] = 1.4208282620381565e+18;
  dedt_expected[1][0][0] = 1.3914302855474371e+18;
  dedt_expected[1][0][1] = 1.3913842153750838e+18;
  dedt_expected[1][1][0] = 1.3915839621991084e+18;
  dedt_expected[1][1][1] = 1.3915283697927603e+18;
  CHECK(sample_data.get_rank3_dataset("dedt") == dedt_expected);

  boost::multi_array<double, 3> dpderho_expected(boost::extents[2][2][2]);
  dpderho_expected[0][0][0] = 704.39241942245394;
  dpderho_expected[0][0][1] = 760.59561843311644;
  dpderho_expected[0][1][0] = 704.35793122929726;
  dpderho_expected[0][1][1] = 760.56090764534656;
  dpderho_expected[1][0][0] = 704.35394804365501;
  dpderho_expected[1][0][1] = 760.55427457755161;
  dpderho_expected[1][1][0] = 704.3156360818092;
  dpderho_expected[1][1][1] = 760.5158477148301;
  CHECK(sample_data.get_rank3_dataset("dpderho") == dpderho_expected);

  boost::multi_array<double, 3> dpdrhoe_expected(boost::extents[2][2][2]);
  dpdrhoe_expected[0][0][0] = 946635080552566.0;
  dpdrhoe_expected[0][0][1] = 946623484746662.75;
  dpdrhoe_expected[0][1][0] = 1022196607501370.9;
  dpdrhoe_expected[0][1][1] = 1022182353756068.2;
  dpdrhoe_expected[1][0][0] = 926314324149490.25;
  dpdrhoe_expected[1][0][1] = 926288146766470.0;
  dpdrhoe_expected[1][1][0] = 1000313905133207.1;
  dpdrhoe_expected[1][1][1] = 1000284548311442.8;
  CHECK(sample_data.get_rank3_dataset("dpdrhoe") == dpdrhoe_expected);

  boost::multi_array<double, 3> entropy_expected(boost::extents[2][2][2]);
  entropy_expected[0][0][0] = 12.439959418394057;
  entropy_expected[0][0][1] = 12.363977600038989;
  entropy_expected[0][1][0] = 12.553961989417337;
  entropy_expected[0][1][1] = 12.477976613417658;
  entropy_expected[1][0][0] = 12.142114662263879;
  entropy_expected[1][0][1] = 12.067704524057188;
  entropy_expected[1][1][0] = 12.25376620450713;
  entropy_expected[1][1][1] = 12.179352008001388;
  CHECK(sample_data.get_rank3_dataset("entropy") == entropy_expected);

  boost::multi_array<double, 3> gamma_expected(boost::extents[2][2][2]);
  gamma_expected[0][0][0] = 1.6667180692156467;
  gamma_expected[0][0][1] = 1.6667273271084526;
  gamma_expected[0][1][0] = 1.66668659680834;
  gamma_expected[0][1][1] = 1.6666969738652102;
  gamma_expected[1][0][0] = 1.6671718466996817;
  gamma_expected[1][0][1] = 1.6671938524936938;
  gamma_expected[1][1][0] = 1.6671006949123433;
  gamma_expected[1][1][1] = 1.6671232575231221;
  CHECK(sample_data.get_rank3_dataset("gamma") == gamma_expected);

  boost::multi_array<double, 3> logenergy_expected(boost::extents[2][2][2]);
  logenergy_expected[0][0][0] = 19.279083431017359;
  logenergy_expected[0][0][1] = 19.279083430081528;
  logenergy_expected[0][1][0] = 19.279086019802637;
  logenergy_expected[0][1][1] = 19.279086018868753;
  logenergy_expected[1][0][0] = 19.273645709972406;
  logenergy_expected[1][0][1] = 19.273645706932353;
  logenergy_expected[1][1][0] = 19.273648277270205;
  logenergy_expected[1][1][1] = 19.273648274167705;
  CHECK(sample_data.get_rank3_dataset("logenergy") == logenergy_expected);

  boost::multi_array<double, 3> logpress_expected(boost::extents[2][2][2]);
  logpress_expected[0][0][0] = 18.000096394732644;
  logpress_expected[0][0][1] = 18.033424170052783;
  logpress_expected[0][1][0] = 18.033447660355776;
  logpress_expected[0][1][1] = 18.066775331565321;
  logpress_expected[1][0][0] = 17.990459396691801;
  logpress_expected[1][0][1] = 18.0237747523636;
  logpress_expected[1][1][0] = 18.023852243798054;
  logpress_expected[1][1][1] = 18.057168258597677;
  CHECK(sample_data.get_rank3_dataset("logpress") == logpress_expected);

  boost::multi_array<double, 3> meffn_expected(boost::extents[2][2][2]);
  meffn_expected[0][0][0] = 2.4744534162653053;
  meffn_expected[0][0][1] = -8.997614958121309;
  meffn_expected[0][1][0] = 7.4562655881310071;
  meffn_expected[0][1][1] = -7.8985993911032804;
  meffn_expected[1][0][0] = 8.1298728332416808;
  meffn_expected[1][0][1] = -1.5340273998086609;
  meffn_expected[1][1][0] = -3.6860947464281306;
  meffn_expected[1][1][1] = 8.1586649741865287;
  CHECK(sample_data.get_rank3_dataset("meffn") == meffn_expected);

  boost::multi_array<double, 3> meffp_expected(boost::extents[2][2][2]);
  meffp_expected[0][0][0] = 0.78824408539654733;
  meffp_expected[0][0][1] = -9.5483672316917243;
  meffp_expected[0][1][0] = 0.54136393120109716;
  meffp_expected[0][1][1] = 2.6773060080962736;
  meffp_expected[1][0][0] = -2.8801827291174398;
  meffp_expected[1][0][1] = 5.2981584719540482;
  meffp_expected[1][1][0] = 8.5371935011578408;
  meffp_expected[1][1][1] = -4.1656335726133253;
  CHECK(sample_data.get_rank3_dataset("meffp") == meffp_expected);

  boost::multi_array<double, 3> mu_e_expected(boost::extents[2][2][2]);
  mu_e_expected[0][0][0] = 0.50689824836686137;
  mu_e_expected[0][0][1] = 0.50697546449327213;
  mu_e_expected[0][1][0] = 0.50644579668153178;
  mu_e_expected[0][1][1] = 0.50652911816373358;
  mu_e_expected[1][0][0] = 0.50800847018382267;
  mu_e_expected[1][0][1] = 0.50808661150047807;
  mu_e_expected[1][1][0] = 0.50764321962573911;
  mu_e_expected[1][1][1] = 0.5077274312638953;
  CHECK(sample_data.get_rank3_dataset("mu_e") == mu_e_expected);

  boost::multi_array<double, 3> mu_n_expected(boost::extents[2][2][2]);
  mu_n_expected[0][0][0] = -0.010095301103328893;
  mu_n_expected[0][0][1] = -0.010018547102765752;
  mu_n_expected[0][1][0] = -0.011024970810644845;
  mu_n_expected[0][1][1] = -0.010942093884027599;
  mu_n_expected[1][0][0] = -0.010127118677803536;
  mu_n_expected[1][0][1] = -0.010050364713731597;
  mu_n_expected[1][1][0] = -0.011059326583649635;
  mu_n_expected[1][1][1] = -0.010976449692047936;
  CHECK(sample_data.get_rank3_dataset("mu_n") == mu_n_expected);

  boost::multi_array<double, 3> mu_p_expected(boost::extents[2][2][2]);
  mu_p_expected[0][0][0] = -24.710273945544664;
  mu_p_expected[0][0][1] = -24.710437248365768;
  mu_p_expected[0][1][0] = -24.708423126879229;
  mu_p_expected[0][1][1] = -24.708598930672853;
  mu_p_expected[1][0][0] = -24.710274705168015;
  mu_p_expected[1][0][1] = -24.710440937269624;
  mu_p_expected[1][1][0] = -24.708414929439002;
  mu_p_expected[1][1][1] = -24.708593662516304;
  CHECK(sample_data.get_rank3_dataset("mu_p") == mu_p_expected);

  boost::multi_array<double, 3> muhat_expected(boost::extents[2][2][2]);
  muhat_expected[0][0][0] = 24.700178644441333;
  muhat_expected[0][0][1] = 24.700418701263001;
  muhat_expected[0][1][0] = 24.697398156068584;
  muhat_expected[0][1][1] = 24.697656836788827;
  muhat_expected[1][0][0] = 24.70014758649021;
  muhat_expected[1][0][1] = 24.700390572555893;
  muhat_expected[1][1][0] = 24.697355602855353;
  muhat_expected[1][1][1] = 24.697617212824255;
  CHECK(sample_data.get_rank3_dataset("muhat") == muhat_expected);

  boost::multi_array<double, 3> munu_expected(boost::extents[2][2][2]);
  munu_expected[0][0][0] = -24.193280396074471;
  munu_expected[0][0][1] = -24.193443236769728;
  munu_expected[0][1][0] = -24.190952359387051;
  munu_expected[0][1][1] = -24.191127718625093;
  munu_expected[1][0][0] = -24.192139116306389;
  munu_expected[1][0][1] = -24.192303961055416;
  munu_expected[1][1][0] = -24.189712383229615;
  munu_expected[1][1][1] = -24.18988978156036;
  CHECK(sample_data.get_rank3_dataset("munu") == munu_expected);

  boost::multi_array<double, 3> r_expected(boost::extents[2][2][2]);
  r_expected[0][0][0] = -0.60525665755039704;
  r_expected[0][0][1] = 2.2179733206970909;
  r_expected[0][1][0] = -1.0549307927354015;
  r_expected[0][1][1] = -6.0983303982526049;
  r_expected[1][0][0] = -9.6472871040770141;
  r_expected[1][0][1] = -9.5451610615235349;
  r_expected[1][1][0] = -1.2051288530836537;
  r_expected[1][1][1] = 6.0435070768646284;
  CHECK(sample_data.get_rank3_dataset("r") == r_expected);

  boost::multi_array<double, 3> u_expected(boost::extents[2][2][2]);
  u_expected[0][0][0] = 9.8002955742002342;
  u_expected[0][0][1] = 2.7463512929327969;
  u_expected[0][1][0] = -2.2719797545844012;
  u_expected[0][1][1] = 0.12238635453091185;
  u_expected[1][0][0] = 1.2619706692408634;
  u_expected[1][0][1] = 4.8905663798386279;
  u_expected[1][1][0] = 7.6749788245333903;
  u_expected[1][1][1] = 7.7886279535935081;
  CHECK(sample_data.get_rank3_dataset("u") == u_expected);
}

SPECTRE_TEST_CASE("Unit.IO.H5.StellarCollapseEos", "[Unit][IO][H5]") {
  test_tabulated(unit_test_path() + "/IO/StellarCollapse2017Sample.h5", "/");
  test_tabulated(unit_test_path() + "/IO/StellarCollapse2017Sample.h5",
                 "/sample_data");
}
}  // namespace
