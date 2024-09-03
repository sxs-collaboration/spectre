// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackHelper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace ylm {
namespace {

// Since we are sharing this code with users and it is intentionally design to
// be very C-style, we just exclude the entire block from clang-tidy.
// NOLINTBEGIN

// [spectre_cce_grid_point_locations]
struct SpectreCceGridPointLocations {
  int number_of_theta_points;
  int number_of_phi_points;
  // We use C-style arrays to avoid dealing with malloc/free and new/delete
  // interoperability between different libraries.
  double theta_points[33];
  double phi_points[2 * 33 + 1];
};

SpectreCceGridPointLocations spectre_ylm_theta_phi_points(const int l_max) {
  SpectreCceGridPointLocations result{};
  if (l_max < 4 or l_max > 32) {
    result.number_of_theta_points = 0;
    result.number_of_phi_points = 0;
    return result;
  }
  result.number_of_theta_points = l_max + 1;
  result.number_of_phi_points = 2 * l_max + 1;

  for (int i = 0; i < result.number_of_theta_points; ++i) {
    switch (result.number_of_theta_points) {
      case 5: {
        const double temp[5] = {
            4.366349492255221509e-01, 1.002176803643121561e+00,
            1.570796326794896558e+00, 2.139415849946671777e+00,
            2.704957704364271187e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 6: {
        const double temp[6] = {
            3.696066519448289456e-01, 8.483666264874876184e-01,
            1.329852612388110256e+00, 1.811740041201682860e+00,
            2.293226027102305498e+00, 2.771986001644964226e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 7: {
        const double temp[7] = {
            3.204050902900620335e-01, 7.354466143229519970e-01,
            1.152892953722227221e+00, 1.570796326794896558e+00,
            1.988699699867565895e+00, 2.406146039266841008e+00,
            2.821187563299730972e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 8: {
        const double temp[8] = {
            2.827570635937968202e-01, 6.490365804607796107e-01,
            1.017455539490153438e+00, 1.386317078892131294e+00,
            1.755275574697661822e+00, 2.124137114099639678e+00,
            2.492556073129013505e+00, 2.858835589995996074e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 9: {
        const double temp[9] = {
            2.530224166119307005e-01, 5.807869795060065510e-01,
            9.104740292261472856e-01, 1.240573923404363343e+00,
            1.570796326794896558e+00, 1.901018730185429773e+00,
            2.231118624363645608e+00, 2.560805674083786343e+00,
            2.888570236977862304e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 10: {
        const double temp[10] = {
            2.289442988470259954e-01, 5.255196555285001070e-01,
            8.238386589997556131e-01, 1.122539327631709494e+00,
            1.421366498439524895e+00, 1.720226155150268221e+00,
            2.019053325958083622e+00, 2.317753994590037614e+00,
            2.616072998061293120e+00, 2.912648354742767065e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 11: {
        const double temp[11] = {
            2.090492874137409585e-01, 4.798534223256742948e-01,
            7.522519395990820978e-01, 1.025003226369574749e+00,
            1.297877729331450292e+00, 1.570796326794896558e+00,
            1.843714924258342824e+00, 2.116589427220218589e+00,
            2.389340713990710796e+00, 2.661739231264118821e+00,
            2.932543366176052047e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 12: {
        const double temp[12] = {
            1.923346793046672165e-01, 4.414870814893317452e-01,
            6.921076988818409825e-01, 9.430552870605736215e-01,
            1.194120375947706592e+00, 1.445233238471440140e+00,
            1.696359415118352976e+00, 1.947472277642086524e+00,
            2.198537366529219383e+00, 2.449484954707952244e+00,
            2.700105572100461426e+00, 2.949257974285125705e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 13: {
        const double temp[13] = {
            1.780944581262765558e-01, 4.088002373420211999e-01,
            6.408663264733868159e-01, 8.732366099401630555e-01,
            1.105718066248490006e+00, 1.338247676100454475e+00,
            1.570796326794896558e+00, 1.803344977489338641e+00,
            2.035874587341303332e+00, 2.268356043649629949e+00,
            2.500726327116406189e+00, 2.732792416247772138e+00,
            2.963498195463516449e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 14: {
        const double temp[14] = {
            1.658171411523663707e-01, 3.806189306666775130e-01,
            5.966877608172733716e-01, 8.130407055389454740e-01,
            1.029498592525136758e+00, 1.246003586776677663e+00,
            1.462529992921481892e+00, 1.679062660668311224e+00,
            1.895589066813115453e+00, 2.112094061064656358e+00,
            2.328551948050847642e+00, 2.544904892772519744e+00,
            2.760973722923115492e+00, 2.975775512437426773e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 15: {
        const double temp[15] = {
            1.551231069747375235e-01, 3.560718303314724942e-01,
            5.582062109125313087e-01, 7.606069572889918584e-01,
            9.631067821301482201e-01, 1.165652065603030252e+00,
            1.368219536992351770e+00, 1.570796326794896558e+00,
            1.773373116597441346e+00, 1.975940587986762864e+00,
            2.178485871459645118e+00, 2.380985696300801369e+00,
            2.583386442677261918e+00, 2.785520823258320622e+00,
            2.986469546615055481e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 16: {
        const double temp[16] = {
            1.457246820036738055e-01, 3.344986386876292461e-01,
            5.243866409035942144e-01, 7.145252532340252705e-01,
            9.047575323895165056e-01, 1.095033401803444439e+00,
            1.285331444322965311e+00, 1.475640280808194316e+00,
            1.665952372781598800e+00, 1.856261209266827805e+00,
            2.046559251786348455e+00, 2.236835121200276610e+00,
            2.427067400355767735e+00, 2.617206012686199124e+00,
            2.807094014902163703e+00, 2.995867971586119172e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 17: {
        const double temp[17] = {
            1.373998952992547817e-01, 3.153898594929282484e-01,
            4.944303818194982769e-01, 6.737074594242522529e-01,
            8.530732514258505539e-01, 1.032480728417239479e+00,
            1.211909966211469625e+00, 1.391350647015287434e+00,
            1.570796326794896558e+00, 1.750242006574505682e+00,
            1.929682687378323491e+00, 2.109111925172553637e+00,
            2.288519402163942562e+00, 2.467885194165540863e+00,
            2.647162271770294950e+00, 2.826202794096865034e+00,
            3.004192758290538556e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 18: {
        const double temp[18] = {
            1.299747364196768562e-01, 2.983460782092324792e-01,
            4.677113145328286592e-01, 6.373005058706191495e-01,
            8.069738930788195042e-01, 9.766871104439832640e-01,
            1.146421481056642211e+00, 1.316167494718022635e+00,
            1.485919440392653001e+00, 1.655673213197140115e+00,
            1.825425158871770481e+00, 1.995171172533150905e+00,
            2.164905543145809741e+00, 2.334618760510973612e+00,
            2.504292147719174189e+00, 2.673881339056964457e+00,
            2.843246575380560692e+00, 3.011617917170116066e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 19: {
        const double temp[19] = {
            1.233108673082312784e-01, 2.830497588453067537e-01,
            4.437316659960951482e-01, 6.046261769405451014e-01,
            7.656007620508340494e-01, 9.266134127998189030e-01,
            1.087646521650454945e+00, 1.248691224331339278e+00,
            1.409742336767428883e+00, 1.570796326794896558e+00,
            1.731850316822364233e+00, 1.892901429258453838e+00,
            2.053946131939338393e+00, 2.214979240789974213e+00,
            2.375991891538959067e+00, 2.536966476649248126e+00,
            2.697860987593697857e+00, 2.858542894744486418e+00,
            3.018281786281561629e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 20: {
        const double temp[20] = {
            1.172969277059561499e-01, 2.692452880289302186e-01,
            4.220907301111165855e-01, 5.751385026314284055e-01,
            7.282625848696072657e-01, 8.814230742890135639e-01,
            1.034603297590104276e+00, 1.187794926634099024e+00,
            1.340993178589955148e+00, 1.494194914310399636e+00,
            1.647397739279393480e+00, 1.800599474999837968e+00,
            1.953797726955694092e+00, 2.106989355999688840e+00,
            2.260169579300779663e+00, 2.413330068720185739e+00,
            2.566454150958364711e+00, 2.719501923478676364e+00,
            2.872347365560862897e+00, 3.024295725883836994e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 21: {
        const double temp[21] = {
            1.118422651428890996e-01, 2.567245837448891010e-01,
            4.024623099018152517e-01, 5.483930281810389662e-01,
            6.943966110110700862e-01, 8.404350520135058789e-01,
            9.864925055883793092e-01, 1.132561101012537597e+00,
            1.278636375242898637e+00, 1.424715475176742796e+00,
            1.570796326794896558e+00, 1.716877178413050320e+00,
            1.862956278346894479e+00, 2.009031552577255297e+00,
            2.155100148001413807e+00, 2.301157601576287348e+00,
            2.447196042578723141e+00, 2.593199625408754372e+00,
            2.739130343687977920e+00, 2.884868069844904070e+00,
            3.029750388446903919e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 22: {
        const double temp[22] = {
            1.068723357985259942e-01, 2.453165389983613109e-01,
            3.845781703583910915e-01, 5.240242709487281658e-01,
            6.635400754448063099e-01, 8.030892957063359150e-01,
            9.426568273796608333e-01, 1.082235198111836771e+00,
            1.221820208990359813e+00, 1.361409225664372169e+00,
            1.501000399130816065e+00, 1.640592254458977051e+00,
            1.780183427925420947e+00, 1.919772444599433303e+00,
            2.059357455477956123e+00, 2.198935826210132394e+00,
            2.338503357883457312e+00, 2.478052578144986917e+00,
            2.617568382641064950e+00, 2.757014483231401858e+00,
            2.896276114591431750e+00, 3.034720317791267163e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 23: {
        const double temp[23] = {
            1.023252788872632685e-01, 2.348791589702580174e-01,
            3.682157131008290119e-01, 5.017289283414202439e-01,
            6.353089402976822564e-01, 7.689210263823624825e-01,
            9.025507517347876041e-01, 1.036190996404462217e+00,
            1.169837785762829929e+00, 1.303488659735581257e+00,
            1.437141935303526186e+00, 1.570796326794896558e+00,
            1.704450718286266930e+00, 1.838103993854211859e+00,
            1.971754867826963187e+00, 2.105401657185330677e+00,
            2.239041901855005623e+00, 2.372671627207430411e+00,
            2.506283713292110971e+00, 2.639863725248372983e+00,
            2.773376940488963882e+00, 2.906713494619534988e+00,
            3.039267374702530056e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 24: {
        const double temp[24] = {
            9.814932949793685191e-02, 2.252936226353075833e-01,
            3.531886675690780741e-01, 4.812531951313686607e-01,
            6.093818382449566196e-01, 7.375413075437535770e-01,
            8.657177770401081052e-01, 9.939044422989454786e-01,
            1.122097523267250763e+00, 1.250294703417273112e+00,
            1.378494427506219200e+00, 1.506695545558101035e+00,
            1.634897108031692081e+00, 1.763098226083573916e+00,
            1.891297950172520004e+00, 2.019495130322542131e+00,
            2.147688211290847526e+00, 2.275874876549685233e+00,
            2.404051346046039761e+00, 2.532210815344836607e+00,
            2.660339458458424566e+00, 2.788403986020715042e+00,
            2.916299030954485616e+00, 3.043443324091856361e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 25: {
        const double temp[25] = {
            9.430083986305520805e-02, 2.164597408964339387e-01,
            3.393399712563371362e-01, 4.623830630132757524e-01,
            5.854877911108011812e-01, 7.086221837538611013e-01,
            8.317729718814276252e-01, 9.549336362382321308e-01,
            1.078100568411879845e+00, 1.201271573324181219e+00,
            1.324445197736386692e+00, 1.447620393135667038e+00,
            1.570796326794896558e+00, 1.693972260454126078e+00,
            1.817147455853406424e+00, 1.940321080265611897e+00,
            2.063492085177913271e+00, 2.186659017351560763e+00,
            2.309819681708365380e+00, 2.432970469835932015e+00,
            2.556104862478991713e+00, 2.679209590576517197e+00,
            2.802252682333456146e+00, 2.925132912693359177e+00,
            3.047291813726737963e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 26: {
        const double temp[26] = {
            9.074274842993197698e-02, 2.082924425598466356e-01,
            3.265362611165358309e-01, 4.449368152119130282e-01,
            5.633967073169293682e-01, 6.818851814129298639e-01,
            8.003894803353296394e-01, 9.189033445598993044e-01,
            1.037423319077439121e+00, 1.155947313793812103e+00,
            1.274473959424494041e+00, 1.393002286179807925e+00,
            1.511531546703289264e+00, 1.630061106886503852e+00,
            1.748590367409985191e+00, 1.867118694165299075e+00,
            1.985645339795981013e+00, 2.104169334512353995e+00,
            2.222689309029894034e+00, 2.341203173254463366e+00,
            2.459707472176863252e+00, 2.578195946272863637e+00,
            2.696655838377880254e+00, 2.815056392473257230e+00,
            2.933300211029946425e+00, 3.050849905159861208e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 27: {
        const double temp[27] = {
            8.744338280630299665e-02, 2.007190266590380412e-01,
            3.146635662674374112e-01, 4.287591577660783693e-01,
            5.429119513798658092e-01, 6.570923167092416195e-01,
            7.712879690777516561e-01, 8.854928869950799974e-01,
            9.997037539874953360e-01, 1.113918572282611930e+00,
            1.228136043468909699e+00, 1.342355260834552144e+00,
            1.456575541704195897e+00, 1.570796326794896558e+00,
            1.685017111885597219e+00, 1.799237392755240972e+00,
            1.913456610120883417e+00, 2.027674081307181186e+00,
            2.141888899602297780e+00, 2.256099766594712897e+00,
            2.370304684512041682e+00, 2.484500336880551608e+00,
            2.598680702209927418e+00, 2.712833495823714802e+00,
            2.826929087322355816e+00, 2.940873626930755158e+00,
            3.054149270783490078e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 28: {
        const double temp[28] = {
            8.437551461511597073e-02, 1.936769929947376179e-01,
            3.036239070914333316e-01, 4.137165857369637378e-01,
            5.238644768825679865e-01, 6.340389954584301213e-01,
            7.442282945111358128e-01, 8.544265718392254350e-01,
            9.646306371285441328e-01, 1.074838574917869272e+00,
            1.185049147889021492e+00, 1.295261501292316098e+00,
            1.405475003062348627e+00, 1.515689149557281068e+00,
            1.625903504032512048e+00, 1.736117650527444489e+00,
            1.846331152297477018e+00, 1.956543505700771624e+00,
            2.066754078671923622e+00, 2.176962016461248872e+00,
            2.287166081750567681e+00, 2.397364359078657081e+00,
            2.507553658131362884e+00, 2.617728176707225352e+00,
            2.727876067852829323e+00, 2.837968746498359618e+00,
            2.947915660595055609e+00, 3.057217138974677173e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 29: {
        const double temp[29] = {
            8.151560650977880684e-02, 1.871123137498062194e-01,
            2.933325857619472621e-01, 3.996936914666951446e-01,
            5.061081521563000063e-01, 6.125483562383020608e-01,
            7.190028636037067988e-01, 8.254660749671546283e-01,
            9.319349156915986976e-01, 1.038407544520296710e+00,
            1.144882777708662536e+00, 1.251359804334884807e+00,
            1.357838033080061679e+00, 1.464317002991565309e+00,
            1.570796326794896558e+00, 1.677275650598227807e+00,
            1.783754620509731437e+00, 1.890232849254908309e+00,
            1.996709875881130580e+00, 2.103185109069496406e+00,
            2.209657737898194529e+00, 2.316126578622638377e+00,
            2.422589789986086206e+00, 2.529044297351490833e+00,
            2.635484501433492888e+00, 2.741898962123098027e+00,
            2.848260067827845798e+00, 2.954480339839987035e+00,
            3.060077047080014268e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 30: {
        const double temp[30] = {
            7.884320726554944203e-02, 1.809780449917272049e-01,
            2.837160095793466730e-01, 3.865901987860504985e-01,
            4.895160050896970039e-01, 5.924667257887386018e-01,
            6.954313000299366943e-01, 7.984043170121235544e-01,
            9.013828087667156153e-01, 1.004365001539081037e+00,
            1.107349759228459130e+00, 1.210336308624476498e+00,
            1.313324092045794700e+00, 1.416312682230741693e+00,
            1.519301729274526558e+00, 1.622290924315266558e+00,
            1.725279971359051423e+00, 1.828268561543998416e+00,
            1.931256344965316618e+00, 2.034242894361334208e+00,
            2.137227652050712301e+00, 2.240209844823077390e+00,
            2.343188336577669340e+00, 2.446161353559856533e+00,
            2.549125927801054736e+00, 2.652076648500095946e+00,
            2.755002454803742395e+00, 2.857876644010446388e+00,
            2.960614608598065800e+00, 3.062749446324243507e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 31: {
        const double temp[31] = {
            7.634046205384431572e-02, 1.752332025619508515e-01,
            2.747099287638327669e-01, 3.743185619229329464e-01,
            4.739771829190733698e-01, 5.736599396529727946e-01,
            6.733561257504194764e-01, 7.730605060747958168e-01,
            8.727702114891848773e-01, 9.724835301003497134e-01,
            1.072199368669106478e+00, 1.171916986981363706e+00,
            1.271635855736122256e+00, 1.371355574944659095e+00,
            1.471075823713997366e+00, 1.570796326794896558e+00,
            1.670516829875795750e+00, 1.770237078645134021e+00,
            1.869956797853670860e+00, 1.969675666608429410e+00,
            2.069393284920686860e+00, 2.169109123489443292e+00,
            2.268822442100608239e+00, 2.368532147514997188e+00,
            2.468236527839373640e+00, 2.567932713936820210e+00,
            2.667615470670719802e+00, 2.767274091666860336e+00,
            2.866882724825960516e+00, 2.966359451027842375e+00,
            3.065252191535948967e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 32: {
        const double temp[32] = {
            7.399171309970958843e-02, 1.698418454282150103e-01,
            2.662579994723859311e-01, 3.628020075350028018e-01,
            4.593944730762095641e-01, 5.560103418005302167e-01,
            6.526392394594561219e-01, 7.492760951181414164e-01,
            8.459181315837993598e-01, 9.425636940046776546e-01,
            1.039211728068951679e+00, 1.135861522840293736e+00,
            1.232512573416362889e+00, 1.329164502391080749e+00,
            1.425817011963825376e+00, 1.522469852641529231e+00,
            1.619122800948263885e+00, 1.715775641625967740e+00,
            1.812428151198712367e+00, 1.909080080173430227e+00,
            2.005731130749499158e+00, 2.102380925520841437e+00,
            2.199028959585115572e+00, 2.295674522005993978e+00,
            2.392316558471651700e+00, 2.488953414130337105e+00,
            2.585582311789262899e+00, 2.682198180513583718e+00,
            2.778790646054790425e+00, 2.875334654117406963e+00,
            2.971750808161578217e+00, 3.067600940490083694e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      case 33: {
        const double temp[33] = {
            7.178317184275122276e-02, 1.647723231643112574e-01,
            2.583106041071417946e-01, 3.519729273095236199e-01,
            4.456822679082866334e-01, 5.394143214244183637e-01,
            6.331590254855161692e-01, 7.269114630504562857e-01,
            8.206689427646121082e-01, 9.144298626454031576e-01,
            1.008193204014774080e+00, 1.101958282220461438e+00,
            1.195724613675799519e+00, 1.289491840051302685e+00,
            1.383259682348271680e+00, 1.477027911291552309e+00,
            1.570796326794896558e+00, 1.664564742298240807e+00,
            1.758332971241521436e+00, 1.852100813538490431e+00,
            1.945868039913993597e+00, 2.039634371369331678e+00,
            2.133399449575018814e+00, 2.227162790944389847e+00,
            2.320923710825181008e+00, 2.414681190539337052e+00,
            2.508433628104277169e+00, 2.602178332165374641e+00,
            2.695910385681506316e+00, 2.789619726280269330e+00,
            2.883282049482651210e+00, 2.976820330425481664e+00,
            3.069809481747042046e+00};
        result.theta_points[i] = temp[i];
        break;
      }
      default: {
        result.number_of_theta_points = 0;
        result.number_of_phi_points = 0;
        break;
      }
    }
  }

  const double d_phi = (2.0 * M_PI) / result.number_of_phi_points;
  for (int i = 0; i < result.number_of_phi_points; ++i) {
    result.phi_points[i] = i * d_phi;
  }
  return result;
}
// [spectre_cce_grid_point_locations]

// NOLINTEND

void test_spectre_cce_grid_point_locations() {
  for (size_t l_max = 4; l_max < 33; ++l_max) {
    const SpectreCceGridPointLocations t =
        spectre_ylm_theta_phi_points(static_cast<int>(l_max));
    const Spherepack ylm{l_max, l_max};
    CHECK(ylm.theta_points().size() ==
          static_cast<size_t>(t.number_of_theta_points));
    CHECK(ylm.phi_points().size() ==
          static_cast<size_t>(t.number_of_phi_points));
    for (size_t i = 0; i < ylm.theta_points().size(); ++i) {
      // NOLINTNEXTLINE
      CHECK(ylm.theta_points()[i] == approx(t.theta_points[i]));
    }
    for (size_t i = 0; i < ylm.phi_points().size(); ++i) {
      // NOLINTNEXTLINE
      CHECK(ylm.phi_points()[i] == approx(t.phi_points[i]));
    }
  }
}

using SecondDeriv = Spherepack::SecondDeriv;

void test_prolong_restrict() {
  Spherepack ylm_a(10, 10);

  const YlmTestFunctions::FuncA func_a{};
  const YlmTestFunctions::FuncB func_b{};
  const YlmTestFunctions::FuncC func_c{};

  const auto& theta = ylm_a.theta_points();
  const auto& phi = ylm_a.phi_points();

  const auto u_a = func_a.func(theta, phi);
  const auto u_b = func_b.func(theta, phi);
  const auto u_c = func_c.func(theta, phi);

  const auto u_coef_a = ylm_a.phys_to_spec(u_a);
  {
    Spherepack ylm_b(10, 7);
    const auto u_coef_a2b = ylm_a.prolong_or_restrict(u_coef_a, ylm_b);
    const auto u_coef_a2b2a = ylm_b.prolong_or_restrict(u_coef_a2b, ylm_a);
    const auto u_b_test = ylm_a.spec_to_phys(u_coef_a2b2a);
    CHECK_ITERABLE_APPROX(u_b, u_b_test);
  }

  {
    Spherepack ylm_c(6, 2);
    const auto u_coef_a2c = ylm_a.prolong_or_restrict(u_coef_a, ylm_c);
    const auto u_coef_a2c2a = ylm_c.prolong_or_restrict(u_coef_a2c, ylm_a);
    const auto u_c_test = ylm_a.spec_to_phys(u_coef_a2c2a);
    CHECK_ITERABLE_APPROX(u_c, u_c_test);
  }
}

void test_loop_over_offset(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);

  // Fill data vectors
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * physical_stride;
  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill analytic solution
  const std::vector<double>& theta = ylm_spherepack.theta_points();
  const std::vector<double>& phi = ylm_spherepack.phi_points();
  for (size_t off = 0; off < physical_stride; ++off) {
    func.func(&u, physical_stride, off, theta, phi);
  }

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec_all_offsets(u_spec.data(), u.data(),
                                          physical_stride);

  // Test whether phys_to_spec and spec_to_phys are inverses.
  {
    std::vector<double> u_test(physical_size);
    ylm_spherepack.spec_to_phys_all_offsets(u_test.data(), u_spec.data(),
                                            physical_stride);
    for (size_t s = 0; s < physical_size; ++s) {
      CHECK(u[s] == approx(u_test[s]));
    }
  }

  // Test simplified interface
  {
    auto u_spec_simple =
        ylm_spherepack.phys_to_spec_all_offsets(u, physical_stride);
    CHECK_ITERABLE_APPROX(u_spec, u_spec_simple);

    auto u_test =
        ylm_spherepack.spec_to_phys_all_offsets(u_spec_simple, physical_stride);
    CHECK_ITERABLE_APPROX(u, u_test);
  }

  // Test gradient
  {
    std::vector<std::vector<double>> duteststor(2);
    std::vector<std::vector<double>> dustor(2);
    std::vector<std::vector<double>> duSpecstor(2);
    for (size_t i = 0; i < 2; ++i) {
      dustor[i].resize(physical_size);
      duteststor[i].resize(physical_size);
      duSpecstor[i].resize(physical_size);
    }
    std::array<double*, 2> du({{dustor[0].data(), dustor[1].data()}});
    std::array<double*, 2> duSpec(
        {{duSpecstor[0].data(), duSpecstor[1].data()}});
    std::array<double*, 2> dutest(
        {{duteststor[0].data(), duteststor[1].data()}});

    // Fill analytic result
    for (size_t off = 0; off < physical_stride; ++off) {
      func.dfunc(&dutest, physical_stride, off, theta, phi);
    }

    // Differentiate
    ylm_spherepack.gradient_from_coefs_all_offsets(duSpec, u_spec.data(),
                                                   physical_stride);
    ylm_spherepack.gradient_all_offsets(du, u.data(), physical_stride);

    // Test vs analytic result
    for (size_t d = 0; d < 2; ++d) {
      for (size_t s = 0; s < physical_size; ++s) {
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(du, d)[s]));
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(duSpec, d)[s]));
      }
    }

    // Test simplified interface of gradient
    {
      auto du_simple = ylm_spherepack.gradient_all_offsets(u, physical_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
      du_simple = ylm_spherepack.gradient_from_coefs_all_offsets(
          u_spec, physical_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
    }
  }
}

void test_theta_phi_points(
    const size_t l_max, const size_t m_max,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);

  // Fill with analytic function
  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();
  DataVector u(ylm_spherepack.physical_size());
  func.func(&u, 1, 0, theta, phi);

  const auto theta_phi = ylm_spherepack.theta_phi_points();
  DataVector u_test(ylm_spherepack.physical_size());
  // fill pointwise using offset
  for (size_t s = 0; s < ylm_spherepack.physical_size(); ++s) {
    func.func(&u_test, 1, s, {gsl::at(theta_phi, 0)[s]},
              {gsl::at(theta_phi, 1)[s]});
  }
  CHECK_ITERABLE_APPROX(u, u_test);
}

void test_phys_to_spec(const size_t l_max, const size_t m_max,
                       const size_t physical_stride,
                       const size_t spectral_stride,
                       const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  const size_t n_th = l_max + 1;
  const size_t n_ph = 2 * m_max + 1;
  const size_t physical_size = n_th * n_ph * physical_stride;
  const size_t spectral_size = 2 * (l_max + 1) * (m_max + 1) * spectral_stride;

  Spherepack ylm_spherepack(l_max, m_max);
  CHECK(physical_size == ylm_spherepack.physical_size() * physical_stride);
  CHECK(spectral_size == ylm_spherepack.spectral_size() * spectral_stride);

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test whether phys_to_spec and spec_to_phys are inverses.
  {
    std::vector<double> u_test(physical_size);
    std::vector<double> u_spec_test(spectral_size);
    ylm_spherepack.phys_to_spec(u_spec_test.data(), u.data(), physical_stride,
                                0, spectral_stride, 0);
    ylm_spherepack.spec_to_phys(u_test.data(), u_spec.data(), spectral_stride,
                                0, physical_stride, 0);
    for (size_t s = 0; s < physical_size; s += physical_stride) {
      CHECK(u[s] == approx(u_test[s]));
    }
    for (size_t s = 0; s < spectral_size; s += spectral_stride) {
      CHECK(u_spec[s] == u_spec_test[s]);
    }
  }

  // Test simplified interface of phys_to_spec/spec_to_phys
  if (physical_stride == 1 and spectral_stride == 1) {
    auto u_spec_simple = ylm_spherepack.phys_to_spec(u);
    CHECK_ITERABLE_APPROX(u_spec, u_spec_simple);

    auto u_test = ylm_spherepack.spec_to_phys(u_spec_simple);
    CHECK_ITERABLE_APPROX(u, u_test);
  }
}

void test_gradient(const size_t l_max, const size_t m_max,
                   const size_t physical_stride, const size_t spectral_stride,
                   const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test gradient
  {
    std::vector<std::vector<double>> duteststor(2);
    std::vector<std::vector<double>> dustor(2);
    std::vector<std::vector<double>> duSpecstor(2);
    for (size_t i = 0; i < 2; ++i) {
      dustor[i].resize(physical_size);
      duteststor[i].resize(physical_size);
      duSpecstor[i].resize(physical_size);
    }
    std::array<double*, 2> du({{dustor[0].data(), dustor[1].data()}});
    std::array<double*, 2> duSpec(
        {{duSpecstor[0].data(), duSpecstor[1].data()}});
    std::array<double*, 2> dutest(
        {{duteststor[0].data(), duteststor[1].data()}});

    // Differentiate
    ylm_spherepack.gradient_from_coefs(duSpec, u_spec.data(), spectral_stride,
                                       0, physical_stride, 0);
    ylm_spherepack.gradient(du, u.data(), physical_stride, 0);

    // Test vs analytic result
    func.dfunc(&dutest, physical_stride, 0, theta, phi);
    for (size_t d = 0; d < 2; ++d) {
      for (size_t s = 0; s < physical_size; s += physical_stride) {
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(du, d)[s]));
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(duSpec, d)[s]));
      }
    }

    if (physical_stride == 1 && spectral_stride == 1) {
      // Without strides and offsets.
      ylm_spherepack.gradient_from_coefs(duSpec, u_spec.data());
      ylm_spherepack.gradient(du, u.data());
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(du, d)[s]));
          CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(duSpec, d)[s]));
        }
      }

      // Test simplified interface of gradient
      auto du_simple = ylm_spherepack.gradient(u);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
      du_simple = ylm_spherepack.gradient_from_coefs(u_spec);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
    } else {
      // Test simplified interface of gradient for non-unit stride
      auto du_simple = ylm_spherepack.gradient(u, physical_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < ylm_spherepack.physical_size(); ++s) {
          CHECK(gsl::at(dutest, d)[s * physical_stride] ==
                approx(du_simple.get(d)[s]));
        }
      }
      du_simple = ylm_spherepack.gradient_from_coefs(u_spec, spectral_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < ylm_spherepack.physical_size(); ++s) {
          CHECK(gsl::at(dutest, d)[s * physical_stride] ==
                approx(du_simple.get(d)[s]));
        }
      }
    }
  }
}

void test_second_derivative(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test second_derivative
  {
    SecondDeriv ddu(physical_size);
    SecondDeriv ddutest(physical_size);

    std::vector<std::vector<double>> dustor(2);
    for (size_t i = 0; i < 2; ++i) {
      dustor[i].resize(physical_size);
    }
    std::array<double*, 2> du{{dustor[0].data(), dustor[1].data()}};

    // Differentiate
    ylm_spherepack.second_derivative(du, &ddu, u.data(), physical_stride, 0);

    // Test ylm_spherepack derivative against func analytical result
    func.ddfunc(&ddutest, physical_stride, 0, theta, phi);
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        for (size_t s = 0; s < physical_size; s += physical_stride) {
          CHECK(ddutest.get(i, j)[s] == approx(ddu.get(i, j)[s]));
        }
      }
    }

    if (physical_stride == 1 && spectral_stride == 1) {
      ylm_spherepack.second_derivative(du, &ddu, u.data());
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          CHECK_ITERABLE_APPROX(ddutest.get(i, j), ddu.get(i, j));
        }
      }

      // Test first_and_second_derivative
      auto deriv_test = ylm_spherepack.first_and_second_derivative(u);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(std::get<0>(deriv_test).get(i)[s] == approx(gsl::at(du, i)[s]));
        }
        for (size_t j = 0; j < 2; ++j) {
          CHECK_ITERABLE_APPROX(std::get<1>(deriv_test).get(i, j),
                                ddu.get(i, j));
        }
      }
    }
  }
}

void test_scalar_laplacian(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test scalar_laplacian
  {
    DataVector slaptest(physical_size);
    DataVector slap(physical_size);
    DataVector slapSpec(physical_size);

    // Differentiate
    ylm_spherepack.scalar_laplacian(slap.data(), u.data(), physical_stride, 0);
    ylm_spherepack.scalar_laplacian_from_coefs(
        slapSpec.data(), u_spec.data(), spectral_stride, 0, physical_stride, 0);

    // Test ylm_spherepack derivative against func analytical result
    func.scalar_laplacian(&slaptest, physical_stride, 0, theta, phi);
    for (size_t s = 0; s < physical_size; s += physical_stride) {
      CHECK(slaptest[s] == approx(slap[s]));
      CHECK(slaptest[s] == approx(slapSpec[s]));
    }

    // Test the default arguments for stride and offset
    if (physical_stride == 1 && spectral_stride == 1) {
      ylm_spherepack.scalar_laplacian(slap.data(), u.data());
      ylm_spherepack.scalar_laplacian_from_coefs(slapSpec.data(),
                                                 u_spec.data());
      CHECK_ITERABLE_APPROX(slaptest, slap);
      CHECK_ITERABLE_APPROX(slaptest, slapSpec);

      // Test simplified interface of scalar_laplacian
      auto slap1 = ylm_spherepack.scalar_laplacian(u);
      auto slap2 = ylm_spherepack.scalar_laplacian_from_coefs(u_spec);
      CHECK_ITERABLE_APPROX(slaptest, slap1);
      CHECK_ITERABLE_APPROX(slaptest, slap2);
    }
  }
}

void test_interpolation(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);
  // test with a seperate instance if it can use interpolation_info from the
  // first one.
  Spherepack ylm_spherepack_2(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test interpolation
  {
    // Choose random points
    DataVector thetas(50);
    DataVector phis(50);
    {
      std::uniform_real_distribution<double> ran(0.0, 1.0);
      MAKE_GENERATOR(gen);
      // Here we generate 10 * 5 different random (theta, phi) pairs. Each
      // iteration adds five more elements to the vectors of `thetas` and
      // `phis`, so the index increases by five.
      for (size_t n = 0; n < 10; ++n) {
        const double th = (2.0 * ran(gen) - 1.0) * M_PI;
        const double ph = 2.0 * ran(gen) * M_PI;

        thetas.at(n * 5) = th;
        phis.at(n * 5) = ph;

        // For the next point, increase ph by 2pi so it is out of range.
        // Should be equivalent to the first point.
        thetas.at(n * 5 + 1) = th;
        phis.at(n * 5 + 1) = ph + 2.0 * M_PI;

        // For the next point, decrease ph by 2pi so it is out of range.
        // Should be equivalent to the first point.
        thetas.at(n * 5 + 2) = th;
        phis.at(n * 5 + 2) = ph - 2.0 * M_PI;

        // For the next point, use negative theta so it is out of range,
        // and also add pi to phi.
        // Should be equivalent to the first point.
        thetas.at(n * 5 + 3) = -th;
        phis.at(n * 5 + 3) = ph + M_PI;

        // For the next point, theta -> 2pi - theta so that theta is out of
        // range.  Also add pi to Phi.
        // Should be equivalent to the first point.
        thetas.at(n * 5 + 4) = 2.0 * M_PI - th;
        phis.at(n * 5 + 4) = ph + M_PI;
      }
    }

    std::array<DataVector, 2> points{std::move(thetas), std::move(phis)};

    // Get interp info
    auto interpolation_info = ylm_spherepack.set_up_interpolation_info(points);

    // Interpolate
    DataVector uintPhys(interpolation_info.size());
    DataVector uintSpec(interpolation_info.size());

    DataVector uintPhys2(interpolation_info.size());
    DataVector uintSpec2(interpolation_info.size());

    ylm_spherepack.interpolate(make_not_null(&uintPhys), u.data(),
                               interpolation_info, physical_stride, 0);
    ylm_spherepack.interpolate_from_coefs(make_not_null(&uintSpec), u_spec,
                                          interpolation_info, spectral_stride);

    ylm_spherepack_2.interpolate(make_not_null(&uintPhys2), u.data(),
                                 interpolation_info, physical_stride, 0);
    ylm_spherepack_2.interpolate_from_coefs(
        make_not_null(&uintSpec2), u_spec, interpolation_info, spectral_stride);

    // Test vs analytic solution
    DataVector uintanal(interpolation_info.size());
    for (size_t s = 0; s < uintanal.size(); ++s) {
      func.func(&uintanal, 1, s, {points[0][s]}, {points[1][s]});
      CHECK(uintanal[s] == approx(uintPhys[s]));
      CHECK(uintanal[s] == approx(uintSpec[s]));

      CHECK(uintanal[s] == approx(uintPhys2[s]));
      CHECK(uintanal[s] == approx(uintSpec2[s]));
    }

    // Test for angles out of range.
    for (size_t s = 0; s < uintanal.size() / 5; ++s) {
      // All answers should agree in each group of five, since the values
      // of all the out-of-range angles should represent the same point.
      CHECK(uintPhys[5 * s + 1] == approx(uintPhys[5 * s]));
      CHECK(uintPhys[5 * s + 2] == approx(uintPhys[5 * s]));
      CHECK(uintPhys[5 * s + 3] == approx(uintPhys[5 * s]));
      CHECK(uintPhys[5 * s + 4] == approx(uintPhys[5 * s]));

      CHECK(uintPhys2[5 * s + 1] == approx(uintPhys2[5 * s]));
      CHECK(uintPhys2[5 * s + 2] == approx(uintPhys2[5 * s]));
      CHECK(uintPhys2[5 * s + 3] == approx(uintPhys2[5 * s]));
      CHECK(uintPhys2[5 * s + 4] == approx(uintPhys2[5 * s]));
    }

    // Tests default values of stride and offset.
    if (physical_stride == 1 && spectral_stride == 1) {
      ylm_spherepack.interpolate(make_not_null(&uintPhys), u.data(),
                                 interpolation_info);
      ylm_spherepack_2.interpolate(make_not_null(&uintPhys2), u.data(),
                                   interpolation_info);
      for (size_t s = 0; s < uintanal.size(); ++s) {
        CHECK(uintanal[s] == approx(uintPhys[s]));
        CHECK(uintanal[s] == approx(uintPhys2[s]));
      }
    }

    // Test simplified interpolation interface
    if (physical_stride == 1) {
      auto test_interp = ylm_spherepack.interpolate(u, points);
      auto test_interp_2 = ylm_spherepack_2.interpolate(u, points);

      for (size_t s = 0; s < uintanal.size(); ++s) {
        CHECK(uintanal[s] == approx(test_interp[s]));
        CHECK(uintanal[s] == approx(test_interp_2[s]));
      }
    }
    if (spectral_stride == 1) {
      auto test_interp = ylm_spherepack.interpolate_from_coefs(u_spec, points);
      auto test_interp_2 =
          ylm_spherepack_2.interpolate_from_coefs(u_spec, points);
      for (size_t s = 0; s < uintanal.size(); ++s) {
        CHECK(uintanal[s] == approx(test_interp[s]));
        CHECK(uintanal[s] == approx(test_interp_2[s]));
      }
    }
  }
}

void test_integral(const size_t l_max, const size_t m_max,
                   const size_t physical_stride, const size_t spectral_stride,
                   const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  Spherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size);
  DataVector u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test integral
  if (physical_stride == 1) {
    const double integral1 = ylm_spherepack.definite_integral(u.data());
    const auto& weights = ylm_spherepack.integration_weights();
    double integral2 = 0.0;
    for (size_t s = 0; s < physical_size; ++s) {
      integral2 += u[s] * weights[s];
    }
    const double integral_test = func.integral();
    CHECK(integral1 == approx(integral_test));
    CHECK(integral2 == approx(integral_test));
  }

  // Test average
  if (spectral_stride == 1) {
    const auto avg = ylm_spherepack.average(u_spec);
    const double avg_test = func.integral() / (4.0 * M_PI);
    CHECK(avg == approx(avg_test));
  }

  // Test add_constant
  if (spectral_stride == 1) {
    const double value_to_add = 1.367;
    DataVector u_spec_plus_value = u_spec;
    ylm_spherepack.add_constant(&u_spec_plus_value, value_to_add);
    const auto avg = ylm_spherepack.average(u_spec_plus_value);
    const double avg_test = func.integral() / (4.0 * M_PI) + value_to_add;
    CHECK(avg == approx(avg_test));
  }
}

void test_Spherepack(const size_t l_max, const size_t m_max,
                     const size_t physical_stride, const size_t spectral_stride,
                     const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  test_phys_to_spec(l_max, m_max, physical_stride, spectral_stride, func);
  test_gradient(l_max, m_max, physical_stride, spectral_stride, func);
  test_second_derivative(l_max, m_max, physical_stride, spectral_stride, func);
  test_scalar_laplacian(l_max, m_max, physical_stride, spectral_stride, func);
  test_interpolation(l_max, m_max, physical_stride, spectral_stride, func);
  test_integral(l_max, m_max, physical_stride, spectral_stride, func);
  if (physical_stride == 1) {
    test_theta_phi_points(l_max, m_max, func);
  }
}

void test_memory_pool() {
  const size_t n_pts = 100;
  Spherepack_detail::MemoryPool pool;

  // Fill all the temps.
  std::vector<double>& tmp1 = pool.get(n_pts);
  std::vector<double>& tmp2 = pool.get(n_pts);
  std::vector<double>& tmp3 = pool.get(n_pts);
  std::vector<double>& tmp4 = pool.get(n_pts);
  std::vector<double>& tmp5 = pool.get(n_pts);
  std::vector<double>& tmp6 = pool.get(n_pts);
  std::vector<double>& tmp7 = pool.get(n_pts);
  std::vector<double>& tmp8 = pool.get(n_pts);
  std::vector<double>& tmp9 = pool.get(n_pts);

  // Allocate more than the number of available temps
  CHECK_THROWS_WITH((pool.get(n_pts)),
                    Catch::Matchers::ContainsSubstring(
                        "Attempt to allocate more than 9 temps."));

  // Clear too early.
  CHECK_THROWS_WITH((pool.clear()),
                    Catch::Matchers::ContainsSubstring(
                        "Attempt to clear element that is in use"));

  // Free all the temps (not necessarily in the same order as get).
  pool.free(tmp1);
  pool.free(tmp3);
  pool.free(tmp2);
  pool.free(tmp5);
  pool.free(tmp4);
  pool.free(tmp6);
  pool.free(tmp8);
  pool.free(tmp9);
  pool.free(tmp7);

  // Get a vector of a smaller size.  Here the vector returned will
  // still have size n_pts, since there is only a resize when the
  // vector is larger than the current size.
  auto& vec1 = pool.get(n_pts / 2);
  CHECK(vec1.size() == n_pts);
  pool.free(vec1);

  // Get a vector of a larger size.  Here the vector returned will
  auto& vec2 = pool.get(n_pts * 2);
  CHECK(vec2.size() == n_pts * 2);
  pool.free(vec2);

  // Clearing the temps resets all the sizes.
  pool.clear();

  // Now the size should be n_pts/2.
  auto& vec3 = pool.get(n_pts / 2);
  CHECK(vec3.size() == n_pts / 2);
  pool.free(vec3);
  pool.clear();

  std::vector<double> dum1(1, 0.0);
  CHECK_THROWS_WITH((pool.free(dum1)),
                    Catch::Matchers::ContainsSubstring(
                        "Attempt to free temp that was never allocated."));
  CHECK_THROWS_WITH((pool.free(make_not_null(dum1.data()))),
                    Catch::Matchers::ContainsSubstring(
                        "Attempt to free temp that was never allocated."));

  std::vector<double> dum2;
  CHECK_THROWS_WITH((pool.free(dum2)),
                    Catch::Matchers::ContainsSubstring(
                        "Attempt to free temp that was never allocated."));
}

void test_ylm_errors() {
  CHECK_THROWS_WITH((Spherepack(1, 1)), Catch::Matchers::ContainsSubstring(
                                            "Must use l_max>=2, not l_max=1"));
  CHECK_THROWS_WITH((Spherepack(2, 1)), Catch::Matchers::ContainsSubstring(
                                            "Must use m_max>=2, not m_max=1"));
  CHECK_THROWS_WITH(
      ([]() {
        Spherepack ylm(4, 3);
        const auto interp_info =
            ylm.set_up_interpolation_info(std::array<DataVector, 2>{
                DataVector{0.1, 0.3}, DataVector{0.2, 0.3}});
        Spherepack ylm_wrong_l_max(5, 3);
        DataVector res{2};
        // no need to initialize as the values should not be accessed
        const DataVector spectral_values{ylm_wrong_l_max.spectral_size()};
        ylm_wrong_l_max.interpolate_from_coefs(make_not_null(&res),
                                               spectral_values, interp_info);
      }()),
      Catch::Matchers::ContainsSubstring(
          "Different l_max for InterpolationInfo (4) "
          "and Spherepack instance (5)"));
  CHECK_THROWS_WITH(
      ([]() {
        Spherepack ylm(4, 3);
        const auto interp_info =
            ylm.set_up_interpolation_info(std::array<DataVector, 2>{
                DataVector{0.1, 0.3}, DataVector{0.2, 0.3}});
        Spherepack ylm_wrong_m_max(4, 4);
        DataVector res{2};
        // no need to initialize as the values should not be accessed
        const DataVector spectral_values{ylm_wrong_m_max.spectral_size()};
        ylm_wrong_m_max.interpolate_from_coefs(make_not_null(&res),
                                               spectral_values, interp_info);
      }()),
      Catch::Matchers::ContainsSubstring(
          "Different m_max for InterpolationInfo (3) "
          "and Spherepack instance (4)"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Spherepack",
                  "[ApparentHorizonFinder][Unit]") {
  test_spectre_cce_grid_point_locations();

  test_memory_pool();
  test_ylm_errors();

  for (size_t l_max = 3; l_max < 5; ++l_max) {
    for (size_t m_max = 2; m_max <= l_max; ++m_max) {
      for (size_t physical_stride = 1; physical_stride <= 4;
           physical_stride += 3) {
        for (size_t spectral_stride = 1; spectral_stride <= 4;
             spectral_stride += 3) {
          test_Spherepack(l_max, m_max, physical_stride, spectral_stride,
                          YlmTestFunctions::Y00());
          test_Spherepack(l_max, m_max, physical_stride, spectral_stride,
                          YlmTestFunctions::Y10());
          test_Spherepack(l_max, m_max, physical_stride, spectral_stride,
                          YlmTestFunctions::Y11());
        }
      }
    }
  }

  for (size_t l_max = 3; l_max < 5; ++l_max) {
    for (size_t m_max = 2; m_max <= l_max; ++m_max) {
      test_loop_over_offset(l_max, m_max, 3, YlmTestFunctions::Y00());
      test_loop_over_offset(l_max, m_max, 3, YlmTestFunctions::Y10());
      test_loop_over_offset(l_max, m_max, 3, YlmTestFunctions::Y11());
      test_loop_over_offset(l_max, m_max, 1, YlmTestFunctions::Y11());
    }
  }

  test_prolong_restrict();

  Spherepack s(4, 4);
  auto s_copy(s);
  CHECK(s_copy == s);
  test_move_semantics(std::move(s), s_copy, 6_st, 5_st);
}

}  // namespace ylm
