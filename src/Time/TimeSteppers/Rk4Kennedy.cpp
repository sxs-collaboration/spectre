// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk4Kennedy.hpp"

namespace TimeSteppers {

size_t Rk4Kennedy::order() const { return 4; }

double Rk4Kennedy::stable_step() const { return 2.1172491998184686; }

size_t Rk4Kennedy::imex_order() const { return 4; }

size_t Rk4Kennedy::implicit_stage_order() const { return 2; }

// The numbers in the tableaus are not exact.  Despite being written
// as rationals, the true values are irrational numbers.
const RungeKutta::ButcherTableau& Rk4Kennedy::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0},
      // Substep coefficients
      {{1.0 / 2.0},
       {13861.0 / 62500.0, 6889.0 / 62500.0},
       {-116923316275.0 / 2393684061468.0, -2731218467317.0 / 15368042101831.0,
        9408046702089.0 / 11113171139209.0},
       {-451086348788.0 / 2902428689909.0, -2682348792572.0 / 7519795681897.0,
        12662868775082.0 / 11960479115383.0,
        3355817975965.0 / 11060851509271.0},
       {647845179188.0 / 3216320057751.0, 73281519250.0 / 8382639484533.0,
        552539513391.0 / 3454668386233.0, 3354512671639.0 / 8306763924573.0,
        4040.0 / 17871.0}},
      // Result coefficients
      {82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
       -2260.0 / 8211.0, 1.0 / 4.0},
      // Coefficients for the embedded method for generating an error measure.
      {4586570599.0 / 29645900160.0, 0.0, 178811875.0 / 945068544.0,
       814220225.0 / 1159782912.0, -3700637.0 / 11593932.0, 61727.0 / 225920.0},
      // Dense output coefficient polynomials
      {{0.0, 6943876665148.0 / 7220017795957.0, -54480133.0 / 30881146.0,
        6818779379841.0 / 7100303317025.0},
       {},
       {0.0, 7640104374378.0 / 9702883013639.0, -11436875.0 / 14766696.0,
        2173542590792.0 / 12501825683035.0},
       {0.0, -20649996744609.0 / 7521556579894.0, 174696575.0 / 18121608.0,
        -31592104683404.0 / 5083833661969.0},
       {0.0, 8854892464581.0 / 2390941311638.0, -12120380.0 / 966161.0,
        61146701046299.0 / 7138195549469.0},
       {0.0, -11397109935349.0 / 6675773540249.0, 3843.0 / 706.0,
        -17219254887155.0 / 4939391667607.0}}};
  return tableau;
}

const ImexRungeKutta::ImplicitButcherTableau&
Rk4Kennedy::implicit_butcher_tableau() const {
  static const ImplicitButcherTableau tableau{
      {{1.0 / 4.0, 1.0 / 4.0},
       {8611.0 / 62500.0, -1743.0 / 31250.0, 1.0 / 4.0},
       {5012029.0 / 34652500.0, -654441.0 / 2922500.0, 174375.0 / 388108.0,
        1.0 / 4.0},
       {15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0,
        730878875.0 / 902184768.0, 2285395.0 / 8070912.0, 1.0 / 4.0},
       {82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
        -2260.0 / 8211.0, 1.0 / 4.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk4Kennedy::my_PUP_ID = 0;  // NOLINT
