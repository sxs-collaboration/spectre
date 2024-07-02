// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk3Kennedy.hpp"

namespace TimeSteppers {

size_t Rk3Kennedy::order() const { return 3; }

double Rk3Kennedy::stable_step() const { return 1.832102281377816; }

size_t Rk3Kennedy::imex_order() const { return 3; }

size_t Rk3Kennedy::implicit_stage_order() const { return 2; }

// The numbers in the tableaus are not exact.  Despite being written
// as rationals, the true values are irrational numbers.
const RungeKutta::ButcherTableau& Rk3Kennedy::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {1767732205903.0 / 2027836641118.0, 3.0 / 5.0, 1.0},
      // Substep coefficients
      {{1767732205903.0 / 2027836641118.0},
       {5535828885825.0 / 10492691773637.0, 788022342437.0 / 10882634858940.0},
       {6485989280629.0 / 16251701735622.0, -4246266847089.0 / 9704473918619.0,
        10755448449292.0 / 10357097424841.0}},
      // Result coefficients
      {1471266399579.0 / 7840856788654.0, -4482444167858.0 / 7529755066697.0,
       11266239266428.0 / 11593286722821.0, 1767732205903.0 / 4055673282236.0},
      // Coefficients for the embedded method for generating an error measure.
      {2756255671327.0 / 12835298489170.0, -10771552573575.0 / 22201958757719.0,
       9247589265047.0 / 10645013368117.0, 2193209047091.0 / 5459859503100.0},
      // Dense output coefficient polynomials
      {{0.0, 4655552711362.0 / 22874653954995.0,
        -215264564351.0 / 13552729205753.0},
       {0.0, -18682724506714.0 / 9892148508045.0,
        17870216137069.0 / 13817060693119.0},
       {0.0, 34259539580243.0 / 13192909600954.0,
        -28141676662227.0 / 17317692491321.0},
       {0.0, 584795268549.0 / 6622622206610.0,
        2508943948391.0 / 7218656332882.0}}};
  return tableau;
}

const ImexRungeKutta::ImplicitButcherTableau&
Rk3Kennedy::implicit_butcher_tableau() const {
  static const ImplicitButcherTableau tableau{
      {{1767732205903.0 / 4055673282236.0, 1767732205903.0 / 4055673282236.0},
       {2746238789719.0 / 10658868560708.0, -640167445237.0 / 6845629431997.0,
        1767732205903.0 / 4055673282236.0},
       {1471266399579.0 / 7840856788654.0, -4482444167858.0 / 7529755066697.0,
        11266239266428.0 / 11593286722821.0,
        1767732205903.0 / 4055673282236.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk3Kennedy::my_PUP_ID = 0;  // NOLINT
