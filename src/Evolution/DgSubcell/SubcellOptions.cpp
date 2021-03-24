// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SubcellOptions.hpp"

#include <pup.h>

namespace evolution::dg::subcell {
SubcellOptions::SubcellOptions(double initial_data_rdmp_delta0,
                               double initial_data_rdmp_epsilon,
                               double rdmp_delta0, double rdmp_epsilon,
                               double initial_data_persson_exponent,
                               double persson_exponent,
                               bool always_use_subcells) noexcept
    : initial_data_rdmp_delta0_(initial_data_rdmp_delta0),
      initial_data_rdmp_epsilon_(initial_data_rdmp_epsilon),
      rdmp_delta0_(rdmp_delta0),
      rdmp_epsilon_(rdmp_epsilon),
      initial_data_persson_exponent_(initial_data_persson_exponent),
      persson_exponent_(persson_exponent),
      always_use_subcells_(always_use_subcells) {}

void SubcellOptions::pup(PUP::er& p) noexcept {
  p | initial_data_rdmp_delta0_;
  p | initial_data_rdmp_epsilon_;
  p | rdmp_delta0_;
  p | rdmp_epsilon_;
  p | initial_data_persson_exponent_;
  p | persson_exponent_;
  p | always_use_subcells_;
}

bool operator==(const SubcellOptions& lhs, const SubcellOptions& rhs) noexcept {
  return lhs.initial_data_rdmp_delta0() == rhs.initial_data_rdmp_delta0() and
         lhs.initial_data_rdmp_epsilon() == rhs.initial_data_rdmp_epsilon() and
         lhs.rdmp_delta0() == rhs.rdmp_delta0() and
         lhs.rdmp_epsilon() == rhs.rdmp_epsilon() and
         lhs.initial_data_persson_exponent() ==
             rhs.initial_data_persson_exponent() and
         lhs.persson_exponent() == rhs.persson_exponent() and
         lhs.always_use_subcells() == rhs.always_use_subcells();
}

bool operator!=(const SubcellOptions& lhs, const SubcellOptions& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace evolution::dg::subcell
