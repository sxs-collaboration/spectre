// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"

#include <random>

#include "Domain/Amr/Flag.hpp"

namespace amr::Criteria {
Random::Random(const double do_something_fraction,
               const size_t maximum_refinement_level)
    : do_something_fraction_(do_something_fraction),
      maximum_refinement_level_(maximum_refinement_level) {}

Random::Random(CkMigrateMessage* msg) : Criterion(msg) {}

// NOLINTNEXTLINE(google-runtime-references)
void Random::pup(PUP::er& p) {
  Criterion::pup(p);
  p | do_something_fraction_;
  p | maximum_refinement_level_;
}

amr::Flag Random::random_flag(const size_t current_refinement_level) const {
  static std::random_device r;
  static const auto seed = r();
  static std::mt19937 generator(seed);
  static std::uniform_real_distribution<> distribution(0.0, 1.0);

  const double random_number = distribution(generator);
  if (random_number > do_something_fraction_) {
    return amr::Flag::DoNothing;
  }
  const double join_fraction =
      current_refinement_level / static_cast<double>(maximum_refinement_level_);
  if (random_number < join_fraction * do_something_fraction_) {
    return amr::Flag::Join;
  }
  return amr::Flag::Split;
}

PUP::able::PUP_ID Random::my_PUP_ID = 0;  // NOLINT
}  // namespace amr::Criteria
