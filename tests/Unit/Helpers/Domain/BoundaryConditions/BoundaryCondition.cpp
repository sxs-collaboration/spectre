// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::domain::BoundaryConditions {
template <size_t Dim>
BoundaryConditionBase<Dim>::BoundaryConditionBase(CkMigrateMessage* const msg)
    : ::domain::BoundaryConditions::BoundaryCondition(msg) {}

template <size_t Dim>
void BoundaryConditionBase<Dim>::pup(PUP::er& p) {
  ::domain::BoundaryConditions::BoundaryCondition::pup(p);
}

template <size_t Dim>
TestBoundaryCondition<Dim>::TestBoundaryCondition(Direction<Dim> direction,
                                                  size_t block_id)
    : direction_(std::move(direction)), block_id_(block_id) {}

template <size_t Dim>
TestBoundaryCondition<Dim>::TestBoundaryCondition(const std::string& direction,
                                                  size_t block_id)
    : block_id_(block_id) {
  std::array<std::pair<std::string, Direction<Dim>>, 6> known_directions{};
  known_directions[0] = std::pair{"lower-xi", Direction<Dim>::lower_xi()};
  known_directions[1] = std::pair{"upper-xi", Direction<Dim>::upper_xi()};
  if constexpr (Dim > 1) {
    known_directions[2] = std::pair{"lower-eta", Direction<Dim>::lower_eta()};
    known_directions[3] = std::pair{"upper-eta", Direction<Dim>::upper_eta()};
  }
  if constexpr (Dim > 2) {
    known_directions[4] = std::pair{"lower-zeta", Direction<Dim>::lower_zeta()};
    known_directions[5] = std::pair{"upper-zeta", Direction<Dim>::upper_zeta()};
  }
  for (size_t i = 0; i < 2 * Dim; ++i) {
    if (direction == gsl::at(known_directions, i).first) {
      direction_ = gsl::at(known_directions, i).second;
      break;
    }
    if (i == 2 * Dim - 1) {
      std::string known_dirs{};
      for (const auto& dir : known_directions) {
        known_dirs += " " + dir.first;
      }
      ERROR("Unknown direction: " << direction
                                  << "\nKnown directions are: " << known_dirs);
    }
  }
}

template <size_t Dim>
TestBoundaryCondition<Dim>::TestBoundaryCondition(CkMigrateMessage* const msg)
    : BoundaryConditionBase<Dim>(msg) {}

template <size_t Dim>
auto TestBoundaryCondition<Dim>::get_clone() const
    -> std::unique_ptr<::domain::BoundaryConditions::BoundaryCondition> {
  return std::make_unique<TestBoundaryCondition<Dim>>(*this);
}

template <size_t Dim>
void TestBoundaryCondition<Dim>::pup(PUP::er& p) {
  BoundaryConditionBase<Dim>::pup(p);
  p | direction_;
  p | block_id_;
}

template <size_t Dim>
bool operator==(const TestBoundaryCondition<Dim>& lhs,
                const TestBoundaryCondition<Dim>& rhs) {
  return lhs.direction() == rhs.direction() and
         lhs.block_id() == rhs.block_id();
}

template <size_t Dim>
bool operator!=(const TestBoundaryCondition<Dim>& lhs,
                const TestBoundaryCondition<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
PUP::able::PUP_ID TestBoundaryCondition<Dim>::my_PUP_ID = 0;  // NOLINT

void register_derived_with_charm() {
  Parallel::register_derived_classes_with_charm<BoundaryConditionBase<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryConditionBase<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryConditionBase<3>>();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template class BoundaryConditionBase<DIM(data)>;                       \
  template class TestBoundaryCondition<DIM(data)>;                       \
  template bool operator==(const TestBoundaryCondition<DIM(data)>& lhs,  \
                           const TestBoundaryCondition<DIM(data)>& rhs); \
  template bool operator!=(const TestBoundaryCondition<DIM(data)>& lhs,  \
                           const TestBoundaryCondition<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef DIM
#undef INSTANTIATION
}  // namespace TestHelpers::domain::BoundaryConditions
