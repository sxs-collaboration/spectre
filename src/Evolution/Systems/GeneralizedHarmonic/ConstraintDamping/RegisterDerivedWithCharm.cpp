// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace GeneralizedHarmonic::ConstraintDamping {
namespace {
template <size_t Dim, typename Fr>
void register_damping_functions_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<DampingFunction<Dim, Fr>>();
}
}  // namespace

void register_derived_with_charm() noexcept {
  register_damping_functions_with_charm<1, Frame::Grid>();
  register_damping_functions_with_charm<2, Frame::Grid>();
  register_damping_functions_with_charm<3, Frame::Grid>();
  register_damping_functions_with_charm<1, Frame::Inertial>();
  register_damping_functions_with_charm<2, Frame::Inertial>();
  register_damping_functions_with_charm<3, Frame::Inertial>();
}
}  // namespace GeneralizedHarmonic::ConstraintDamping
