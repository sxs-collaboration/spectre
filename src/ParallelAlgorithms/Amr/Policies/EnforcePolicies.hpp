// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

/// \cond
namespace amr {
enum class Flag;
class Policies;
}  // namespace amr
template <size_t Dim>
class ElementId;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Mesh;
/// \endcond

namespace amr {
/// \brief Updates amr_decision so that it satisfies amr_policies
///
/// \details
/// - If amr_policies.isotropy() is Isotropic, the Flag%s are changed to all be
///   the same as the maximum priority flag
/// - If any Flag would cause the refinement levels or resolution to violate
///   their bounds, the Flag is changed to DoNothing.  Note that the minimum
///   resolution is Basis and Quadrature dependent, so may be higher than the
///   given amr_policies.refinement_limits().minimum_resolution()
template <size_t Dim>
void enforce_policies(gsl::not_null<std::array<Flag, Dim>*> amr_decision,
                      const amr::Policies& amr_policies,
                      const ElementId<Dim>& element_id, const Mesh<Dim>& mesh);
}  // namespace amr
