// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

/// \cond
template <size_t Dim>
struct Block;
template <size_t Dim>
struct Element;
template <size_t Dim>
struct ElementId;
template <size_t Dim>
class Mesh;
namespace Spectral {
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace domain {
/// \ingroup InitializationGroup
/// \brief Construct the initial Mesh of an Element.
///
/// \param initial_extents initial extents for Elements in each Block of the
///        Domain
/// \param element Element
/// \param legendre_quadrature the quadrature rule/grid point distribution for
///        dimensions that use Spectral::Basis::Legendre
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Element<Dim>& element, Spectral::Quadrature legendre_quadrature);

/// \ingroup InitializationGroup
/// \brief Construct the initial Mesh of an Element from its Block and
/// ElementId.
///
/// \param initial_extents initial extents for Elements in each Block of the
///        Domain
/// \param block the Block of the Element
/// \param element_id the ElementId of the Element
/// \param legendre_quadrature the quadrature rule/grid point distribution for
///        dimensions that use Spectral::Basis::Legendre
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Block<Dim>& block, const ElementId<Dim>& element_id,
    Spectral::Quadrature legendre_quadrature);
}  // namespace domain
