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
enum class Basis : uint8_t;
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
/// \param i1_basis the Spectral::Basis used for dimensions with Topology::I1
/// \param i1_quadrature the Spectral::Quadrature for dimensions with
///        Topology::I1
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Element<Dim>& element, Spectral::Basis i1_basis,
    Spectral::Quadrature i1_quadrature);

/// \ingroup InitializationGroup
/// \brief Construct the initial Mesh of an Element from its Block and
/// ElementId.
///
/// \param initial_extents initial extents for Elements in each Block of the
///        Domain
/// \param block the Block of the Element
/// \param element_id the ElementId of the Element
/// \param i1_basis the Spectral::Basis used for dimensions with Topology::I1
/// \param i1_quadrature the Spectral::Quadrature for dimensions with
///        Topology::I1
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Block<Dim>& block, const ElementId<Dim>& element_id,
    Spectral::Basis i1_basis, Spectral::Quadrature i1_quadrature);
}  // namespace domain
