// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>

#include "Utilities/MakeArray.hpp"

namespace domain {

/// \brief  The topology of a Block or Element in a particular dimension
///
/// \details The Topology is used to determine the geometry of the Block or
/// Element, which can be used to determine:
/// - Whether there is an interface (with a neighbor or external boundary) in a
/// given direction
/// - The block (element) logical coordinate bounds
/// - The appropriate Basis and Quadrature for a Mesh on an Element
/// - Whether or not h-refinement is allowed in the given dimension
/// - Whether or not the hybrid DG-Subcell scheme can be used
///
/// \note Choose I1 to represent an open interval \f$[-1, 1]\f$
///
/// \note Choose S1 to represent a periodic interval \f$[0, 2 \pi)\f$
///
/// \note In consecutive dimensions, choose S2Colatitude and S2Longitude to
/// represent the surface of a sphere
///
/// \note In consecutive dimensions, choose B2Radial and B2Angular to represent
/// a disk (including the center) or cross-section of a cylinder
///
/// \note In consecutive dimensions, choose B3Radial, B3Colatitude and
/// B3Longitude to represent a sphere
///
/// \note Currently h-refinement can only be done in dimensions with
/// Topology::I1
///
/// \note Currently the hybrid DG-Subcell scheme can be used only in Elements
/// for which all dimensions have Topology::I1
enum class Topology : uint8_t {
  Uninitialized = 0,
  I1 = 1,
  S1 = 2,
  S2Colatitude = 3,
  S2Longitude = 4,
  B1Radial = 5,
  B2Radial = 6,
  B2Angular = 7,
  B3Radial = 8,
  B3Colatitude = 9,
  B3Longitude = 10,
  CartoonSphere = 11,
  CartoonCylinder = 12
};

/// Output operator for a Topology.
std::ostream& operator<<(std::ostream& os, Topology topology);

namespace topologies {
template <size_t VolumeDim>
static constexpr auto hypercube = make_array<VolumeDim>(Topology::I1);

template <size_t VolumeDim>
static constexpr auto hypertorus = make_array<VolumeDim>(Topology::S1);

static constexpr auto annulus = std::array{Topology::I1, Topology::S1};

static constexpr auto disk =
    std::array{Topology::B2Radial, Topology::B2Angular};

static constexpr auto spherical_shell =
    std::array{Topology::I1, Topology::S2Colatitude, Topology::S2Longitude};

static constexpr auto cylindrical_shell =
    std::array{Topology::I1, Topology::S1, Topology::I1};

static constexpr auto full_cylinder =
    std::array{Topology::B2Radial, Topology::B2Angular, Topology::I1};

static constexpr auto full_sphere = std::array{
    Topology::B3Radial, Topology::B3Colatitude, Topology::B3Longitude};

static constexpr auto cartoon_sphere =
    std::array{Topology::I1, Topology::CartoonSphere, Topology::CartoonSphere};

static constexpr auto cartoon_sphere_inner = std::array{
    Topology::B1Radial, Topology::CartoonSphere, Topology::CartoonSphere};

static constexpr auto cartoon_cylinder =
    std::array{Topology::I1, Topology::I1, Topology::CartoonCylinder};

static constexpr auto cartoon_cylinder_inner =
    std::array{Topology::B1Radial, Topology::I1, Topology::CartoonCylinder};
}  // namespace topologies

}  // namespace domain
