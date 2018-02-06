// Distributed under the MIT License.
// See LICENSE.txt for details.

// Defines the class BulgedCube.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace CoordinateMaps {

/*!
 *  \ingroup CoordinateMapsGroup
 *
 *  \brief Three dimensional map from the cube to a bulged cube.
 *  The cube is shaped such that the surface is compatible
 *  with the inner surface of Wedge3D.
 *  The shape of the object can be chosen to be cubical,
 *  if the sphericity is set to 0, or to a sphere, if
 *  the sphericity is set to 1. The sphericity can
 *  be set to any number between 0 and 1 for a bulged cube.
 *
 *  \details The volume map from the cube to a bulged cube is obtained by
 *  interpolating between six surface maps, twelve bounding curves, and
 *  eight corners. The surface map for the upper +z axis is obtained by
 *  interpolating between a cubical surface and a spherical surface. The
 *  two surfaces are chosen such that the latter circumscribes the former.
 *
 *  We make a choice here as to whether we wish to use the logical coordinates
 *  parameterizing these surface as they are, in which case we have the
 *  equidistant choice of coordinates, or whether to apply a tangent map to them
 *  which leads us to the equiangular choice of coordinates. In terms of the
 *  logical coordinates, the equiangular coordinates are:
 *
 *  \f[\textrm{equiangular xi} : \Xi(\xi) = \textrm{tan}(\xi\pi/4)\f]
 *
 *  \f[\textrm{equiangular eta}  : \mathrm{H}(\eta) = \textrm{tan}(\eta\pi/4)\f]
 *
 *  With derivatives:
 *
 *  \f[\Xi'(\xi) = \frac{\pi}{4}(1+\Xi^2)\f]
 *
 *  \f[\mathrm{H}'(\eta) = \frac{\pi}{4}(1+\mathrm{H}^2)\f]
 *
 *  The equidistant coordinates are:
 *
 *  \f[ \textrm{equidistant xi}  : \Xi = \xi\f]
 *
 *  \f[ \textrm{equidistant eta}  : \mathrm{H} = \eta\f]
 *
 *  with derivatives:
 *
 *  <center>\f$\Xi'(\xi) = 1\f$, and \f$\mathrm{H}'(\eta) = 1\f$</center>
 *
 *  We also define the variable \f$\rho\f$, given by:
 *
 *  \f[\rho = \sqrt{1+\Xi^2+\mathrm{H}^2}\f]
 *
 *  ### The Spherical Face Map
 *  The surface map for the spherical face of radius \f$R\f$ lying in the
 *  \f$+z\f$
 *  direction in either choice of coordinates is then given by:
 *
 *  \f[\vec{\sigma}_{spherical}: \vec{\xi} \rightarrow \vec{x}(\vec{\xi})\f]
 *  Where
 *  \f[
 *  \vec{x}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  z(\xi,\eta)\\
 *  \end{bmatrix}  = \frac{R}{\rho}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}
 *  \f]
 *
 *  ### The Cubical Face Map
 *  The surface map for the cubical face of side length \f$2L\f$ lying in the
 *  \f$+z\f$ direction is given by:
 *
 *  \f[\vec{\sigma}_{cubical}: \vec{\xi} \rightarrow \vec{x}(\vec{\xi})\f]
 *  Where
 *  \f[
 *  \vec{x}(\xi,\eta) =
 *  \begin{bmatrix}
 *  x(\xi,\eta)\\
 *  y(\xi,\eta)\\
 *  L\\
 *  \end{bmatrix}  = L
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}
 *  \f]
 *
 *  ### The Bulged Face Map
 *  To construct the bulged map we interpolate between this cubical face map
 *  and a spherical face map of radius `radius_of_other_surface`, with the
 *  interpolation parameter being \f$s\f$, the `sphericity`.
 *  The surface map for the bulged face lying in the \f$+z\f$ direction is then
 *  given by:
 *
 *  \f[
 *  \vec{\sigma}_{+z}(\xi,\eta) = \left\{(1-s)L + \frac{sR}{\rho}\right\}
 *  \begin{bmatrix}
 *  \Xi\\
 *  \mathrm{H}\\
 *  1\\
 *  \end{bmatrix}
 *  \f]
 *
 *  We constrain L by demanding that the spherical face circumscribe the cube.
 *  With this condition, we have \f$L = R/\sqrt3\f$.
 *
 *  ### The General Formula for 3D Isoparametric Maps
 *  The general formula is given by Eq. 1 in section 2.1 of Hesthaven's paper
 *  "A Stable Penalty Method For The Compressible Navier-Stokes Equations III.
 *  Multidimensional Domain Decomposition Schemes" available
 *  <a href="
 *  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.699.1161&rep=rep1&type=pdf
 *  "> here </a>.
 *
 *  Hesthaven's formula is general in the degree of the shape functions used,
 *  so for our purposes we take the special case where the shape functions are
 *  linear, and define new variables accordingly.
 *
 *  We define the following variables for
 *  \f$\alpha, \beta, \gamma \in\{\xi,\eta,\zeta\}\f$:
 *
 *  \f[
 *  f^{\pm}_{\alpha} = \frac{1}{2}(1\pm\alpha)\\
 *  f^{\pm\pm}_{\alpha \ \beta} = \frac{1}{4}(1\pm\alpha)(1\pm\beta)\\
 *  f^{\pm\pm\pm}_{\alpha \ \beta \ \gamma} =
 *  \frac{1}{8}(1\pm\alpha)(1\pm\beta)(1\pm\gamma)
 *  \f]
 *
 *  The formula involves six surfaces, which we will denote by
 *  \f$\vec{\sigma}\f$, twelve curves, denoted by \f$\vec{\Gamma}\f$, and eight
 *  vertices, denoted by \f$\vec{\pi}\f$, with the subscripts denoting which
 *  face(s) these objects belong to. The full volume map is given by:
 *
 *  \f{align*}
 *  \phi(\xi,\eta,\zeta) = &
 *  f^{+}_{\zeta}\vec{\sigma}_{+\zeta}(\xi, \eta)+
 *  f^{-}_{\zeta}\vec{\sigma}_{-\zeta}(\xi, \eta)\\
 *  &+ f^{+}_{\eta}\vec{\sigma}_{+\eta}(\xi, \zeta)+
 *  f^{-}_{\eta}\vec{\sigma}_{-\eta}(\xi, \zeta)+
 *  f^{+}_{\xi}\vec{\sigma}_{+\xi}(\eta, \zeta)+
 *  f^{-}_{\xi}\vec{\sigma}_{-\xi}(\eta, \zeta)\\
 *  &- f^{++}_{\xi \ \eta}\vec{\Gamma}_{+\xi +\eta}(\zeta)-
 *  f^{-+}_{\xi \ \eta}\vec{\Gamma}_{-\xi +\eta}(\zeta)-
 *  f^{+-}_{\xi \ \eta}\vec{\Gamma}_{+\xi -\eta}(\zeta)-
 *  f^{--}_{\xi \ \eta}\vec{\Gamma}_{-\xi -\eta}(\zeta)\\
 *  &- f^{++}_{\xi \ \zeta}\vec{\Gamma}_{+\xi +\zeta}(\eta)-
 *  f^{-+}_{\xi \ \zeta}\vec{\Gamma}_{-\xi +\zeta}(\eta)-
 *  f^{+-}_{\xi \ \zeta}\vec{\Gamma}_{+\xi -\zeta}(\eta)-
 *  f^{--}_{\xi \ \zeta}\vec{\Gamma}_{-\xi -\zeta}(\eta)\\
 *  &- f^{++}_{\eta \ \zeta}\vec{\Gamma}_{+\eta +\zeta}(\xi)-
 *  f^{-+}_{\eta \ \zeta}\vec{\Gamma}_{-\eta +\zeta}(\xi)-
 *  f^{+-}_{\eta \ \zeta}\vec{\Gamma}_{+\eta -\zeta}(\xi)-
 *  f^{--}_{\eta \zeta}\vec{\Gamma}_{-\eta -\zeta}(\xi)\\
 *  &+ f^{+++}_{\xi \ \eta \ \zeta}\vec{\pi}_{+\xi +\eta +\zeta}+
 *  f^{-++}_{\xi \ \eta \ \zeta}\vec{\pi}_{-\xi +\eta +\zeta}+
 *  f^{+-+}_{\xi \ \eta \ \zeta}\vec{\pi}_{+\xi -\eta +\zeta}+
 *  f^{--+}_{\xi \ \eta \ \zeta}\vec{\pi}_{-\xi -\eta +\zeta}\\
 *  &+ f^{++-}_{\xi \ \eta \ \zeta}\vec{\pi}_{+\xi +\eta -\zeta}+
 *  f^{-+-}_{\xi \ \eta \ \zeta}\vec{\pi}_{-\xi +\eta -\zeta}+
 *  f^{+--}_{\xi \ \eta \ \zeta}\vec{\pi}_{+\xi -\eta -\zeta}+
 *  f^{---}_{\xi \ \eta \ \zeta}\vec{\pi}_{-\xi -\eta -\zeta}
 *  \f}
 *
 *
 *  ### The Special Case for Octahedral Symmetry
 *  The general formula is for the case in which there are six independently
 *  specified bounding surfaces. In our case, the surfaces are obtained by
 *  rotations and reflections of the upper-\f$\zeta\f$ face.
 *
 * We define the matrices corresponding to these transformations to be:
 *
 * \f[
 * S_{xy} =
 *  \begin{bmatrix}
 *  0 & 1 & 0\\
 *  1 & 0 & 0\\
 *  0 & 0 & 1\\
 *  \end{bmatrix},\
 *
 * S_{xz} =
 *  \begin{bmatrix}
 *  0 & 0 & 1\\
 *  0 & 1 & 0\\
 *  1 & 0 & 0\\
 *  \end{bmatrix},\
 *
 * S_{yz} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 0 & 1\\
 *  0 & 1 & 0\\
 *  \end{bmatrix}\f]
 *
 * \f[C_{zxy} =
 *  \begin{bmatrix}
 *  0 & 0 & 1\\
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 * C_{yzx} =
 *  \begin{bmatrix}
 *  0 & 1 & 0\\
 *  0 & 0 & 1\\
 *  1 & 0 & 0\\
 *  \end{bmatrix}\f]
 *
 * \f[N_{x} =
 *  \begin{bmatrix}
 *  -1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & 1\\
 *  \end{bmatrix},\
 *
 * N_{y} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & -1 & 0\\
 *  0 & 0 & 1\\
 *  \end{bmatrix},\
 *
 * N_{z} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & -1\\
 *  \end{bmatrix}
 *  \f]
 *
 * The surface maps can now all be written in terms of
 * \f$\vec{\sigma}_{+\zeta}\f$ and these matrices:
 * <center>
 * \f$\vec{\sigma}_{-\zeta}(\xi, \eta) = N_z\vec{\sigma}_{+\zeta}(\xi, \eta)\\
 * \vec{\sigma}_{+\eta}(\xi, \zeta) = S_{yz}\vec{\sigma}_{+\zeta}(\xi, \zeta)\\
 * \vec{\sigma}_{-\eta}(\xi, \zeta) = N_yS_{yz}\vec{\sigma}_{+\zeta}(\xi,
 * \zeta)\\
 * \vec{\sigma}_{+\xi}(\eta, \zeta) = C_{zxy}\vec{\sigma}_{+\zeta}(\eta,
 * \zeta)\\
 * \vec{\sigma}_{-\xi}(\eta, \zeta) = N_xC_{zyx}\vec{\sigma}_{+\zeta}(\eta,
 * \zeta)\f$
 * </center>
 *
 * The four bounding curves \f$\vec{\Gamma}\f$ on the \f$+\zeta\f$ face are
 * given by:
 *
 * <center>
 * \f$\vec{\Gamma}_{+\xi,+\zeta}(\eta) = \vec{\sigma}_{+\zeta}(+1,\eta)\\
 * \vec{\Gamma}_{-\xi,+\zeta}(\eta) = \vec{\sigma}_{+\zeta}(-1,\eta)
 * = N_x\vec{\sigma}_{+\zeta}(+1, \eta)\\
 * \vec{\Gamma}_{+\eta,+\zeta}(\xi) = \vec{\sigma}_{+\zeta}(\xi,+1)
 * = S_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\\
 * \vec{\Gamma}_{-\eta,+\zeta}(\xi) = \vec{\sigma}_{+\zeta}(\xi,-1)
 * = N_yS_{xy}\vec{\sigma}_{+\zeta}(+1,\xi)\f$
 * </center>
 *
 * The bounding curves on the other surfaces can be obtained by transformations
 * on the \f$+\zeta\f$ face:
 *
 * <center>
 * \f$\vec{\Gamma}_{+\xi,-\zeta}(\eta) = N_z\vec{\sigma}_{+\zeta}(+1,\eta)\\
 * \vec{\Gamma}_{-\xi,-\zeta}(\eta) = N_z\vec{\sigma}_{+\zeta}(-1,\eta)
 * = N_zN_x\vec{\sigma}_{+\zeta}(+1,\eta)\\
 * \vec{\Gamma}_{+\eta,-\zeta}(\xi) = N_z\vec{\sigma}_{+\zeta}(\xi,+1)
 * = N_zS_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\\
 * \vec{\Gamma}_{-\eta,-\zeta}(\xi) = N_z\vec{\sigma}_{+\zeta}(\xi,-1)
 * = N_zN_yS_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\\
 * \vec{\Gamma}_{+\xi,+\eta}(\zeta) =
 * C_{zxy}\vec{\sigma}_{+\zeta}(+1,\zeta)\\
 * \vec{\Gamma}_{-\xi,+\eta}(\zeta) =
 * N_xC_{zxy}\vec{\sigma}_{+\zeta}(+1,\zeta)\\
 * \vec{\Gamma}_{+\xi,-\eta}(\zeta) = C_{zxy}\vec{\sigma}_{+\zeta}(-1,\zeta)
 * = C_{zxy}N_x\vec{\sigma}_{+\zeta}(+1,\zeta)\\
 * \vec{\Gamma}_{-\xi,-\eta}(\zeta) = N_xC_{zxy}\vec{\sigma}_{+\zeta}(-1,\zeta)
 * = N_xC_{zxy}N_x\vec{\sigma}_{+\zeta}(+1,\zeta)\f$
 * </center>
 *
 * Now we can write the volume map \f$\phi\f$ in terms of
 * \f$\vec{\sigma}_{+\zeta}\f$ only:
 * \f{align*}\phi(\xi,\eta,\zeta) = &
 * (f^{+}_{\zeta} + f^{-}_{\zeta}N_z)
 * \vec{\sigma}_{+\zeta}(\xi, \eta)\\
 * &+ (f^{+}_{\eta} + f^{-}_{\eta}N_y)
 * S_{yz}\vec{\sigma}_{+\zeta}(\xi, \zeta)\\
 * &+ (f^{+}_{\xi} + f^{-}_{\xi}N_x)
 * C_{zxy}\vec{\sigma}_{+\zeta}(\eta, \zeta)\\
 * &- (f^{+}_{\xi}+f^{-}_{\xi}N_x)
 * (f^{+}_{\eta}+f^{-}_{\eta}N_y)
 * C_{zxy}\vec{\sigma}_{+\zeta}(+1, \zeta)\\
 * &- (f^{+}_{\zeta}+f^{-}_{\zeta}N_z)\left\{
 * (f^{+}_{\xi}+f^{-}_{\xi}N_x)\vec{\sigma}_{+\zeta}(+1, \eta)+
 * (f^{+}_{\eta}+f^{-}_{\eta}N_y)S_{xy}\vec{\sigma}_{+\zeta}(+1, \xi)\right\}\\
 * &+ \frac{r}{\sqrt{3}}\vec{\xi}
 *  \f}
 *
 * Note that we can now absorb all of the \f$f\f$s into the matrix prefactors
 * in the above equation and obtain a final set of matrices. We define the
 * following *blending matrices*:
 *
 *  \f[
 *  B_{\xi} =
 *  \begin{bmatrix}
 *  0 & 0 & \xi\\
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 *  B_{\eta} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 0 & \eta\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 *  B_{\zeta} =
 *  \begin{bmatrix}
 *  1 & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & \zeta\\
 *  \end{bmatrix}\\
 *
 *  B_{\xi\eta} =
 *  \begin{bmatrix}
 *  0 & 0 & \xi\\
 *  \eta & 0 & 0\\
 *  0 & 1 & 0\\
 *  \end{bmatrix},\
 *
 *  B_{\xi\zeta} =
 *  \begin{bmatrix}
 *  \xi & 0 & 0\\
 *  0 & 1 & 0\\
 *  0 & 0 & \zeta\\
 *  \end{bmatrix},\
 *
 *  B_{\eta\zeta} =
 *  \begin{bmatrix}
 *  0 & 1 & 0\\
 *  \eta & 0 & 0\\
 *  0 & 0 & \zeta\\
 *  \end{bmatrix}\\
 *
 *  B_{\xi\eta\zeta} =
 *  \begin{bmatrix}
 *  \xi & 0 & 0\\
 *  0 & \eta & 0\\
 *  0 & 0 & \zeta\\
 *  \end{bmatrix}
 *  \f]
 *
 *  Now we can write the volume map \f$\phi\f$ in these terms:
 *
 * \f{align*}
 * \phi(\xi,\eta,\zeta) = &
 * B_{\zeta}
 * \vec{\sigma}_{+\zeta}(\xi, \eta)\\& +
 * B_{\eta}
 * \vec{\sigma}_{+\zeta}(\xi, \zeta)+
 * B_{\xi}
 * \vec{\sigma}_{+\zeta}(\eta, \zeta)\\& -
 * B_{\xi \eta}
 * \vec{\sigma}_{+\zeta}(+1, \zeta)-
 * B_{\xi \zeta}
 * \vec{\sigma}_{+\zeta}(+1, \eta)+
 * B_{\eta \zeta}
 * \vec{\sigma}_{+\zeta}(+1, \xi)\\& +
 * B_{\xi \eta \zeta}
 * \vec{\sigma}_{+\zeta}(+1, +1)
 * \f}
 *
 * ### The Bulged Cube Map
 * We now use the result above to provide the mapping for the bulged cube.
 * First we will define the variables \f$\rho_A\f$ and \f$\rho_{AB}\f$, for
 * \f$A, B \in \{\Xi,\mathrm{H}, \mathrm{Z}\} \f$:
 * \f[
 * \rho_A = \sqrt{2 + A^2}\\
 * \rho_{AB} = \sqrt{1 + A^2 + B^2}
 * \f]
 * The final mapping is then:
 * \f[
 * \vec{x}(\xi,\eta,\zeta) = \frac{(1-s)r}{\sqrt{3}}
 * \begin{bmatrix}
 * \Xi\\
 * \mathrm{H}\\
 * \mathrm{Z}\\
 * \end{bmatrix} +
 * \frac{sr}{\sqrt{3}}
 * \begin{bmatrix}
 * \xi\\
 * \eta\\
 * \zeta\\
 * \end{bmatrix} + sr
 * \begin{bmatrix}
 * \xi & \Xi & \Xi\\
 * \mathrm{H} & \eta &\mathrm{H}\\
 * \mathrm{Z} & \mathrm{Z} & \zeta\\
 * \end{bmatrix}
 * \begin{bmatrix}
 * 1/\rho_{\mathrm{H}\mathrm{Z}}\\
 * 1/\rho_{\Xi\mathrm{Z}}\\
 * 1/\rho_{\Xi\mathrm{H}}\\
 * \end{bmatrix} - sr
 * \begin{bmatrix}
 * \Xi & \xi & \xi\\
 * \eta & \mathrm{H} &\eta\\
 * \zeta & \zeta & \mathrm{Z}\\
 * \end{bmatrix}
 * \begin{bmatrix}
 * 1/\rho_{\Xi}\\
 * 1/\rho_{\mathrm{H}}\\
 * 1/\rho_{\mathrm{Z}}\\
 * \end{bmatrix}
 * \f]
 *
 * In the case where the same coordinates are used for the cube and the sphere,
 * we have \f$\xi = \Xi\f$, etc. In this case, the formula reduces further.
 * It is given by:
 *
 * \f[
 * \vec{x}(\xi,\eta,\zeta) =
 * \left\{
 * \frac{r}{\sqrt{3}}
 * + sr
 * \left(
 * 1/\rho_{\mathrm{H}\mathrm{Z}}+
 * 1/\rho_{\Xi\mathrm{Z}}+
 * 1/\rho_{\Xi\mathrm{H}}-
 * 1/\rho_{\Xi}-
 * 1/\rho_{\mathrm{H}}-
 * 1/\rho_{\mathrm{Z}}
 * \right)
 * \right\}
 * \begin{bmatrix}
 * \Xi\\
 * \mathrm{H}\\
 * \mathrm{Z}\\
 * \end{bmatrix}
 * \f]
 *
 */
class BulgedCube {
 public:
  static constexpr size_t dim = 3;
  BulgedCube(double radius, double sphericity,
             bool use_equiangular_map) noexcept;
  BulgedCube() noexcept = default;
  ~BulgedCube() noexcept = default;
  BulgedCube(BulgedCube&&) noexcept = default;
  BulgedCube(const BulgedCube&) noexcept = default;
  BulgedCube& operator=(const BulgedCube&) noexcept = default;
  BulgedCube& operator=(BulgedCube&&) noexcept = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> inverse(
      const std::array<T, 3>& target_coords) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> xi_derivative(
      const std::array<T, 3>& source_coords) const noexcept;
  friend bool operator==(const BulgedCube& lhs, const BulgedCube& rhs) noexcept;
  double radius_;
  double sphericity_;
  bool use_equiangular_map_;
};

bool operator!=(const BulgedCube& lhs, const BulgedCube& rhs) noexcept;
}  // namespace CoordinateMaps
