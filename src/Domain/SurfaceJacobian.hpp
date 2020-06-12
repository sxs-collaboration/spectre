// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
struct Mesh;
template <size_t Dim>
struct Direction;
template <size_t Dim, typename TargetFrame>
struct ElementMap;
/// \endcond

namespace domain {

/*!
 * \brief The surface Jacobian \f$S=J\sqrt{ \frac{\partial\xi_j}{\boldsymbol{x}}
 * \cdot \frac{\partial\xi_j}{\boldsymbol{x}}}\f$
 *
 * Since the unnormalized face normal is
 *
 * \begin{equation} \boldsymbol{n} = \sign{\xi_j}\frac{\partial
 * \xi_j}{\boldsymbol{x}} \end{equation}
 *
 * the surface Jacobian can also be written as
 *
 * \begin{equation} S = J |\boldsymbol{n}| \end{equation}
 */
void surface_jacobian(gsl::not_null<Scalar<DataVector>*> surface_jacobian,
                      const Scalar<DataVector>& det_jacobian_on_face,
                      const Scalar<DataVector>& face_normal_magnitude) noexcept;

Scalar<DataVector> surface_jacobian(
    const Scalar<DataVector>& det_jacobian_on_face,
    const Scalar<DataVector>& face_normal_magnitude) noexcept;

template <size_t Dim, typename TargetFrame>
void surface_jacobian(gsl::not_null<Scalar<DataVector>*> surface_jacobian,
                      const ElementMap<Dim, TargetFrame>& element_map,
                      const Mesh<Dim - 1>& face_mesh,
                      const Direction<Dim>& direction,
                      const Scalar<DataVector>& face_normal_magnitude) noexcept;

template <size_t Dim, typename TargetFrame>
Scalar<DataVector> surface_jacobian(
    const ElementMap<Dim, TargetFrame>& element_map,
    const Mesh<Dim - 1>& face_mesh, const Direction<Dim>& direction,
    const Scalar<DataVector>& face_normal_magnitude) noexcept;

namespace Tags {

template <typename SourceFrame, typename TargetFrame>
struct SurfaceJacobian : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename MapTag>
struct SurfaceJacobianCompute : SurfaceJacobian<typename MapTag::source_frame,
                                                typename MapTag::target_frame>,
                                db::ComputeTag {
  static constexpr size_t dim = MapTag::dim;
  using source_frame = typename MapTag::source_frame;
  using target_frame = typename MapTag::target_frame;
  using base = SurfaceJacobian<source_frame, target_frame>;
  using return_type = tmpl::type_from<base>;
  using argument_tags =
      tmpl::list<ElementMap<dim, target_frame>, Mesh<dim - 1>, Direction<dim>,
                 ::Tags::Magnitude<UnnormalizedFaceNormal<dim, target_frame>>>;
  using volume_tags = tmpl::list<ElementMap<dim, target_frame>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>,
      const ::ElementMap<dim, target_frame>&, const ::Mesh<dim - 1>&,
      const ::Direction<dim>&, const Scalar<DataVector>&) noexcept>(
      &domain::surface_jacobian<dim, target_frame>);
};

}  // namespace Tags
}  // namespace domain
