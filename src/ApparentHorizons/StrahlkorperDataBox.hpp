// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper`.
namespace StrahlkorperTags {

namespace StrahlkorperTags_detail {
// Shorter names for longer types used below.
template <typename Frame>
using ThetaPhi = tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>;
template <typename Frame>
using OneForm = tnsr::i<DataVector, 3, Frame>;
template <typename Frame>
using Vector = tnsr::I<DataVector, 3, Frame>;
template <typename Frame>
using Jacobian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<3, UpLo::Up, Frame>,
                      SpatialIndex<2, UpLo::Lo, ::Frame::Spherical<Frame>>>>;
template <typename Frame>
using InvJacobian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                      SpatialIndex<3, UpLo::Lo, Frame>>>;
template <typename Frame>
using InvHessian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 3, 2, 1>,
           index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                      SpatialIndex<3, UpLo::Lo, Frame>,
                      SpatialIndex<3, UpLo::Lo, Frame>>>;
template <typename Frame>
using SecondDeriv = tnsr::ii<DataVector, 3, Frame>;
}  // namespace StrahlkorperTags_detail

/// Tag referring to a `::Strahlkorper`
template <typename Frame>
struct Strahlkorper : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Strahlkorper";
  using type = ::Strahlkorper<Frame>;
};

/// \f$(\theta,\phi)\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct ThetaPhi : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ThetaPhi";
  static StrahlkorperTags_detail::ThetaPhi<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  using argument_tags = typelist<Strahlkorper<Frame>>;
};

/// \f$x_i/\sqrt{x^2+y^2+z^2}\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct Rhat : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Rhat";
  static StrahlkorperTags_detail::OneForm<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// `Jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
/// and `Jacobian(i,1)`
/// is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `Jacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct Jacobian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Jacobian";
  static StrahlkorperTags_detail::Jacobian<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// `InvJacobian(0,i)` is \f$r\partial\theta/\partial x^i\f$,
/// and `InvJacobian(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `InvJacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvJacobian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InvJacobian";
  static StrahlkorperTags_detail::InvJacobian<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// `InvHessian(k,i,j)` is \f$\partial (J^{-1}){}^k_j/\partial x^i\f$,
/// where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
/// `InvHessian` is not symmetric because the Jacobians are Pfaffian.
/// `InvHessian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvHessian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InvHessian";
  static StrahlkorperTags_detail::InvHessian<Frame> function(
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// (Euclidean) distance \f$r(\theta,\phi)\f$ from the center to each
/// point of the surface.
template <typename Frame>
struct Radius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Radius";
  SPECTRE_ALWAYS_INLINE static auto function(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept {
    return strahlkorper.ylm_spherepack().spec_to_phys(
        strahlkorper.coefficients());
  }
  using argument_tags = typelist<Strahlkorper<Frame>>;
};

/// \f$(x,y,z)\f$ of each point on the surface.
template <typename Frame>
struct CartesianCoords : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "CartesianCoords";
  static StrahlkorperTags_detail::Vector<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<Rhat<Frame>>& r_hat) noexcept;
  using argument_tags =
      typelist<Strahlkorper<Frame>, Radius<Frame>, Rhat<Frame>>;
};

/// `DxRadius(i)` is \f$\partial r/\partial x^i\f$.
/// Here \f$r=r(\theta,\phi)\f$ is the surface, which is considered a
/// function of Cartesian coordinates \f$r=r(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct DxRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "DxRadius";
  static StrahlkorperTags_detail::OneForm<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<InvJacobian<Frame>>& inv_jac) noexcept;
  using argument_tags =
      typelist<Strahlkorper<Frame>, Radius<Frame>, InvJacobian<Frame>>;
};

/// `D2xRadius(i,j)` is \f$\partial^2 r/\partial x^i\partial x^j\f$.
/// Here \f$r=r(\theta,\phi)\f$ is the surface, which is considered a
/// function of Cartesian coordinates \f$r=r(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct D2xRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "D2xRadius";
  static StrahlkorperTags_detail::SecondDeriv<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<InvJacobian<Frame>>& inv_jac,
      const db::item_type<InvHessian<Frame>>& inv_hess) noexcept;
  using argument_tags = typelist<Strahlkorper<Frame>, Radius<Frame>,
                                 InvJacobian<Frame>, InvHessian<Frame>>;
};

/// \f$\nabla^2 r\f$, the flat Laplacian of the surface.
/// This is \f$\eta^{ij}\partial^2 r/\partial x^i\partial x^j\f$,
/// where \f$r=r(\theta(x,y,z),\phi(x,y,z))\f$.
template <typename Frame>
struct LaplacianRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "LaplacianRadius";
  static DataVector function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept;
  using argument_tags =
      typelist<Strahlkorper<Frame>, Radius<Frame>, ThetaPhi<Frame>>;
};

/// Cartesian components of (unnormalized) one-form defining the surface.
/// This is computed by \f$x_i/r-\partial r/\partial x^i\f$,
/// where \f$x_i/r\f$ is `Rhat` and
/// \f$\partial r/\partial x^i\f$ is `DxRadius`.
/// See Eq. (8) of [arXiv:gr-qc/9606010] (https://arxiv.org/abs/gr-qc/9606010)
template <typename Frame>
struct NormalOneForm : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "NormalOneForm";
  static StrahlkorperTags_detail::OneForm<Frame> function(
      const db::item_type<DxRadius<Frame>>& dx_radius,
      const db::item_type<Rhat<Frame>>& r_hat) noexcept;
  using argument_tags = typelist<DxRadius<Frame>, Rhat<Frame>>;
};

/// `Tangents(j,i)` is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the surface
/// (i.e. `CartesianCoords`)
/// and are considered functions of \f$(\theta,\phi)\f$;
/// \f$\partial/\partial q^0\f$ means \f$\partial/\partial\theta\f$;
/// and \f$\partial/\partial q^1\f$ means
/// \f$\csc\theta\,\,\partial/\partial\phi\f$.
template <typename Frame>
struct Tangents : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Tangents";
  static StrahlkorperTags_detail::Jacobian<Frame> function(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const db::item_type<Rhat<Frame>>& r_hat,
      const db::item_type<Jacobian<Frame>>& jac) noexcept;
  using argument_tags = typelist<Strahlkorper<Frame>, Radius<Frame>,
                                 Rhat<Frame>, Jacobian<Frame>>;
};

template <typename Frame>
using items_tags = typelist<Strahlkorper<Frame>>;

template <typename Frame>
using compute_items_tags =
    typelist<ThetaPhi<Frame>, Rhat<Frame>, Jacobian<Frame>, InvJacobian<Frame>,
             InvHessian<Frame>, Radius<Frame>, CartesianCoords<Frame>,
             DxRadius<Frame>, D2xRadius<Frame>, LaplacianRadius<Frame>,
             NormalOneForm<Frame>, Tangents<Frame>>;

}  // namespace StrahlkorperTags
