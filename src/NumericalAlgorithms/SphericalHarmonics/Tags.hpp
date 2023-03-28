// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsDeclarations.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsTypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper`.
namespace StrahlkorperTags {

/// Tag referring to a `::Strahlkorper`
template <typename Frame>
struct Strahlkorper : db::SimpleTag {
  using type = ::Strahlkorper<Frame>;
};

/// @{
/// \f$(\theta,\phi)\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct ThetaPhi : db::SimpleTag {
  using type = tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>;
};

template <typename Frame>
struct ThetaPhiCompute : ThetaPhi<Frame>, db::ComputeTag {
  using base = ThetaPhi<Frame>;
  using return_type = tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>*>,
      const ::Strahlkorper<Frame>&)>(
      &::StrahlkorperFunctions::theta_phi<Frame>);
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};
/// @}

/// @{
/// `Rhat(i)` is \f$\hat{r}^i = x_i/\sqrt{x^2+y^2+z^2}\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct Rhat : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame>;
};

template <typename Frame>
struct RhatCompute : Rhat<Frame>, db::ComputeTag {
  using base = Rhat<Frame>;
  using return_type = tnsr::i<DataVector, 3, Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::i<DataVector, 3, Frame>*>,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&)>(
      &::StrahlkorperFunctions::rhat<Frame>);
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `Jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
/// and `Jacobian(i,1)`
/// is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `Jacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct Jacobian : db::SimpleTag {
  using type = aliases::Jacobian<Frame>;
};

template <typename Frame>
struct JacobianCompute : Jacobian<Frame>, db::ComputeTag {
  using base = Jacobian<Frame>;
  using return_type = aliases::Jacobian<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<StrahlkorperTags::aliases::Jacobian<Frame>*>,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&)>(
      &::StrahlkorperFunctions::jacobian<Frame>);
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `InvJacobian(0,i)` is \f$r\partial\theta/\partial x^i\f$,
/// and `InvJacobian(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `InvJacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvJacobian : db::SimpleTag {
  using type = aliases::InvJacobian<Frame>;
};

template <typename Frame>
struct InvJacobianCompute : InvJacobian<Frame>, db::ComputeTag {
  using base = InvJacobian<Frame>;
  using return_type = aliases::InvJacobian<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<StrahlkorperTags::aliases::InvJacobian<Frame>*>,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&)>(
      &::StrahlkorperFunctions::inv_jacobian<Frame>);
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `InvHessian(k,i,j)` is \f$\partial (J^{-1}){}^k_j/\partial x^i\f$,
/// where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
/// `InvHessian` is not symmetric because the Jacobians are Pfaffian.
/// `InvHessian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvHessian : db::SimpleTag {
  using type = aliases::InvHessian<Frame>;
};

template <typename Frame>
struct InvHessianCompute : InvHessian<Frame>, db::ComputeTag {
  using base = InvHessian<Frame>;
  using return_type = aliases::InvHessian<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<StrahlkorperTags::aliases::InvHessian<Frame>*>,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&)>(
      &::StrahlkorperFunctions::inv_hessian<Frame>);
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the center to each
/// point of the surface.
template <typename Frame>
struct Radius : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct RadiusCompute : Radius<Frame>, db::ComputeTag {
  using base = Radius<Frame>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataVector>*>, const ::Strahlkorper<Frame>&)>(
      &(::StrahlkorperFunctions::radius<Frame>));
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};
/// @}

/// @{
/// The geometrical center of the surface.  Uses
/// `Strahlkorper::physical_center`.
template <typename Frame>
struct PhysicalCenter : db::SimpleTag {
  using type = std::array<double, 3>;
};

template <typename Frame>
struct PhysicalCenterCompute : PhysicalCenter<Frame>, db::ComputeTag {
  using base = PhysicalCenter<Frame>;
  using return_type = std::array<double, 3>;
  static void function(gsl::not_null<std::array<double, 3>*> physical_center,
                       const ::Strahlkorper<Frame>& strahlkorper);
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};
/// @}

/// @{
/// `CartesianCoords(i)` is \f$x_{\rm surf}^i\f$,
/// the vector of \f$(x,y,z)\f$ coordinates of each point
/// on the surface.
template <typename Frame>
struct CartesianCoords : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame>;
};

template <typename Frame>
struct CartesianCoordsCompute : CartesianCoords<Frame>, db::ComputeTag {
  using base = CartesianCoords<Frame>;
  using return_type = tnsr::I<DataVector, 3, Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::I<DataVector, 3, Frame>*> coords,
      const ::Strahlkorper<Frame>& strahlkorper,
      const Scalar<DataVector>& radius,
      const tnsr::i<DataVector, 3, Frame>& r_hat)>(
      &StrahlkorperFunctions::cartesian_coords);
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, Rhat<Frame>>;
};
/// @}

/// @{
/// `DxRadius(i)` is \f$\partial r_{\rm surf}/\partial x^i\f$.  Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct DxRadius : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame>;
};

template <typename Frame>
struct DxRadiusCompute : DxRadius<Frame>, db::ComputeTag {
  using base = DxRadius<Frame>;
  using return_type = tnsr::i<DataVector, 3, Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> dx_radius,
      const Scalar<DataVector>& scalar,
      const ::Strahlkorper<Frame>& strahlkorper,
      const Scalar<DataVector>& radius_of_strahlkorper,
      const aliases::InvJacobian<Frame>& inv_jac)>(
      &StrahlkorperFunctions::cartesian_derivs_of_scalar);
  using argument_tags = tmpl::list<Radius<Frame>, Strahlkorper<Frame>,
                                   Radius<Frame>, InvJacobian<Frame>>;
};
/// @}

/// @{
/// `D2xRadius(i,j)` is
/// \f$\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$. Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct D2xRadius : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3, Frame>;
};

template <typename Frame>
struct D2xRadiusCompute : D2xRadius<Frame>, db::ComputeTag {
  using base = D2xRadius<Frame>;
  using return_type = tnsr::ii<DataVector, 3, Frame>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> d2x_radius,
      const Scalar<DataVector>& scalar,
      const ::Strahlkorper<Frame>& strahlkorper,
      const Scalar<DataVector>& radius_of_strahlkorper,
      const aliases::InvJacobian<Frame>& inv_jac,
      const aliases::InvHessian<Frame>& inv_hess)>(
      &StrahlkorperFunctions::cartesian_second_derivs_of_scalar);
  using argument_tags =
      tmpl::list<Radius<Frame>, Strahlkorper<Frame>, Radius<Frame>,
                 InvJacobian<Frame>, InvHessian<Frame>>;
};
/// @}

/// @{
/// \f$\nabla^2 r_{\rm surf}\f$, the flat Laplacian of the surface.
/// This is \f$\eta^{ij}\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$,
/// where \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$.
template <typename Frame>
struct LaplacianRadius : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct LaplacianRadiusCompute : LaplacianRadius<Frame>, db::ComputeTag {
  using base = LaplacianRadius<Frame>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*> lap_radius,
      const Scalar<DataVector>& radius,
      const ::Strahlkorper<Frame>& strahlkorper,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>& theta_phi)>(
      &StrahlkorperFunctions::laplacian_of_scalar);
  using argument_tags =
      tmpl::list<Radius<Frame>, Strahlkorper<Frame>, ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `NormalOneForm(i)` is \f$s_i\f$, the (unnormalized) normal one-form
/// to the surface, expressed in Cartesian components.
/// This is computed by \f$x_i/r-\partial r_{\rm surf}/\partial x^i\f$,
/// where \f$x_i/r\f$ is `Rhat` and
/// \f$\partial r_{\rm surf}/\partial x^i\f$ is `DxRadius`.
/// See Eq. (8) of \cite Baumgarte1996hh.
/// Note on the word "normal": \f$s_i\f$ points in the correct direction
/// (it is "normal" to the surface), but it does not have unit length
/// (it is not "normalized"; normalization requires a metric).
template <typename Frame>
struct NormalOneForm : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame>;
};

template <typename Frame>
struct NormalOneFormCompute : NormalOneForm<Frame>, db::ComputeTag {
  using base = NormalOneForm<Frame>;
  using return_type = tnsr::i<DataVector, 3, Frame>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::i<DataVector, 3, Frame>*> one_form,
      const tnsr::i<DataVector, 3, Frame>& dx_radius,
      const tnsr::i<DataVector, 3, Frame>& r_hat)>(
      &StrahlkorperFunctions::normal_one_form);
  using argument_tags = tmpl::list<DxRadius<Frame>, Rhat<Frame>>;
};
/// @}

/// @{
/// `Tangents(i,j)` is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the
/// surface (i.e. `CartesianCoords`) and are considered functions of
/// \f$(\theta,\phi)\f$.
///
/// \f$\partial/\partial q^0\f$ means
/// \f$\partial/\partial\theta\f$; and \f$\partial/\partial q^1\f$
/// means \f$\csc\theta\,\,\partial/\partial\phi\f$.  Note that the
/// vectors `Tangents(i,0)` and `Tangents(i,1)` are orthogonal to the
/// `NormalOneForm` \f$s_i\f$, i.e.
/// \f$s_i \partial x_{\rm surf}^i/\partial q^j = 0\f$; this statement
/// is independent of a metric.  Also, `Tangents(i,0)` and
/// `Tangents(i,1)` are not necessarily orthogonal to each other,
/// since orthogonality between 2 vectors (as opposed to a vector and
/// a one-form) is metric-dependent.
template <typename Frame>
struct Tangents : db::SimpleTag {
  using type = aliases::Jacobian<Frame>;
};

template <typename Frame>
struct TangentsCompute : Tangents<Frame>, db::ComputeTag {
  using base = Tangents<Frame>;
  using return_type = aliases::Jacobian<Frame>;
  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<aliases::Jacobian<Frame>*> tangents,
                           const ::Strahlkorper<Frame>& strahlkorper,
                           const Scalar<DataVector>& radius,
                           const tnsr::i<DataVector, 3, Frame>& r_hat,
                           const aliases::Jacobian<Frame>& jac)>(
          &StrahlkorperFunctions::tangents);
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   Rhat<Frame>, Jacobian<Frame>>;
};
/// @}

template <typename Frame>
using items_tags = tmpl::list<Strahlkorper<Frame>>;

template <typename Frame>
using compute_items_tags =
    tmpl::list<ThetaPhiCompute<Frame>, RhatCompute<Frame>,
               JacobianCompute<Frame>, InvJacobianCompute<Frame>,
               InvHessianCompute<Frame>, RadiusCompute<Frame>,
               CartesianCoordsCompute<Frame>, DxRadiusCompute<Frame>,
               D2xRadiusCompute<Frame>, LaplacianRadiusCompute<Frame>,
               NormalOneFormCompute<Frame>, TangentsCompute<Frame>>;
}  // namespace StrahlkorperTags

/// Tags related to symmetric trace-free tensors
namespace Stf::Tags {

/*!
 * \brief Tag used to hold a symmetric trace-free tensor of a certain rank.
 * \details The type is a symmetric tensor of the requested rank. A
 * ScalarBaseTag of type `Scalar<double>` is used to identify the tag of which
 * the symmetric trace-free expansion is done.
 */
template <typename ScalarBaseTag, size_t rank, size_t Dim, typename Frame>
struct StfTensor : db::SimpleTag {
  static_assert(std::is_same_v<typename ScalarBaseTag::type, Scalar<double>>,
                "StfTensor base tags must be a Scalar.");
  static_assert(rank <= 3, "StfTensor tag is only implemented up to rank 3.");
  static std::string name() {
    return MakeString{} << "StfTensor(" << db::tag_name<ScalarBaseTag>()
                        << "," << rank << ")";
  }
  using type_list =
      tmpl::list<Scalar<double>, tnsr::i<double, Dim, Frame>,
                 tnsr::ii<double, Dim, Frame>, tnsr::iii<double, Dim, Frame>>;
  using type = tmpl::at<type_list, tmpl::size_t<rank>>;
};
}  // namespace Stf::Tags
