// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare gr::Tags::SpatialMetric
/// \cond
class DataVector;
class FastFlow;
/// \endcond

namespace ah {
namespace Tags {
struct FastFlow : db::SimpleTag {
  using type = ::FastFlow;
};
}  // namespace Tags
}  // namespace ah

/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper`.
namespace StrahlkorperTags {

/// Tag referring to a `::Strahlkorper`
template <typename Frame>
struct Strahlkorper : db::SimpleTag {
  using type = ::Strahlkorper<Frame>;
};

/// \f$(\theta,\phi)\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct ThetaPhi : db::ComputeTag {
  static std::string name() noexcept { return "ThetaPhi"; }
  using return_type = aliases::ThetaPhi<Frame>;
  static void function(gsl::not_null<aliases::ThetaPhi<Frame>*> theta_phi,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};

/// `Rhat(i)` is \f$\hat{r}^i = x_i/\sqrt{x^2+y^2+z^2}\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct Rhat : db::ComputeTag {
  static std::string name() noexcept { return "Rhat"; }
  using return_type = aliases::OneForm<Frame>;
  static void function(gsl::not_null<aliases::OneForm<Frame>*> r_hat,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// `Jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
/// and `Jacobian(i,1)`
/// is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `Jacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct Jacobian : db::ComputeTag {
  static std::string name() noexcept { return "Jacobian"; }
  using return_type = aliases::Jacobian<Frame>;
  static void function(gsl::not_null<aliases::Jacobian<Frame>*> jac,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// `InvJacobian(0,i)` is \f$r\partial\theta/\partial x^i\f$,
/// and `InvJacobian(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `InvJacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvJacobian : db::ComputeTag {
  static std::string name() noexcept { return "InvJacobian"; }
  using return_type = aliases::InvJacobian<Frame>;
  static void function(gsl::not_null<aliases::InvJacobian<Frame>*> inv_jac,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// `InvHessian(k,i,j)` is \f$\partial (J^{-1}){}^k_j/\partial x^i\f$,
/// where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
/// `InvHessian` is not symmetric because the Jacobians are Pfaffian.
/// `InvHessian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvHessian : db::ComputeTag {
  static std::string name() noexcept { return "InvHessian"; }
  using return_type = aliases::InvHessian<Frame>;
  static void function(gsl::not_null<aliases::InvHessian<Frame>*> inv_hess,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};

/// (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the center to each
/// point of the surface.
template <typename Frame>
struct Radius : db::ComputeTag {
  static std::string name() noexcept { return "Radius"; }
  using return_type = DataVector;
  static void function(gsl::not_null<DataVector*> radius,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};

/// `CartesianCoords(i)` is \f$x_{\rm surf}^i\f$,
/// the vector of \f$(x,y,z)\f$ coordinates of each point
/// on the surface.
template <typename Frame>
struct CartesianCoords : db::ComputeTag {
  static std::string name() noexcept { return "CartesianCoords"; }
  using return_type = aliases::Vector<Frame>;
  static void function(gsl::not_null<aliases::Vector<Frame>*> coords,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const DataVector& radius,
                       const aliases::OneForm<Frame>& r_hat) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, Rhat<Frame>>;
};

/// `DxRadius(i)` is \f$\partial r_{\rm surf}/\partial x^i\f$.  Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct DxRadius : db::ComputeTag {
  static std::string name() noexcept { return "DxRadius"; }
  using return_type = aliases::OneForm<Frame>;
  static void function(gsl::not_null<aliases::OneForm<Frame>*> dx_radius,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const DataVector& radius,
                       const aliases::InvJacobian<Frame>& inv_jac) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, InvJacobian<Frame>>;
};

/// `D2xRadius(i,j)` is
/// \f$\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$. Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct D2xRadius : db::ComputeTag {
  static std::string name() noexcept { return "D2xRadius"; }
  using return_type = aliases::SecondDeriv<Frame>;
  static void function(gsl::not_null<aliases::SecondDeriv<Frame>*> d2x_radius,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const DataVector& radius,
                       const aliases::InvJacobian<Frame>& inv_jac,
                       const aliases::InvHessian<Frame>& inv_hess) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   InvJacobian<Frame>, InvHessian<Frame>>;
};

/// \f$\nabla^2 r_{\rm surf}\f$, the flat Laplacian of the surface.
/// This is \f$\eta^{ij}\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$,
/// where \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$.
template <typename Frame>
struct LaplacianRadius : db::ComputeTag {
  static std::string name() noexcept { return "LaplacianRadius"; }
  using return_type = DataVector;
  static void function(gsl::not_null<DataVector*> lap_radius,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const DataVector& radius,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, ThetaPhi<Frame>>;
};

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
struct NormalOneForm : db::ComputeTag {
  static std::string name() noexcept { return "NormalOneForm"; }
  using return_type = aliases::OneForm<Frame>;
  static void function(gsl::not_null<aliases::OneForm<Frame>*> one_form,
                       const aliases::OneForm<Frame>& dx_radius,
                       const aliases::OneForm<Frame>& r_hat) noexcept;
  using argument_tags = tmpl::list<DxRadius<Frame>, Rhat<Frame>>;
};

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
struct Tangents : db::ComputeTag {
  static std::string name() noexcept { return "Tangents"; }
  using return_type = aliases::Jacobian<Frame>;
  static void function(gsl::not_null<aliases::Jacobian<Frame>*> tangents,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const DataVector& radius,
                       const aliases::OneForm<Frame>& r_hat,
                       const aliases::Jacobian<Frame>& jac) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   Rhat<Frame>, Jacobian<Frame>>;
};

/// Computes the Euclidean area element on a Strahlkorper.
/// Useful for flat space integrals.
template <typename Frame>
struct EuclideanAreaElement : db::ComputeTag {
  static std::string name() noexcept { return "EuclideanAreaElement"; }
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>,
      const StrahlkorperTags::aliases::Jacobian<Frame>&,
      const tnsr::i<DataVector, 3, Frame>&, const DataVector&,
      const tnsr::i<DataVector, 3, Frame>&) noexcept>(
      &::StrahlkorperGr::euclidean_area_element<Frame>);
  using argument_tags = tmpl::list<
      StrahlkorperTags::Jacobian<Frame>, StrahlkorperTags::NormalOneForm<Frame>,
      StrahlkorperTags::Radius<Frame>, StrahlkorperTags::Rhat<Frame>>;
};

/// Computes the flat-space integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegral : db::ComputeTag {
  static std::string name() noexcept {
    return "EuclideanSurfaceIntegral(" + db::tag_name<IntegrandTag>() + ")";
  }
  // return type is `double` so returning by value is OK
  static constexpr auto function =
      ::StrahlkorperGr::surface_integral_of_scalar<Frame>;
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};

/// Computes the Euclidean-space integral of a vector over a
/// Strahlkorper, \f$\oint V^i s_i (s_j s_k \delta^{jk})^{-1/2} d^2 S\f$,
/// where \f$s_i\f$ is the Strahlkorper surface unit normal and
/// \f$\delta^{ij}\f$ is the Kronecker delta.  Note that \f$s_i\f$ is
/// not assumed to be normalized; the denominator of the integrand
/// effectively normalizes it using the Euclidean metric.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralVector : db::ComputeTag {
  static std::string name() noexcept {
    return "EuclideanSurfaceIntegralVector(" + db::tag_name<IntegrandTag>() +
           ")";
  }
  // return type is `double` so returning by value is OK
  static constexpr auto function =
      ::StrahlkorperGr::euclidean_surface_integral_of_vector<Frame>;
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::NormalOneForm<Frame>,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};

template <typename Frame>
using items_tags = tmpl::list<Strahlkorper<Frame>>;

template <typename Frame>
using compute_items_tags =
    tmpl::list<ThetaPhi<Frame>, Rhat<Frame>, Jacobian<Frame>,
               InvJacobian<Frame>, InvHessian<Frame>, Radius<Frame>,
               CartesianCoords<Frame>, DxRadius<Frame>, D2xRadius<Frame>,
               LaplacianRadius<Frame>, NormalOneForm<Frame>, Tangents<Frame>>;

}  // namespace StrahlkorperTags

namespace StrahlkorperGr {
/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper` that
/// also need a metric.
namespace Tags {

/// Computes the area element on a Strahlkorper. Useful for integrals.
template <typename Frame>
struct AreaElement : db::ComputeTag {
  static std::string name() noexcept { return "AreaElement"; }
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const StrahlkorperTags::aliases::Jacobian<Frame>&,
      const tnsr::i<DataVector, 3, Frame>&, const DataVector&,
      const tnsr::i<DataVector, 3, Frame>&) noexcept>(&area_element<Frame>);
  using argument_tags = tmpl::list<
      gr::Tags::SpatialMetric<3, Frame>, StrahlkorperTags::Jacobian<Frame>,
      StrahlkorperTags::NormalOneForm<Frame>, StrahlkorperTags::Radius<Frame>,
      StrahlkorperTags::Rhat<Frame>>;
};

/// Computes the integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral : db::ComputeTag {
  static std::string name() noexcept {
    return "SurfaceIntegral(" + db::tag_name<IntegrandTag>() + ")";
  }
  // return type is `double` so returning by value is OK
  static constexpr auto function = surface_integral_of_scalar<Frame>;
  using argument_tags = tmpl::list<AreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};
/// Tag representing the surface area of a Strahlkorper
struct Area : db::SimpleTag {
  using type = double;
};
/// Computes the surface area of a Strahlkorer, \f$A = \oint_S dA\f$ given an
/// AreaElement \f$dA\f$ and a Strahlkorper \f$S\f$.
template <typename Frame>
struct AreaCompute : Area, db::ComputeTag {
  using base = Area;
  static double function(const Strahlkorper<Frame>& strahlkorper,
                         const Scalar<DataVector>& area_element) noexcept {
    return strahlkorper.ylm_spherepack().definite_integral(
        get(area_element).data());
  }
  using argument_tags =
      tmpl::list<StrahlkorperTags::Strahlkorper<Frame>, AreaElement<Frame>>;
};

}  // namespace Tags
}  // namespace StrahlkorperGr
