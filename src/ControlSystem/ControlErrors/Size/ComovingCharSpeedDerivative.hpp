// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace Frame {
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace control_system::size {

/*!
 * \brief Computes the derivative of the comoving characteristic speed
 * with respect to the size map parameter.
 *
 * \tparam Frame should be ::Frame::Distorted if ::Frame::Distorted exists.
 * \param result the derivative of the comoving char speed
 *        \f$d v_c/d\lambda_{00}\f$, which is computed here using
 *        Eq. (\f$\ref{eq:result}\f$).
 * \param lambda_00 the map parameter \f$\lambda_{00}\f$
 * \param dt_lambda_00 the time derivative of the map parameter
 * \param horizon_00 the average coefficient of the horizon \f$\hat{S}_{00}\f$
 * \param dt_horizon_00 the time derivative of horizon_00
 * \param grid_frame_excision_sphere_radius radius of the excision boundary
 *        in the grid frame, \f$r_{\mathrm{EB}}\f$.
 * \param excision_rhat the direction cosine \f$\xi^\hat{i}\f$
 * \param excision_normal_one_form the unnormalized one-form
 *        \f$\hat{s}_\hat{i}\f$
 * \param excision_normal_one_form_norm the norm of the one-form \f$a\f$
 * \param frame_components_of_grid_shift the quantity
 *        \f$\beta^i \frac{\partial x^\hat{i}}{\partial x_i}\f$
 *        evaluated on the excision boundary.  This is not the shift in
 *        the distorted frame.
 * \param inverse_spatial_metric_on_excision_boundary metric in frame Frame.
 * \param spatial_christoffel_second_kind the Christoffel symbols
 *        \f$\Gamma^\hat{k}_{\hat{i}\hat{j}}\f$
 * \param deriv_lapse the spatial derivative of the lapse
 *        \f$\partial_\hat{i} \alpha\f$
 * \param deriv_shift the spatial derivative of the shift
 *        \f$\partial_\hat{j} \hat{\beta}^\hat{i}\f$. This is not the
 *        derivative of frame_components_of_grid_shift.
 *
 * ## Background
 *
 * The characteristic speed on the excision boundary is
 * \f{align}
 *     v &= -\alpha + n_i\beta^i
 * \f}
 * where \f$\alpha\f$ is the lapse (invariant under frame transformations),
 * \f$\beta^i\f$ is the grid-frame shift, and \f$n_i\f$ is the metric-normalized
 * **outward-pointing** (i.e. pointing out of the black hole,
 * toward larger radius)
 * normal one-form to the excision boundary in the grid frame.
 * (Note that the usual expression for the characteristic speed has
 * a minus sign and defines \f$n_i\f$ as the inward-pointing (i.e. out of the
 * computational domain) normal; here
 * we have a plus sign and we define \f$n_i\f$ as outward-pointing because
 * the outward-pointing normal is passed into comoving_char_speed_derivative.)
 *
 * For the size/shape map at the excision boundary,
 * \f{align}
 *   \hat{x}^i &= x^i (1 - \lambda_{00} Y_{00}/r_{\mathrm{EB}} + B),
 *   \label{eq:map}
 * \f}
 * where \f$\hat{x}^i\f$ are the distorted-frame coordinates and \f$x^i\f$
 * are the grid-frame coordinates.
 * Here \f$Y_{00}\f$ is a
 * spherical harmonic, \f$\lambda_{00}\f$ is
 * the map parameter, and \f$r_{\mathrm{EB}}\f$ is the radius of the
 * excision boundary in the
 * grid frame (where the excision boundary is a sphere). Here
 * \f$B\f$ represents terms that depend on angles but are independent of
 * \f$\lambda_{00}\f$.  These \f$B\f$ terms depend on \f$\lambda_{\ell m}\f$ for
 * \f$\ell>0\f$ but the precise form of these terms will not be important here,
 * because below we will be differentiating the map with respect
 * to \f$\lambda_{00}\f$.
 *
 * The comoving characteristic speed is
 * \f{align}
 *     v_c &= -\alpha -\hat{n}_\hat{i}\beta^\hat{i}
 *           - Y_{00} \hat{n}_{\hat i} \xi^{\hat i}
 *           \left[ \dot{\hat{S}}_{00} (\lambda_{00}
 *                   - r_{\mathrm{EB}}/Y_{00}) / \hat{S}_{00}
 *           -\dot{\lambda}_{00} \right], \label{eq:comovingspeed}
 * \f}
 * where \f$\dot{\lambda}_{00}\f$ is the time derivative of
 * \f$\lambda_{00}\f$, and
 * \f$\hat{S}_{00}\f$ is the constant spherical-harmonic coefficient of the
 * horizon and \f$\dot{\hat{S}}_{00}\f$ is its time derivative.
 * The symbol \f$\xi^{\hat i}\f$ is
 * a direction cosine, i.e. \f$x^i/r_{\mathrm{EB}}\f$ evaluated on the
 * excision boundary, which is the same as
 * \f$\hat{x}^i/\hat{r}_{\mathrm{EB}}\f$ evaluated on the excision boundary
 * because the size and shape maps preserve angles.  Note that
 * \f$r_{\mathrm{EB}}\f$ is a constant but \f$\hat{r}_{\mathrm{EB}}\f$ is
 * a function of angles.
 * The normal
 * \f$\hat{n}_\hat{i}\f$ is the same as $n_i$
 * transformed into the distorted frame, that is
 * \f$\hat{n}_\hat{i} = n_j \partial x^j/\partial\hat{x}^\hat{i}\f$.
 * We have put a hat on \f$\hat{n}\f$ in addition to putting a hat on
 * its index
 * (despite the usual convention that tensors have
 * decorations on indices and not on the tensors themselves)
 * to reduce later ambiguities
 * in notation that arise because
 * Eq. (\f$\ref{eq:map}\f$) has the same index on both sides of the equation.
 * The quantity \f$\beta^\hat{i}\f$ in Eq. (\f$\ref{eq:comovingspeed}\f$)
 * is the distorted-frame
 * component of the grid-frame shift, defined by
 * \f{align}
 * \beta^\hat{i} &= \beta^i \frac{\partial \hat{x}^\hat{i}}{\partial x^i}.
 * \label{eq:shiftyquantity}
 * \f}
 * This is **not** the shift in the distorted frame, because the shift does
 * not transform like a spatial tensor under the maps.
 *
 * If the comoving characteristic speed \f$v_c\f$ is negative and remains
 * negative forever, then size control will fail.  Therefore \f$v_c\f$ is
 * used in making decisions in size control. We care about
 * \f$d v_c/d\lambda_{00}\f$ because the sign of that quantity tells us
 * whether \f$v_c\f$ will increase or decrease when we increase or decrease
 * the map parameter \f$\lambda_{00}\f$; this information will be used to
 * decide whether to transition between different size control states.
 *
 * ## Derivation of comoving_char_speed_derivative
 *
 * The function comoving_char_speed_derivative computes
 * \f$d v_c/d\lambda_{00}\f$ on the excision boundary, where the total
 * derivative means that all other map parameters
 * (like \f$\lambda_{\ell m}\f$ for \f$\ell>0\f$) are held fixed, and the
 * coordinates of the excision boundary (the grid coordinates) are held fixed.
 * We also hold fixed \f$\dot{\lambda}_{00}\f$ because we are interested
 * in how $v_c$ changes from a configuration with a given
 * \f$\lambda_{00}\f$ and \f$\dot{\lambda}_{00}\sim 0\f$ to another
 * configuration with a different nearby \f$\lambda_{00}\f$ and also with
 * \f$\dot{\lambda}_{00}\sim 0\f$.
 *
 * Here we derive an expression for \f$d v_c/d\lambda_{00}\f$.
 * This expression will be
 * complicated, mostly because of the normals \f$\hat{n}_\hat{i}\f$ that appear
 * in Eq. (\f$\ref{eq:comovingspeed}\f$) and because of the Jacobians.
 *
 * ### Derivative of the Jacobian
 *
 * First, note that by differentiating Eq. (\f$\ref{eq:map}\f$) we obtain
 * \f{align}
 *   \frac{d\hat{x}^i}{d\lambda_{00}}
 *      &= - \frac{x^i Y_{00}}{r_{\mathrm{EB}}} \\
 *      &= -\xi^i Y_{00},
 * \f}
 * where the last line follows from the definition of the direction cosines.
 *
 * The Jacobian of the map is
 * \f{align}
 *  \frac{\partial \hat{x}^i}{\partial x^j}
 *    &= (1 - \lambda_{00} Y_{00}/r_{\mathrm{EB}} + B) \delta^i_j
 *    + x^i \frac{\partial B}{\partial x^j},
 * \f}
 * where the last term is independent of \f$\lambda_{00}\f$.
 * Therefore, we have
 * \f{align}
 *  \frac{d}{d\lambda_{00}}\frac{\partial \hat{x}^i}{\partial x^j} &=
 *  -\frac{Y_{00}}{r_{\mathrm{EB}}} \delta^i_j.
 * \f}
 *
 * By taking the derivative of the identity
 * \f{align}
 *  \frac{\partial \hat{x}^\hat{i}}{\partial x^k}
 *  \frac{\partial x^k}{\partial \hat{x}^\hat{j}} &= \delta^\hat{i}_\hat{j}
 * \f}
 * we can derive
 * \f{align}
 *  \frac{d}{d\lambda_{00}}\frac{\partial x^i}{\partial \hat{x}^j} &=
 *  +\frac{Y_{00}}{r_{\mathrm{EB}}} \delta^i_j.
 * \f}
 *
 * ### Derivative of a function of space
 *
 * Assume we have arbitrary function of space \f$f(\hat{x}^i)\f$.
 * Here we treat \f$f\f$ as a function of the distorted-frame
 * coordinates \f$\hat{x}^i\f$ and not a function of the grid-frame
 * coordinates.  This is because we consider the metric functions to be defined
 * in the inertial frame (and equivalently for our purposes the functions are
 * defined in the distorted frame because the distorted-to-inertial map
 * is independent of \f$\lambda_{00}\f$), and we consider \f$\lambda_{00}\f$
 * a parameter in a map that moves the grid with respect to these
 * distorted-frame metric functions.
 * The derivative of \f$f\f$ can be written
 * \f{align}
 *  \frac{d}{d\lambda_{00}}f &= \frac{\partial f}{\partial \hat{x}^i}
 *  \frac{d \hat{x}^i}{d\lambda_{00}}\\
 *  &= -\xi^\hat{i} Y_{00} \frac{\partial f}{\partial \hat{x}^i}.
 * \label{eq:derivf}
 * \f}
 * This is how we will evaluate derivatives of metric functions like
 * the lapse.
 *
 * ### Derivative of the distorted-frame components of the grid-frame shift.
 *
 * To differentiate the quantity defined by Eq. (\f$\ref{eq:shiftyquantity}\f$)
 * note that
 * \f{align}
 * \beta^\hat{i} &=
 *   \beta^i \frac{\partial \hat{x}^\hat{i}}{\partial x^i} \\
 *   &= \hat{\beta}^\hat{i} + \frac{\partial \hat{x}^\hat{i}}{\partial t},
 * \f}
 * where \f$\hat{\beta}^\hat{i} \equiv \alpha^2 g^{\hat{0}\hat{i}}\f$ is
 * the shift in the distorted frame.
 * From the map, Eq. (\f$\ref{eq:map}\f$), we see that
 * \f{align}
 *  \frac{d}{d\lambda_{00}} \frac{\partial \hat{x}^\hat{i}}{\partial t} &=0,
 * \f}
 * because there is no remaining \f$\lambda_{00}\f$ in
 * \f$\frac{\partial \hat{x}^\hat{i}}{\partial t}\f$.
 * So
 * \f{align}
 * \frac{d}{d\lambda_{00}}\beta^\hat{i} &=
 * \frac{d}{d\lambda_{00}} \hat{\beta}^\hat{i} \\
 * &= -\xi^\hat{j} Y_{00} \partial_\hat{j} \hat{\beta}^\hat{i},
 * \f}
 * where we have used Eq. (\f$\ref{eq:derivf}\f$) in the last line.
 * Note that we cannot use Eq. (\f$\ref{eq:derivf}\f$) on
 * \f$\beta^\hat{i}\f$ directly,
 * because \f$\beta^\hat{i}\f$ depends in a complicated
 * way on the grid-to-distorted map. In particular, we will be evaluating
 * \f$\partial_\hat{j} \hat{\beta}^\hat{i}\f$ numerically, and numerical
 * spatial derivatives \f$\partial_\hat{j} \hat{\beta}^\hat{i}\f$ are not
 * the same as numerical spatial derivatives
 * \f$\partial_\hat{j} \beta^\hat{i}\f$.
 *
 * ### Derivative of the normal one-form
 *
 * The normal to the surface is the most complicated expression in
 * Eq. (\f$\ref{eq:comovingspeed}\f$), because of how it depends on the
 * map and on the metric.
 * The grid-frame un-normalized outward-pointing one-form
 * to the excision boundary is
 * \f{align}
 *    s_i &= \xi^j \delta_{ij},
 * \f}
 * because the excision boundary is a sphere in the grid frame. Therefore
 * \f$s_i\f$ doesn't depend on \f$\lambda_{00}\f$.
 *
 * The normalized one-form \f$\hat{n}_\hat{i}\f$ is given by
 * \f{align}
 *   \hat{n}_\hat{i} &= \frac{\hat{s}_{\hat i}}{a},
 * \f}
 * where
 * \f{align}
 *   \hat{s}_{\hat i} &= s_i \frac{\partial x^i}{\partial \hat{x}^{\hat i}},\\
 *   a^2 &= \hat{s}_{\hat i} \hat{s}_{\hat j} \gamma^{\hat{i} \hat{j}}.
 * \f}
 * Here \f$\gamma^{\hat{i} \hat{j}}\f$ is the inverse 3-metric in the
 * distorted frame.  Again, to avoid ambiguity later,
 * we have put hats on \f$n\f$ and \f$s\f$, despite
 * the usual convention that when transforming tensors one puts
 * hats on the indices and not on the tensors.
 *
 * Now
 * \f{align}
 *  \frac{d}{d\lambda_{00}} \hat{s}_{\hat i} &=
 *  s_i\frac{Y_{00}}{r_{\mathrm{EB}}} \delta^i_\hat{i},\\
 *  \frac{d}{d\lambda_{00}} a^2 &= 2 s_i\frac{Y_{00}}{r_{\mathrm{EB}}}
 *  \delta^i_\hat{i} \hat{s}_{\hat j} \gamma^{\hat{i} \hat{j}}
 *  + \hat{s}_{\hat i} \hat{s}_{\hat j}
 *  \gamma^{\hat{i} \hat{k}} \gamma^{\hat{j} \hat{l}}
 *  \xi^\hat{m} Y_{00} \partial_{\hat m} \gamma_{\hat{k} \hat{l}}.
 * \f}
 * Here we have used Eq. (\f$\ref{eq:derivf}\f$) to differentiate the 3-metric.
 * We have also refrained from raising and lowering indices
 * to alleviate potential confusion over whether to raise or lower using
 * \f$\gamma_{\hat{i} \hat{j}}\f$ or using \f$\delta_{\hat{i}\hat{j}}\f$.
 * Given the above, the derivative of the normalized normal one-form is
 * \f{align}
 *  \frac{d}{d\lambda_{00}} \hat{n}_{\hat i} &=
 *  \frac{1}{a}\frac{d}{d\lambda_{00}} \hat{s}_{\hat i}
 *  - \hat{s}_{\hat i} \frac{1}{2a^3}
 *  \frac{d}{d\lambda_{00}} a^2\\
 *  &=
 *  s_i\frac{Y_{00}}{r_{\mathrm{EB}}} \delta^i_\hat{i}
 *  - \hat{s}_{\hat i} \frac{1}{a^3} s_i\frac{Y_{00}}{r_{\mathrm{EB}}}
 *  \delta^i_\hat{k} \hat{s}_{\hat j} \gamma^{\hat{k} \hat{j}}
 *  - \hat{s}_{\hat i} \frac{Y_{00}}{2a^3} \hat{s}_{\hat p}
 *  \hat{s}_{\hat j} \gamma^{\hat{p} \hat{k}}
 *  \gamma^{\hat{j} \hat{l}}
 *  \xi^\hat{m} \partial_{\hat m} \gamma_{\hat{k} \hat{l}} \\
 *  &=
 *  \xi_\hat{i}\frac{Y_{00}}{r_{\mathrm{EB}}}
 *  - \hat{s}_{\hat i} \frac{1}{a^3} \xi_\hat{k}\frac{Y_{00}}{r_{\mathrm{EB}}}
 *    \hat{s}_{\hat j} \gamma^{\hat{k} \hat{j}}
 *  - \hat{s}_{\hat i} \frac{Y_{00}}{2a^3} \hat{s}_{\hat p}
 *  \hat{s}_{\hat j} \gamma^{\hat{p} \hat{k}}
 *  \gamma^{\hat{j} \hat{l}}
 *  \xi^\hat{m} \partial_{\hat m} \gamma_{\hat{k} \hat{l}}
 *  \label{eq:dnormal} \\
 *  &=
 *  \xi_\hat{i}\frac{Y_{00}}{r_{\mathrm{EB}}}
 *  - \frac{Y_{00}}{a r_{\mathrm{EB}}} \xi_\hat{k} \hat{n}_{\hat i}
 *    \hat{n}_{\hat j} \gamma^{\hat{k} \hat{j}}
 *  - Y_{00} \hat{n}_{\hat i} \hat{n}_{\hat p}
 *  \hat{n}_{\hat j} \gamma^{\hat{p} \hat{k}}
 *  \xi^\hat{m} \Gamma^\hat{j}_{\hat{k} \hat{m}}
 *  \label{eq:dnormalgamma},
 * \f}
 * where in the last line we have substituted \f$\hat{n}_{\hat i}\f$ for
 * \f$\hat{s}_{\hat i}\f$ and we have substituted 3-Christoffel symbols for
 * spatial derivatives of the 3-metric.
 *
 * So we can now differentiate Eq. (\f$\ref{eq:comovingspeed}\f$) to obtain
 * \f{align}
 *  \frac{d}{d\lambda_{00}} v_c &=
 *  -\xi^\hat{i} Y_{00} \partial_\hat{i} \alpha
 *   +\left[ -\beta^\hat{i}
 *   - Y_{00} \xi^\hat{i} \dot{\hat{S}}_{00} (\lambda_{00}
 *                   - r_{\mathrm{EB}}/Y_{00}) / \hat{S}_{00}
 *           + Y_{00} \xi^\hat{i}\dot{\lambda}_{00} \right]
 *  \frac{d}{d\lambda_{00}} \hat{n}_{\hat i} \nonumber \\
 *  &+ \hat{n}_{\hat i} \xi^\hat{j} Y_{00} \partial_\hat{j} \hat{\beta}^\hat{i}
 *  - Y_{00} \hat{n}_{\hat i} \xi^{\hat i} \dot{\hat{S}}_{00}/\hat{S}_{00},
 * \label{eq:result}
 * \f}
 * where \f$\frac{d}{d\lambda_{00}} \hat{n}_{\hat i}\f$ is given by
 * Eq. (\f$\ref{eq:dnormalgamma}\f$).
 */
template <typename Frame>
void comoving_char_speed_derivative(
    gsl::not_null<Scalar<DataVector>*> result, double lambda_00,
    double dt_lambda_00, double horizon_00, double dt_horizon_00,
    double grid_frame_excision_sphere_radius,
    const tnsr::i<DataVector, 3, Frame>& excision_rhat,
    const tnsr::i<DataVector, 3, Frame>& excision_normal_one_form,
    const Scalar<DataVector>& excision_normal_one_form_norm,
    const tnsr::I<DataVector, 3, Frame>& frame_components_of_grid_shift,
    const tnsr::II<DataVector, 3, Frame>&
        inverse_spatial_metric_on_excision_boundary,
    const tnsr::Ijj<DataVector, 3, Frame>& spatial_christoffel_second_kind,
    const tnsr::i<DataVector, 3, Frame>& deriv_lapse,
    const tnsr::iJ<DataVector, 3, Frame>& deriv_shift);
}  // namespace control_system::size
