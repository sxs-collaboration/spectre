// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace Frame {
struct Distorted;
struct Grid;
}  // namespace Frame
/// \endcond

namespace control_system::size {

/*!
 * \brief Computes the derivative of the comoving characteristic speed
 * with respect to the size map parameter.
 *
 * \param result the derivative of the comoving char speed
 *        \f$d v_c/d\lambda_{00}\f$, which is computed here using
 *        Eq. (\f$\ref{eq:result}\f$).
 * \param lambda_00 the map parameter \f$\lambda_{00}\f$. This is the usual
 *        spherical harmonic coefficient, not a Spherepack value.
 * \param dt_lambda_00 the time derivative of the map parameter
 * \param horizon_00 the average coefficient of the horizon \f$\hat{S}_{00}\f$.
 *        This is the usual spherical harmonic coefficient, not a Spherepack
 *        value.
 * \param dt_horizon_00 the time derivative of horizon_00
 * \param grid_frame_excision_sphere_radius radius of the excision boundary
 *        in the grid frame, \f$r_{\mathrm{EB}}\f$.
 * \param excision_rhat the direction cosine \f$\xi_\hat{i}\f$. Not a
 *        spacetime tensor: it is raised/lowered with \f$\delta_{ij}\f$
 * \param excision_normal_one_form the unnormalized one-form
 *        \f$\hat{s}_\hat{i}\f$
 * \param excision_normal_one_form_norm the norm of the one-form \f$a\f$
 * \param distorted_components_of_grid_shift the quantity
 *        \f$\beta^i \frac{\partial x^\hat{i}}{\partial x_i}\f$
 *        evaluated on the excision boundary.  This is not the shift in
 *        the distorted frame.
 * \param inverse_spatial_metric_on_excision_boundary metric in
 *        the distorted frame.
 * \param spatial_christoffel_second_kind the Christoffel symbols
 *        \f$\Gamma^\hat{k}_{\hat{i}\hat{j}}\f$
 * \param deriv_lapse the spatial derivative of the lapse
 *        \f$\partial_\hat{i} \alpha\f$
 * \param deriv_of_distorted_shift the spatial derivative of the shift in the
 *        distorted frame
 *        \f$\partial_\hat{j} \hat{\beta}^\hat{i}\f$. This is not the
 *        derivative of distorted_components_of_grid_shift.
 * \param inverse_jacobian_grid_to_distorted the quantity
 *        \f$J^i_\hat{k}=\partial_\hat{k} x^i\f$,
 *        where \f$x^i\f$ are the grid frame coordinates and
 *        \f$x^{\hat k}\f$ are the distorted frame coordinates.
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
 * (Note that the usual expression for the characteristic speed, as in
 * eq. 87 of \cite Hemberger2012jz, has
 * a minus sign and defines \f$n_i\f$ as the inward-pointing (i.e. out of the
 * computational domain) normal; here
 * we have a plus sign and we define \f$n_i\f$ as outward-pointing because
 * the outward-pointing normal is passed into comoving_char_speed_derivative.)
 *
 * The size/shape map at the excision boundary is given by Eq. 72 of
 * \cite Hemberger2012jz :
 * \f{align}
 *   \hat{x}^i &= \frac{x^i}{r_{\mathrm{EB}}}
 *      \left(1 - \lambda_{00} Y_{00}
 *      -\sum_{\ell>0} Y_{\ell m} \lambda_{\ell m}\right),
 *   \label{eq:map}
 * \f}
 * where \f$\hat{x}^i\f$ are the distorted-frame coordinates and \f$x^i\f$
 * are the grid-frame coordinates, and where we have separated the \f$\ell=0\f$
 * piece from the sum.
 * Here \f$Y_{\ell m}\f$ are
 * spherical harmonics, \f$\lambda_{\ell m}\f$ are
 * the map parameters, and \f$r_{\mathrm{EB}}\f$ is the radius of the
 * excision boundary in the
 * grid frame (where the excision boundary is a sphere). The final term with
 * the sum over $\ell>0$ is independent of \f$\lambda_{00}\f$,
 * and will not be important
 * because below we will be differentiating the map with respect
 * to \f$\lambda_{00}\f$.
 *
 * The comoving characteristic speed is given by rewriting Eq. 98
 * of \cite Hemberger2012jz in terms of the distorted-frame shift:
 * \f{align}
 *     v_c &= -\alpha +\hat{n}_\hat{i}\hat{\beta}^\hat{i}
 *           - Y_{00} \hat{n}_{\hat i} \xi^{\hat i}
 *           \left[ \dot{\hat{S}}_{00} (\lambda_{00}
 *                   - r_{\mathrm{EB}}/Y_{00}) / \hat{S}_{00}
 *           + \frac{1}{Y_{00}} \sum_{\ell>0} Y_{\ell m} \dot{\lambda}_{\ell m}
 *         \right], \\
 *         &= -\alpha +\hat{n}_\hat{i}\beta^\hat{i}
 *           - Y_{00} \hat{n}_{\hat i} \xi^{\hat i}
 *           \left[ \dot{\hat{S}}_{00} (\lambda_{00}
 *                   - r_{\mathrm{EB}}/Y_{00}) / \hat{S}_{00}
 *           -\dot{\lambda}_{00} \right], \label{eq:comovingspeed}
 * \f}
 * where in the last line we have rewritten $\hat{\beta}^\hat{i}$
 * in terms of $\beta^\hat{i}$ (see Eq. (\f$\ref{eq:framecompsshiftdef}\f$)
 * below) and we have substituted
 * the time derivative of Eq. (\f$\ref{eq:map}\f$).
 * Here \f$\dot{\lambda}_{00}\f$ is the time derivative of
 * \f$\lambda_{00}\f$, and
 * \f$\hat{S}_{00}\f$ is the constant spherical-harmonic coefficient of the
 * horizon and \f$\dot{\hat{S}}_{00}\f$ is its time derivative.
 * The symbol \f$\xi^{\hat i}\f$ is
 * a direction cosine, i.e. \f$x^i/r_{\mathrm{EB}}\f$ evaluated on the
 * excision boundary, which is the same as
 * \f$\hat{x}^i/\hat{r}_{\mathrm{EB}}\f$ evaluated on the excision boundary
 * because the size and shape maps preserve angles.  Note that
 * \f$r_{\mathrm{EB}}\f$ is a constant but \f$\hat{r}_{\mathrm{EB}}\f$ is
 * a function of angles.  Note also
 * that \f$\xi^{\hat i}\f$ is **not** a vector; it
 * is a coordinate quantity. In particular,
 * the lower-index \f$\xi_{\hat i}\f$ is \f$\delta_{ij}x^j/r_{\mathrm{EB}}\f$.
 * The non-vectorness of \f$\xi^{\hat i}\f$ (and of \f$x^i\f$ itself
 * in Eq. (\f$\ref{eq:map}\f$)) might cause some confusion when using the
 * Einstein summation convention; we attempt to alleviate that confusion by
 * never using the lower-index \f$\xi_{\hat i}\f$ and by keeping
 * \f$\delta_{ij}\f$ in formulas below.
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
 * Eq. (\f$\ref{eq:map}\f$) has the same index on both sides of the equation
 * and because \f$\xi^{\hat i}\f$ and \f$x^i\f$ are not tensors.
 * The quantity \f$\beta^\hat{i}\f$ in Eq. (\f$\ref{eq:comovingspeed}\f$)
 * is the distorted-frame
 * component of the grid-frame shift, defined by
 * \f{align}
 * \beta^\hat{i} &= \beta^i \frac{\partial \hat{x}^\hat{i}}{\partial x^i}.
 * \label{eq:shiftyquantity}
 * \f}
 * This is **not** the shift in the distorted frame \f$\hat{\beta}^\hat{i}\f$,
 * because the shift does
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
 * ## Derivation of derivative of comoving characteristic speed
 *
 * This function computes
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
 * where \f$B\f$ represents the term with the sum over \f$\ell>0\f$
 * in Eq. (\f$\ref{eq:map}\f$); this term is independent
 * of \f$\lambda_{00}\f$.
 * Therefore, we have
 * \f{align}
 *  \frac{d}{d\lambda_{00}}\frac{\partial \hat{x}^i}{\partial x^j} &=
 *  -\frac{Y_{00}}{r_{\mathrm{EB}}} \delta^i_j.
 * \label{eq:derivjacobian}
 * \f}
 *
 * But we want the derivative of the inverse Jacobian, not the forward
 * Jacobian. By taking the derivative of the identity
 * \f{align}
 *  \frac{\partial \hat{x}^\hat{i}}{\partial x^k}
 *  \frac{\partial x^k}{\partial \hat{x}^\hat{j}} &= \delta^\hat{i}_\hat{j}
 * \f}
 * and by using Eq. (\f$\ref{eq:derivjacobian}\f$) we can derive
 * \f{align}
 *  \frac{d}{d\lambda_{00}}\frac{\partial x^i}{\partial \hat{x}^j} &=
 *  +\frac{Y_{00}}{r_{\mathrm{EB}}}
 *   \frac{\partial x^i}{\partial \hat{x}^k}
 *   \frac{\partial x^k}{\partial \hat{x}^j}.
 *   \label{eq:strangederivjacobian}
 * \f}
 * Note that the right-hand side of Eq. (\f$\ref{eq:strangederivjacobian}\f$)
 * has two inverse Jacobians contracted with each other, which is not
 * the same as \f$\delta^i_j\f$.
 *
 * ### Derivative of a function of space
 *
 * Assume we have an arbitrary function of space \f$f(\hat{x}^i)\f$.
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
 *  \label{eq:framecompsshiftdef}
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
 * because the excision boundary is a sphere of fixed radius in the
 * grid frame. Therefore
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
 *  \frac{Y_{00}}{r_{\mathrm{EB}}}
 *  \hat{s}_k\frac{\partial x^k}{\partial \hat{x}^\hat{i}}, \\
 *  \frac{d}{d\lambda_{00}} a^2 &= 2 \frac{Y_{00}}{r_{\mathrm{EB}}}
 *  \hat{s}_k\frac{\partial x^k}{\partial \hat{x}^\hat{i}}
 *  \hat{s}_{\hat j} \gamma^{\hat{i} \hat{j}}
 *  + \hat{s}_{\hat i} \hat{s}_{\hat j}
 *  \gamma^{\hat{i} \hat{k}} \gamma^{\hat{j} \hat{l}}
 *  \xi^\hat{m} Y_{00} \partial_{\hat m} \gamma_{\hat{k} \hat{l}}.
 * \f}
 * Here we have used Eq. (\f$\ref{eq:strangederivjacobian}\f$) to differentiate
 * the Jacobian, and Eq. (\f$\ref{eq:derivf}\f$) to differentiate the 3-metric.
 * We have also refrained from raising and lowering indices
 * on \f$\hat{n}_\hat{i}\f$, \f$\hat{s}_\hat{i}\f$, and \f$\xi^\hat{i}\f$
 * to alleviate potential confusion over whether to raise or lower using
 * \f$\gamma_{\hat{i} \hat{j}}\f$ or using \f$\delta_{\hat{i}\hat{j}}\f$.
 * The factor \f$\hat{s}_k \partial x^k/\partial \hat{x}^\hat{i}\f$
 * is unusal and is not a tensor
 * (\f$\hat{s}_k\f$ is a tensor but the Jacobian it is being multiplied by
 * is the inverse of the one that would transform it into a different frame);
 * this factor arises because some quantities being differentiated are
 * not tensors.
 *
 * Given the above, the derivative of the normalized normal one-form is
 * \f{align}
 *  \frac{d}{d\lambda_{00}} \hat{n}_{\hat i} &=
 *  \frac{1}{a}\frac{d}{d\lambda_{00}} \hat{s}_{\hat i}
 *  - \hat{s}_{\hat i} \frac{1}{2a^3}
 *  \frac{d}{d\lambda_{00}} a^2\\
 *  &=
 *  \hat{s}_i\frac{Y_{00}}{a r_{\mathrm{EB}}}
 *    \frac{\partial x^i}{\partial \hat{x}^\hat{i}}
 *  - \hat{s}_{\hat i} \frac{1}{a^3} \hat{s}_i\frac{Y_{00}}{r_{\mathrm{EB}}}
 *    \frac{\partial x^i}{\partial \hat{x}^\hat{k}}
 *    \hat{s}_{\hat j} \gamma^{\hat{k} \hat{j}}
 *  - \hat{s}_{\hat i} \frac{Y_{00}}{2a^3} \hat{s}_{\hat p}
 *  \hat{s}_{\hat j} \gamma^{\hat{p} \hat{k}}
 *  \gamma^{\hat{j} \hat{l}}
 *  \xi^\hat{m} \partial_{\hat m} \gamma_{\hat{k} \hat{l}} \\
 *  &=
 *  \hat{n}_i
 *  \frac{\partial x^i}{\partial \hat{x}^\hat{k}}
 *  \frac{Y_{00}}{r_{\mathrm{EB}}}
 *  (\delta^\hat{k}_\hat{i} - \hat{n}^\hat{k} \hat{n}_\hat{i})
 *  - \hat{s}_{\hat i} \frac{Y_{00}}{2a^3} \hat{s}_{\hat p}
 *  \hat{s}_{\hat j} \gamma^{\hat{p} \hat{k}}
 *  \gamma^{\hat{j} \hat{l}}
 *  \xi^\hat{m} \partial_{\hat m} \gamma_{\hat{k} \hat{l}}
 *  \label{eq:dnormal} \\
 *  &=
 *  \hat{n}_i
 *  \frac{\partial x^i}{\partial \hat{x}^\hat{k}}
 *  \frac{Y_{00}}{r_{\mathrm{EB}}}
 *  (\delta^\hat{k}_\hat{i} - \hat{n}^\hat{k} \hat{n}_\hat{i})
 *  - Y_{00} \hat{n}_{\hat i} \hat{n}_{\hat p}
 *  \hat{n}_{\hat j} \gamma^{\hat{p} \hat{k}}
 *  \xi^\hat{m} \Gamma^\hat{j}_{\hat{k} \hat{m}}
 *  \label{eq:dnormalgamma},
 * \f}
 * where we have eliminated \f$\hat{s}_{\hat i}\f$ and \f$a\f$ in favor
 * of \f$\hat{n}_{\hat i}\f$
 * and we have substituted 3-Christoffel symbols for
 * spatial derivatives of the 3-metric (and the factor of 2 on the penultimate
 * line has been absorbed into the 3-Christoffel symbol on the last line).
 * Note that the last term in Eq.
 * (\f$\ref{eq:dnormalgamma}\f$) could also be derived by differentiating
 * \f$\hat{n}_\hat{i}\hat{n}_\hat{j}\gamma^{\hat{i}\hat{j}}=1\f$.
 * The first term in Eq. (\f$\ref{eq:dnormalgamma}\f$) is strange because
 * the inverse Jacobian (as opposed to the forward Jacobian) is contracted
 * with \f$\hat{n}_i\f$, so that is not a tensor transformation.
 *
 * We can now differentiate Eq. (\f$\ref{eq:comovingspeed}\f$) to obtain
 * \f{align}
 *  \frac{d}{d\lambda_{00}} v_c &=
 *  \xi^\hat{i} Y_{00} \partial_\hat{i} \alpha
 *   +\left[ \beta^\hat{i}
 *   - Y_{00} \xi^\hat{i} \dot{\hat{S}}_{00} (\lambda_{00}
 *                   - r_{\mathrm{EB}}/Y_{00}) / \hat{S}_{00}
 *           + Y_{00} \xi^\hat{i}\dot{\lambda}_{00} \right]
 *  \frac{d}{d\lambda_{00}} \hat{n}_{\hat i} \nonumber \\
 *  &- \hat{n}_{\hat i} \xi^\hat{j} Y_{00} \partial_\hat{j} \hat{\beta}^\hat{i}
 *  - Y_{00} \hat{n}_{\hat i} \xi^{\hat i} \dot{\hat{S}}_{00}/\hat{S}_{00},
 * \label{eq:result}
 * \f}
 * where \f$\frac{d}{d\lambda_{00}} \hat{n}_{\hat i}\f$ is given by
 * Eq. (\f$\ref{eq:dnormalgamma}\f$).
 */
void comoving_char_speed_derivative(
    gsl::not_null<Scalar<DataVector>*> result, double lambda_00,
    double dt_lambda_00, double horizon_00, double dt_horizon_00,
    double grid_frame_excision_sphere_radius,
    const tnsr::i<DataVector, 3, Frame::Distorted>& excision_rhat,
    const tnsr::i<DataVector, 3, Frame::Distorted>& excision_normal_one_form,
    const Scalar<DataVector>& excision_normal_one_form_norm,
    const tnsr::I<DataVector, 3, Frame::Distorted>&
        distorted_components_of_grid_shift,
    const tnsr::II<DataVector, 3, Frame::Distorted>&
        inverse_spatial_metric_on_excision_boundary,
    const tnsr::Ijj<DataVector, 3, Frame::Distorted>&
        spatial_christoffel_second_kind,
    const tnsr::i<DataVector, 3, Frame::Distorted>& deriv_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Distorted>& deriv_of_distorted_shift,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Distorted>&
        inverse_jacobian_grid_to_distorted);
}  // namespace control_system::size
