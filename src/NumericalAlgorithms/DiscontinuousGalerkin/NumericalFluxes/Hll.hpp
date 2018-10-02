// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace NumericalFluxes {

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \ingroup NumericalFluxesGroup
 * \brief Compute the HLL numerical flux.
 *
 * Let \f$U\f$ and \f$F^i(U)\f$ be the state vector of the system and its
 * corresponding volume flux, respectively. Let \f$n_i\f$ be the unit normal to
 * the interface. Defining \f$F_n(U) := n_i F^i(U)\f$, and denoting the
 * corresponding projections of the numerical fluxes as \f${F_n}^*(U)\f$, the
 * HLL flux \ref hll_ref "[1]" for each variable is
 *
 * \f{align*}
 * {F_n}^*(U) = \frac{c_\text{max} F_n(U_\text{int}) -
 * c_\text{min} F_n(U_\text{ext})}{c_\text{max} - c_\text{min}}
 * - \frac{c_\text{min}c_\text{max}}{c_\text{max} - c_\text{min}}
 * \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, respectively.
 * \f$c_\text{min}\f$ and \f$c_\text{max}\f$ are estimates on the minimum
 * and maximum signal velocities bounding the interior-moving and
 * exterior-moving wavespeeds that arise when solving the Riemann problem.
 * Here we use the simple estimates \ref estimates_ref "[2]"
 *
 * \f{align*}
 * c_\text{min} &= \text{min}\left( \lambda_1(U_\text{int}; n_\text{int}),
 * \lambda_1(U_\text{ext}; n_\text{ext}), 0\right),\\
 * c_\text{max} &= \text{max}\left( \lambda_N(U_\text{int}; n_\text{int}),
 * \lambda_N(U_\text{ext}; n_\text{ext}), 0\right),
 * \f}
 *
 * where \f$\lambda_1(U; n) \leq \lambda_2(U; n) \leq ... \leq
 * \lambda_N(U; n)\f$ are the (ordered) characteristic speeds of the system.
 * Note that the definitions above ensure that \f$c_\text{min} \leq 0\f$ and
 * \f$c_\text{max} \geq 0\f$. Also, for either \f$c_\text{min} = 0\f$ or
 * \f$c_\text{max} = 0\f$ (i.e. all characteristics move in the same direction)
 * the HLL flux reduces to pure upwinding.
 *
 * \anchor hll_ref [1] A. Harten, P. D. Lax, B. van Leer, On Upstream
 * Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws,
 * SIAM Rev. [25 (1983) 35](https://doi.org/10.1137/1025002)
 *
 * \anchor estimates_ref [2] S. F. Davis, Simplified Second-Order Godunov-Type
 * Methods, SIAM J. Sci. Stat. Comput.
 * [9 (1988) 445](https://doi.org/10.1137/0909030)
 */
template <typename System>
struct Hll {
 private:
  using char_speeds_tag = typename System::char_speeds_tag;
  using variables_tag = typename System::variables_tag;

 public:
  /// The minimum signal velocity bounding
  /// the wavespeeds on one side of the interface.
  struct MinSignalSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "MinSignalSpeed"; }
  };

  /// The maximum signal velocity bounding
  /// the wavespeeds on one side of the interface.
  struct MaxSignalSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "MaxSignalSpeed"; }
  };

  using package_tags = tmpl::append<
      db::split_tag<db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>,
      db::split_tag<variables_tag>,
      tmpl::list<MinSignalSpeed, MaxSignalSpeed>>;

  using argument_tags =
      tmpl::push_back<tmpl::append<db::split_tag<db::add_tag_prefix<
                                       ::Tags::NormalDotFlux, variables_tag>>,
                                   db::split_tag<variables_tag>>,
                      char_speeds_tag>;

 private:
  template <typename VariablesTagList, typename NormalDoFluxTagList>
  struct package_data_helper;

  template <typename... VariablesTags, typename... NormalDotFluxTags>
  struct package_data_helper<tmpl::list<VariablesTags...>,
                             tmpl::list<NormalDotFluxTags...>> {
    static void function(
        const gsl::not_null<Variables<package_tags>*> packaged_data,
        const db::item_type<NormalDotFluxTags>&... n_dot_f_to_package,
        const db::item_type<VariablesTags>&... u_to_package,
        const db::item_type<char_speeds_tag>& characteristic_speeds) noexcept {
      expand_pack((get<VariablesTags>(*packaged_data) = u_to_package)...);
      expand_pack(
          (get<NormalDotFluxTags>(*packaged_data) = n_dot_f_to_package)...);

      get<MinSignalSpeed>(*packaged_data) = make_with_value<Scalar<DataVector>>(
          characteristic_speeds[0],
          std::numeric_limits<double>::signaling_NaN());
      get<MaxSignalSpeed>(*packaged_data) = make_with_value<Scalar<DataVector>>(
          characteristic_speeds[0],
          std::numeric_limits<double>::signaling_NaN());

      // This finds the min and max characteristic speeds at each grid point,
      // which are used as estimates of the min and max signal speeds.
      for (size_t s = 0; s < characteristic_speeds[0].size(); ++s) {
        // This ensures that local_min_signal_speed <= 0.0
        const double local_min_signal_speed = (*std::min_element(
            characteristic_speeds.begin(), characteristic_speeds.end(),
            [&s](const auto& a, const auto& b) { return a[s] < b[s]; }))[s];
        get(get<MinSignalSpeed>(*packaged_data))[s] =
            std::min(local_min_signal_speed, 0.0);

        // Likewise, local_max_signal_speed >= 0.0
        const double local_max_signal_speed = (*std::max_element(
            characteristic_speeds.begin(), characteristic_speeds.end(),
            [&s](const auto& a, const auto& b) { return a[s] < b[s]; }))[s];
        get(get<MaxSignalSpeed>(*packaged_data))[s] =
            std::max(local_max_signal_speed, 0.0);
      }
    }
  };

  template <typename NormalDotNumericalFluxTagList, typename VariablesTagList,
            typename NormalDotFluxTagList>
  struct call_operator_helper;
  template <typename... NormalDotNumericalFluxTags, typename... VariablesTags,
            typename... NormalDotFluxTags>
  struct call_operator_helper<tmpl::list<NormalDotNumericalFluxTags...>,
                              tmpl::list<VariablesTags...>,
                              tmpl::list<NormalDotFluxTags...>> {
    static void function(
        const gsl::not_null<
            db::item_type<NormalDotNumericalFluxTags>*>... n_dot_numerical_f,
        const db::item_type<NormalDotFluxTags>&... n_dot_f_interior,
        const db::item_type<VariablesTags>&... u_interior,
        const db::item_type<MinSignalSpeed>& min_signal_speed_interior,
        const db::item_type<MaxSignalSpeed>& max_signal_speed_interior,
        const db::item_type<NormalDotFluxTags>&... minus_n_dot_f_exterior,
        const db::item_type<VariablesTags>&... u_exterior,
        const db::item_type<MinSignalSpeed>& min_signal_speed_exterior,
        const db::item_type<MaxSignalSpeed>&
            max_signal_speed_exterior) noexcept {
      auto min_signal_speed = make_with_value<db::item_type<MinSignalSpeed>>(
          min_signal_speed_interior,
          std::numeric_limits<double>::signaling_NaN());
      auto max_signal_speed = make_with_value<db::item_type<MaxSignalSpeed>>(
          max_signal_speed_interior,
          std::numeric_limits<double>::signaling_NaN());
      for (size_t s = 0; s < min_signal_speed.begin()->size(); ++s) {
        get(min_signal_speed)[s] = std::min(get(min_signal_speed_interior)[s],
                                            get(min_signal_speed_exterior)[s]);
        get(max_signal_speed)[s] = std::max(get(max_signal_speed_interior)[s],
                                            get(max_signal_speed_exterior)[s]);
      }
      const DataVector one_over_cp_minus_cm =
          1.0 / (get(max_signal_speed) - get(min_signal_speed));
      const auto assemble_numerical_flux =
          [&min_signal_speed, &max_signal_speed, &one_over_cp_minus_cm ](
              const auto n_dot_num_f, const auto& n_dot_f_in, const auto& u_in,
              const auto& minus_n_dot_f_ex, const auto& u_ex) noexcept {
        for (size_t i = 0; i < n_dot_num_f->size(); ++i) {
          (*n_dot_num_f)[i] = one_over_cp_minus_cm *
                              (get(max_signal_speed) * n_dot_f_in[i] +
                               get(min_signal_speed) * minus_n_dot_f_ex[i] -
                               get(max_signal_speed) * get(min_signal_speed) *
                                   (u_in[i] - u_ex[i]));
        }
        return nullptr;
      };
      expand_pack(assemble_numerical_flux(n_dot_numerical_f, n_dot_f_interior,
                                          u_interior, minus_n_dot_f_exterior,
                                          u_exterior)...);
    }
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {"Computes the HLL numerical flux."};

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  template <class... Args>
  void package_data(const Args&... args) const noexcept {
    package_data_helper<
        db::split_tag<variables_tag>,
        db::split_tag<db::add_tag_prefix<::Tags::NormalDotFlux,
                                         variables_tag>>>::function(args...);
  }

  template <class... Args>
  void operator()(const Args&... args) const noexcept {
    call_operator_helper<
        db::split_tag<
            db::add_tag_prefix<::Tags::NormalDotNumericalFlux, variables_tag>>,
        db::split_tag<variables_tag>,
        db::split_tag<db::add_tag_prefix<::Tags::NormalDotFlux,
                                         variables_tag>>>::function(args...);
  }
};

}  // namespace NumericalFluxes
}  // namespace dg
