// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
 * \brief Compute the local Lax-Friedrichs numerical flux.
 *
 * Let \f$U\f$ and \f$F^i(U)\f$ be the state vector of the system and its
 * corresponding volume flux, respectively. Let \f$n_i\f$ be the unit normal to
 * the interface. Defining \f$F_n(U) := n_i F^i(U)\f$, and denoting the
 * corresponding projections of the numerical fluxes as \f${F_n}^*(U)\f$, the
 * local Lax-Friedrichs flux for each variable is
 *
 * \f{align*}
 * {F_n}^*(U) = \frac{F_n(U_\text{int}) + F_n(U_\text{ext})}{2} +
 * \frac{\alpha}{2}\left( U_\text{int} - U_\text{ext}\right),
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, respectively,
 * and \f$\alpha\f$ is the maximum eigenvalue of either
 * \f$|\Lambda(U_\text{int}; n_\text{int})|\f$ or
 * \f$|\Lambda(U_\text{ext}; n_\text{ext})|\f$, where
 * \f$\Lambda(U; n)\f$ is the diagonal matrix whose entries are the
 * characteristic speeds of the system.
 */
template <typename System>
struct LocalLaxFriedrichs {
 private:
  using char_speeds_tag = typename System::char_speeds_tag;
  using variables_tag = typename System::variables_tag;

 public:
  // The maximum characteristic speed modulus on one side of the interface.
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "MaxAbsCharSpeed"; }
  };

  using package_tags =
      tmpl::push_back<tmpl::append<db::split_tag<db::add_tag_prefix<
                                       ::Tags::NormalDotFlux, variables_tag>>,
                                   db::split_tag<variables_tag>>,
                      MaxAbsCharSpeed>;

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
    static void apply(
        const gsl::not_null<Variables<package_tags>*> packaged_data,
        const db::item_type<NormalDotFluxTags>&... n_dot_f_to_package,
        const db::item_type<VariablesTags>&... u_to_package,
        const db::item_type<char_speeds_tag>& characteristic_speeds) noexcept {
      expand_pack((get<VariablesTags>(*packaged_data) = u_to_package)...);
      expand_pack(
          (get<NormalDotFluxTags>(*packaged_data) = n_dot_f_to_package)...);
      Scalar<DataVector>& char_speed = get<MaxAbsCharSpeed>(*packaged_data);
      if (get(char_speed).size() != characteristic_speeds[0].size()) {
        get(char_speed) = DataVector(characteristic_speeds[0].size());
      }

      for (size_t s = 0; s < characteristic_speeds[0].size(); ++s) {
        double local_max_speed = 0.0;
        for (size_t u = 0; u < characteristic_speeds.size(); ++u) {
          local_max_speed = std::max(
              local_max_speed, std::abs(gsl::at(characteristic_speeds, u)[s]));
        }
        get(char_speed)[s] = local_max_speed;
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
    static void apply(
        const gsl::not_null<
            db::item_type<NormalDotNumericalFluxTags>*>... n_dot_numerical_f,
        const db::item_type<NormalDotFluxTags>&... n_dot_f_interior,
        const db::item_type<VariablesTags>&... u_interior,
        const db::item_type<MaxAbsCharSpeed>& max_abs_speed_interior,
        const db::item_type<NormalDotFluxTags>&... minus_n_dot_f_exterior,
        const db::item_type<VariablesTags>&... u_exterior,
        const db::item_type<MaxAbsCharSpeed>& max_abs_speed_exterior) noexcept {
      const Scalar<DataVector> max_abs_speed(DataVector(
          max(get(max_abs_speed_interior), get(max_abs_speed_exterior))));
      const auto assemble_numerical_flux = [&max_abs_speed](
          const auto n_dot_num_f, const auto& n_dot_f_in, const auto& u_in,
          const auto& minus_n_dot_f_ex, const auto& u_ex) noexcept {
        for (size_t i = 0; i < n_dot_num_f->size(); ++i) {
          (*n_dot_num_f)[i] = 0.5 * (n_dot_f_in[i] - minus_n_dot_f_ex[i] +
                                     get(max_abs_speed) * (u_in[i] - u_ex[i]));
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
  static constexpr OptionString help = {
      "Computes the local Lax-Friedrichs numerical flux."};

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  template <class... Args>
  void package_data(const Args&... args) const noexcept {
    package_data_helper<
        db::split_tag<variables_tag>,
        db::split_tag<db::add_tag_prefix<::Tags::NormalDotFlux,
                                         variables_tag>>>::apply(args...);
  }

  template <class... Args>
  void operator()(const Args&... args) const noexcept {
    call_operator_helper<
        db::split_tag<
            db::add_tag_prefix<::Tags::NormalDotNumericalFlux, variables_tag>>,
        db::split_tag<variables_tag>,
        db::split_tag<db::add_tag_prefix<::Tags::NormalDotFlux,
                                         variables_tag>>>::apply(args...);
  }
};

}  // namespace NumericalFluxes
}  // namespace dg
