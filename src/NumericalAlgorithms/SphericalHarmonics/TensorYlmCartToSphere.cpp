// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmCartToSphere.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <complex>
#include <optional>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/SparseMatrixFiller.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/WignerThreeJ.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Numeric.hpp"

namespace ylm::TensorYlm {

namespace {
// Inner loops of the rank-1 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_one(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const size_t l_dest, const int m_dest, const int mj,
                     const std::complex<double> kj, const int sign_ysb,
                     WignerThreeJ& threej_m, WignerThreeJ& threej_s,
                     const CoefficientNormalization coefficient_normalization) {
  const auto add_element = [&filler, &iter_src, &iter_dest, dest_comp_index,
                            src_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  for (size_t l_src = threej_s.l1_min();
       (l_src <= threej_s.l1_max() and l_src <= ell_max); ++l_src) {
    const double sqterm =
        sqrt(0.5 * static_cast<double>((2 * l_dest + 1) * (2 * l_src + 1)));
    const int m_src = m_dest + mj;
    const auto sign_m =
        helpers::sign_m<double>(m_src + m_dest, coefficient_normalization);

    const int sign_m_src = (m_src % 2 == 0 ? 1 : -1);
    const double coef_without_kj = sign_m * sqterm * sign_ysb * sign_m_src *
                                   threej_s(l_src) * threej_m(l_src);

    if (m_src >= 0 and static_cast<size_t>(m_src) <= l_src) {
      // Main term.
      if (kj.imag() == 0) {
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReRe
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.real() * coef_without_kj);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImIm
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(kj.real() * coef_without_kj);
        }
      } else {
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReIm
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(-kj.imag() * coef_without_kj);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImRe
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.imag() * coef_without_kj);
        }
      }
    } else if (m_src <= 0 and static_cast<size_t>(-m_src) <= l_src) {
      if (kj.imag() == 0) {
        // Other term, for negative m_src.
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReRe
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.real() * coef_without_kj * sign_m_src);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImIm
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(-kj.real() * coef_without_kj * sign_m_src);
        }
      } else {
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReIm
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(kj.imag() * coef_without_kj * sign_m_src);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImRe
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.imag() * coef_without_kj * sign_m_src);
        }
      }
    }
  }
}

// Inner loops of the rank-2 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_two(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const size_t l_dest, const int m_dest, const size_t ltilde,
                     const std::vector<int>& mtildes,
                     std::vector<std::optional<WignerThreeJ>>& threej_mtildes,
                     WignerThreeJ& threej_ells_stilde, const double symm_factor,
                     const std::array<helpers::BasisVector, 2>& src_bvs,
                     const int sign_ysb, std::vector<WignerThreeJ>& threej_uvs,
                     const double threej_ones_stilde_val,
                     const double ltilde_term, const int sign_stilde,
                     const int sign_m_dest,
                     const CoefficientNormalization coefficient_normalization) {
  const auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                            dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  size_t mtilde_indx = 0;
  for (int u = -1; u <= 1; u += 2) {
    for (int v = -1; v <= 1; v += 2, ++mtilde_indx) {
      if (static_cast<size_t>(std::abs(mtildes[mtilde_indx])) <= ltilde) {
        const int m_src = m_dest - mtildes[mtilde_indx];
        const auto sign_m =
            helpers::sign_m<double>(m_src + m_dest, coefficient_normalization);
        const int sign_m_src = (m_src % 2 == 0 ? 1 : -1);
        const std::complex<double> kj =
            helpers::bv_to_k(src_bvs[0], u) * helpers::bv_to_k(src_bvs[1], v);
        for (size_t l_src = static_cast<size_t>(std::max(
                 static_cast<size_t>(abs(m_src)), threej_ells_stilde.l1_min()));
             (l_src <= threej_ells_stilde.l1_max() and l_src <= ell_max);
             ++l_src) {
          const double sqterm =
              0.5 *
              sqrt(static_cast<double>((2 * l_dest + 1) * (2 * l_src + 1)));
          const double coef_without_kj =
              sign_m * sign_ysb * ltilde_term * sqterm * sign_m_dest *
              sign_stilde * threej_mtildes[mtilde_indx].value()(l_src) *
              threej_uvs[mtilde_indx](ltilde) * threej_ells_stilde(l_src) *
              threej_ones_stilde_val * symm_factor;

          if (m_src >= 0) {
            // Main term.
            if (kj.imag() == 0) {  // This works because kj is either pure
                                   // real or pure imaginary
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReRe
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.real() * coef_without_kj);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImIm
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(kj.real() * coef_without_kj);
              }
            } else {
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReIm
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(-kj.imag() * coef_without_kj);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImRe
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.imag() * coef_without_kj);
              }
            }
          } else {
            if (kj.imag() == 0) {
              // Other term, for negative m_src.
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReRe
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.real() * coef_without_kj * sign_m_src);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImIm
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(-kj.real() * coef_without_kj * sign_m_src);
              }
            } else {
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReIm
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(kj.imag() * coef_without_kj * sign_m_src);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImRe
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.imag() * coef_without_kj * sign_m_src);
              }
            }
          }
        }
      }
    }
  }
}

// Inner loops of the rank-3 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
template <typename Symm>
void inner_loops_three(
    SparseMatrixFiller& filler, SpherepackIterator& iter_src,
    SpherepackIterator& iter_dest, const size_t src_comp_index,
    const size_t dest_comp_index, const size_t ell_max, const size_t l_dest,
    const int m_dest, const size_t lhat, const int mhat, const int shat,
    const int stilde, std::optional<WignerThreeJ>& threej_ones_stilde,
    WignerThreeJ& threej_ells_stot, WignerThreeJ& threej_ems,
    const std::vector<int>& mtildes,
    const std::array<helpers::BasisVector, 3>& src_bvs,
    const size_t src_multiplicity,
    std::vector<std::optional<WignerThreeJ>>& threej_ells_stildes,
    std::vector<WignerThreeJ>& threej_uvs,
    std::vector<std::optional<WignerThreeJ>>& threej_ws, const int sign_ysb,
    const int sign_m_dest,
    const CoefficientNormalization coefficient_normalization) {
  const auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                            dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  size_t mtilde_indx = 0;
  for (int u = -1; u <= 1; u += 2) {
    for (int v = -1; v <= 1; v += 2, ++mtilde_indx) {
      const int sign_stilde =
          ((stilde + shat + mtildes[mtilde_indx]) % 2 == 0 ? 1 : -1);
      const std::complex<double> kj12 =
          helpers::bv_to_k(src_bvs[1], u) * helpers::bv_to_k(src_bvs[2], v);
      for (size_t ltilde = static_cast<size_t>(std::max(
               std::abs(mtildes[mtilde_indx]), std::max(0, abs(stilde))));
           ltilde <= 2; ++ltilde) {
        const double threej_ones_stilde_val =
            threej_ones_stilde.value()(ltilde);
        const double threej_uv_val = threej_uvs[mtilde_indx](ltilde);
        const int ltilde_term =
            static_cast<int>((2 * ltilde + 1) * (2 * lhat + 1));
        const double threej_ells_stilde_val =
            threej_ells_stildes[ltilde].value()(lhat);

        const double symm_factor =
            helpers::get_symm_factor<Symm>(src_multiplicity, ltilde);
        if (symm_factor != 0.0 and threej_ells_stilde_val != 0.0 and
            threej_uv_val != 0.0 and threej_ones_stilde_val != 0.0) {
          for (int w = -1; w <= 1; w += 2) {
            const int mw = helpers::bv_to_m(src_bvs[0], w);
            if (mhat == mtildes[mtilde_indx] - mw) {
              const int m_src = m_dest - mhat;
              const auto sign_m = helpers::sign_m<double>(
                  m_src + m_dest, coefficient_normalization);
              const int sign_m_src = (m_src % 2 == 0 ? 1 : -1);
              const std::complex<double> kj =
                  helpers::bv_to_k(src_bvs[0], w) * kj12;
              const double threej_w_val =
                  threej_ws[ltilde + 3 * static_cast<size_t>((w + 1) / 2) +
                            6 * mtilde_indx]
                      .value()(lhat);

              for (size_t l_src =
                       std::max(threej_ells_stot.l1_min(),
                                std::max(static_cast<size_t>(std::abs(m_src)),
                                         threej_ems.l1_min()));
                   (l_src <= std::min(threej_ems.l1_max(),
                                      threej_ells_stot.l1_max()) and
                    l_src <= ell_max);
                   ++l_src) {
                const double threej_ells_stot_val = threej_ells_stot(l_src);
                const double threej_ems_val = threej_ems(l_src);

                const double sqterm = sqrt(
                    static_cast<double>((2 * l_dest + 1) * (2 * l_src + 1)) /
                    8.0);

                const double coef_without_kj =
                    sign_m * sign_ysb * sign_m_dest * sign_stilde *
                    ltilde_term * sqterm * threej_uv_val *
                    threej_ones_stilde_val * threej_w_val *
                    threej_ells_stilde_val * threej_ells_stot_val *
                    threej_ems_val * symm_factor;

                if (m_src >= 0) {
                  // Main term.
                  if (kj.imag() == 0) {  // This works because kj is either
                                         // pure real or pure imaginary
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReRe
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.real() * coef_without_kj);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImIm
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(kj.real() * coef_without_kj);
                    }
                  } else {
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReIm
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(-kj.imag() * coef_without_kj);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImRe
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.imag() * coef_without_kj);
                    }
                  }
                } else {
                  if (kj.imag() == 0) {
                    // Other term, for negative m_src.
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReRe
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.real() * coef_without_kj * sign_m_src);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImIm
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(-kj.real() * coef_without_kj * sign_m_src);
                    }
                  } else {
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReIm
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(kj.imag() * coef_without_kj * sign_m_src);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImRe
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.imag() * coef_without_kj * sign_m_src);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace

template <typename TensorStructure, typename SparseMatrixType>
void fill_cart_to_sphere(
    const gsl::not_null<SparseMatrixType*> matrix, const size_t ell_max,
    const CoefficientNormalization coefficient_normalization) {
  static constexpr size_t num_independent_components = TensorStructure::size();
  static constexpr size_t rank = TensorStructure::rank();
  static constexpr auto tensor_index_list =
      TensorStructure::storage_to_tensor_index();

  static_assert(rank > 0 and rank < 4, "Implemented only for ranks 1,2,3");

  SpherepackIterator iter_src(ell_max, ell_max, 1, false);
  SpherepackIterator iter_dest(ell_max, ell_max, 1, false);
  SparseMatrixFiller filler(square(num_independent_components) *
                                iter_src.spherepack_array_size() *
                                iter_dest.spherepack_array_size(),
                            true, 1.0);

  // We allocate space for some (not all) of the 3J symbols here, so
  // that they can be used and re-used later inside the inner loops.
  std::vector<WignerThreeJ> threej_uvs;  // NOLINT(misc-const-correctness)
  std::vector<int> mtildes;              // NOLINT(misc-const-correctness)
  if constexpr (rank > 1) {
    threej_uvs.reserve(4);
    mtildes.reserve(4);
  } else {
    // For rank 1 we don't need threej_uvs but we need to declare them
    // anyway for scoping; they will just be unused. We will compute
    // the rank-1 3J symbols below, and below we will also compute
    // additional 3J symbols for higher ranks.
    (void)threej_uvs;
    (void)mtildes;
  }

  // threej_ws contains std::optionals because some of the Wigner 3Js
  // are not well-defined (because |m| > ell).  The ones without
  // values are never used or referenced.  It is possible to compute
  // fewer than 24 threej_ws, by pushing_back only those that have
  // values, but then it is difficult to index the threej_ws.  So
  // keeping 24 of them and making them std::optionals is easier.
  //
  // Same thing is true for threej_ells_stildes.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_ws;
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_ells_stildes;
  if constexpr (rank > 2) {
    threej_ws.reserve(24);
    threej_ells_stildes.reserve(3);
  } else {
    // Unneeded except for rank 3.
    (void)threej_ws;
    (void)threej_ells_stildes;
  }

  // Same thing for threej_mtildes, but for rank 2.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_mtildes;
  if constexpr (rank == 2) {
    threej_mtildes.reserve(4);
  } else {
    // Unneeded except for rank 2.
    (void)threej_mtildes;
  }

  for (size_t dest_comp_index = 0; dest_comp_index < num_independent_components;
       dest_comp_index++) {
    const auto dest_indices = tensor_index_list[dest_comp_index];
    const auto dest_bvs = helpers::to_sphere_basis_vector(dest_indices);
    const int sign_sb =
        alg::accumulate(dest_bvs, 1, [](const double acc, const auto bv) {
          return acc * (helpers::bv_to_s(bv) == -1 ? -1 : 1);
        });

    // threej_ones_stilde is a std::optional because it is
    // defined/used only for rank > 1.
    //
    // threej_ones_stilde is the last 3J symbol in Eq. (21) and the
    // fourth 3J symbol in Eq. (23)
    // NOLINTNEXTLINE(misc-const-correctness)
    int stilde = 0;
    // NOLINTNEXTLINE(misc-const-correctness)
    std::optional<WignerThreeJ> threej_ones_stilde;
    if constexpr (rank > 1) {
      stilde = helpers::bv_to_s(dest_bvs[rank - 1]) +
               helpers::bv_to_s(dest_bvs[rank - 2]);
      threej_ones_stilde =
          WignerThreeJ(1, -helpers::bv_to_s(dest_bvs[rank - 2]), 1,
                       -helpers::bv_to_s(dest_bvs[rank - 1]));
    } else {
      (void)stilde;
      (void)threej_ones_stilde;
    }

    // shat is used only for rank 3.
    const int shat = stilde + helpers::bv_to_s(dest_bvs[0]);

    for (size_t src_comp_index = 0; src_comp_index < num_independent_components;
         src_comp_index++) {
      const auto src_indices = tensor_index_list[src_comp_index];
      const auto src_bvs = helpers::to_cart_basis_vector(src_indices);
      const size_t src_multiplicity =
          TensorStructure::multiplicity(src_comp_index);

      const int sign_y =
          alg::accumulate(src_bvs, 1, [](const double acc, const auto bv) {
            return acc * (bv == helpers::BasisVector::y ? -1 : 1);
          });

      // This is the overall factor of (-1)^bla in Eqs. (20), (21), and (22),
      // where bla are the things that depend only on the tensor component
      // (\tilde(D) and B) and not on l,m.
      const int sign_ysb = sign_y * sign_sb;

      if constexpr (rank > 1) {
        // threej_uvs is the penultimate 3J symbol in Eq. (21)
        // and the third 3J symbol in Eq. (23)
        threej_uvs.clear();
        mtildes.clear();
        for (int u = -1; u <= 1; u += 2) {
          for (int v = -1; v <= 1; v += 2) {
            mtildes.push_back(-(helpers::bv_to_m(src_bvs[rank - 2], u) +
                                helpers::bv_to_m(src_bvs[rank - 1], v)));
            threej_uvs.emplace_back(1, -helpers::bv_to_m(src_bvs[rank - 2], u),
                                    1, -helpers::bv_to_m(src_bvs[rank - 1], v));
          }
        }
      }

      if constexpr (rank > 2) {
        // threej_ws is the penultimate 3J symbol in Eq. (23)
        threej_ws.clear();
        threej_ells_stildes.clear();
        for (const int mtilde : mtildes) {
          for (int w = -1; w <= 1; w += 2) {
            const int mw = helpers::bv_to_m(src_bvs[0], w);
            for (size_t ltilde = 0; ltilde <= 2; ++ltilde) {
              if (abs(mtilde) <= static_cast<int>(ltilde)) {
                threej_ws.emplace_back(WignerThreeJ(1, -mw, ltilde, mtilde));
              } else {
                threej_ws.emplace_back(std::nullopt);
              }
            }
          }
        }
        // threej_ells_stildes is the last 3J symbol in Eq. (23)
        for (size_t ltilde = 0; ltilde <= 2; ++ltilde) {
          if (static_cast<size_t>(abs(stilde)) <= ltilde) {
            threej_ells_stildes.emplace_back(WignerThreeJ(
                1, -helpers::bv_to_s(dest_bvs[0]), ltilde, -stilde));
          } else {
            threej_ells_stildes.emplace_back(std::nullopt);
          }
        }
      }

      for (iter_dest.reset(); iter_dest; ++iter_dest) {
        const auto l_dest = static_cast<size_t>(iter_dest.l());
        const auto m_dest = static_cast<int>(iter_dest.m());
        const int sign_m_dest = (m_dest % 2 == 0 ? 1 : -1);
        if constexpr (rank == 1) {
          (void)sign_m_dest;
          (void)src_multiplicity;
          const int sb = helpers::bv_to_s(dest_bvs[0]);
          if (static_cast<size_t>(std::abs(sb)) <= l_dest) {
            // threej_s is the first 3j symbol in Eq. (20)
            WignerThreeJ threej_s(l_dest, sb, 1, -sb);
            for (int j = -1; j <= 1; j += 2) {
              const int mj = helpers::bv_to_m(src_bvs[0], j);
              // threej_m is the second 3j symbol in Eq. (20)
              WignerThreeJ threej_m(l_dest, -m_dest, 1, -mj);
              const std::complex<double> kj = helpers::bv_to_k(src_bvs[0], j);
              inner_loops_one(filler, iter_src, iter_dest, src_comp_index,
                              dest_comp_index, ell_max, l_dest, m_dest, mj, kj,
                              sign_ysb, threej_m, threej_s,
                              coefficient_normalization);
            }
          }
        } else if constexpr (rank == 2) {
          const int sign_stilde = (stilde % 2 == 0 ? 1 : -1);
          for (size_t ltilde = 0; ltilde <= 2; ++ltilde) {
            const double symm_factor =
                helpers::get_symm_factor<typename TensorStructure::symmetry>(
                    src_multiplicity, ltilde);
            if (symm_factor != 0.0 and
                static_cast<size_t>(std::abs(stilde)) <= ltilde and
                static_cast<size_t>(std::abs(stilde)) <= l_dest) {
              const double threej_ones_stilde_val =
                  threej_ones_stilde.value()(ltilde);
              const auto ltilde_term = static_cast<double>(2 * ltilde + 1);
              // threej_mtildes is the first 3J symbol in Eq. (21)
              threej_mtildes.clear();
              for (const int mtilde : mtildes) {
                if (static_cast<size_t>(std::abs(mtilde)) <= ltilde) {
                  threej_mtildes.emplace_back(
                      WignerThreeJ(l_dest, -m_dest, ltilde, mtilde));
                } else {
                  threej_mtildes.emplace_back(std::nullopt);
                }
              }
              // threej_ells_stilde is the second 3J symbol in Eq. (21)
              WignerThreeJ threej_ells_stilde(l_dest, stilde, ltilde, -stilde);
              inner_loops_two(filler, iter_src, iter_dest, src_comp_index,
                              dest_comp_index, ell_max, l_dest, m_dest, ltilde,
                              mtildes, threej_mtildes, threej_ells_stilde,
                              symm_factor, src_bvs, sign_ysb, threej_uvs,
                              threej_ones_stilde_val, ltilde_term, sign_stilde,
                              sign_m_dest, coefficient_normalization);
            }
          }
        } else if constexpr (rank == 3) {
          const int stot = helpers::bv_to_s(dest_bvs[0]) +
                           helpers::bv_to_s(dest_bvs[1]) +
                           helpers::bv_to_s(dest_bvs[2]);
          if (static_cast<size_t>(std::abs(stot)) <= l_dest) {
            for (size_t lhat = static_cast<size_t>(std::max(0, std::abs(shat)));
                 lhat <= 3; ++lhat) {
              // threej_ells_stot is the second 3J symbol in Eq. (23)
              WignerThreeJ threej_ells_stot(lhat, -shat, l_dest, stot);
              for (int mhat = -static_cast<int>(lhat);
                   mhat <= static_cast<int>(lhat); ++mhat) {
                // threej_ems is the first 3J symbol in Eq. (23)
                WignerThreeJ threej_ems(lhat, mhat, l_dest, -m_dest);
                inner_loops_three<typename TensorStructure::symmetry>(
                    filler, iter_src, iter_dest, src_comp_index,
                    dest_comp_index, ell_max, l_dest, m_dest, lhat, mhat, shat,
                    stilde, threej_ones_stilde, threej_ells_stot, threej_ems,
                    mtildes, src_bvs, src_multiplicity, threej_ells_stildes,
                    threej_uvs, threej_ws, sign_ysb, sign_m_dest,
                    coefficient_normalization);
              }
            }
          }
        }
      }
    }
  }
  filler.fill(matrix);
}

// Explicit instantiations
#define TSTRUCT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void fill_cart_to_sphere<typename TSTRUCT(data) < DataVector, \
                                    3>::structure >                      \
      (gsl::not_null<SimpleSparseMatrix*> matrix, size_t ell_max,        \
       CoefficientNormalization coefficient_normalization);              \
  template void fill_cart_to_sphere<typename TSTRUCT(data) < DataVector, \
                                    3>::structure >                      \
      (gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*>  \
           matrix,                                                       \
       size_t ell_max, CoefficientNormalization coefficient_normalization);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (tnsr::i, tnsr::ii, tnsr::ij, tnsr::ijk, tnsr::ijj))

#undef INSTANTIATE
#undef TSTRUCT

}  // namespace ylm::TensorYlm
