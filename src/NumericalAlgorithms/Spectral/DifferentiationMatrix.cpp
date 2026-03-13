// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/BarycentricWeights.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Fourier.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationWeights.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral {
namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct DifferentiationMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // Algorithm 37 in Kopriva, p. 82
    // It is valid for any collocation points and barycentric weights.
    const DataVector& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    Matrix diff_matrix(num_points, num_points);
    if constexpr (BasisType == Spectral::Basis::FiniteDifference) {
      ASSERT(QuadratureType == Spectral::Quadrature::CellCentered,
             "Currently only support cell-centered SBP FD derivatives. Most "
             "likely supporting cell- or vertex-centered just requires "
             "removing this ASSERT and adding tests.");

      // The summation by parts weights come from:
      // arXiv:gr-qc/0512001
      // https://arxiv.org/pdf/gr-qc/0512001.pdf
      //
      // New, efficient, and accurate high order derivative and dissipation
      // operators satisfying summation by parts, and applications in
      // three-dimensional multi-block evolutions
      //
      // by Peter Diener, Ernst Nils Dorband, Erik Schnetter, and Manuel Tiglio
      if (num_points >= 10) {
        // D_{6-5}
        //
        // Left boundary points
        diff_matrix(0, 0) = -2.465354921110524023660777656111276003457;
        diff_matrix(0, 1) = 6.092129526663144141964665936667656020742;
        diff_matrix(0, 2) = -7.730323816657860354911664841669140051855;
        diff_matrix(0, 3) = 6.973765088877147139882219788892186735807;
        diff_matrix(0, 4) = -3.980323816657860354911664841669140051855;
        diff_matrix(0, 5) = 1.292129526663144141964665936667656020742;
        diff_matrix(0, 6) = -0.1820215877771906903274443227779426701237;
        for (size_t i = 7; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        diff_matrix(1, 0) = -0.2234725650784319828746535134412736890421;
        diff_matrix(1, 1) = -0.9329308121107134563129925525068570679651;
        diff_matrix(1, 2) = 1.586820596545839371759081303802027231274;
        diff_matrix(1, 3) = -0.3647002340377160216914505558624668821400;
        diff_matrix(1, 4) = -0.2666957784872806143914117440166232718819;
        diff_matrix(1, 5) = 0.3112949048634705032101261273629794071371;
        diff_matrix(1, 6) = -0.1404504214762266650000768489896480092493;
        diff_matrix(1, 7) = 0.03488568514730479833596013512958238764128;
        diff_matrix(1, 8) = -0.004964021886392518344179263072091597647654;
        diff_matrix(1, 9) = 0.0002126465201465853095969115943714918742904;

        diff_matrix(2, 0) = 0.1582216737061633151406179477554921935333;
        diff_matrix(2, 1) = -1.137049298003377811733609086574457439398;
        diff_matrix(2, 2) = 1.212364522932578587741649981040340946798;
        diff_matrix(2, 3) = -0.9562288729513894906148167047868730813830;
        diff_matrix(2, 4) = 1.066548057336766350478498057851678826640;
        diff_matrix(2, 5) = -0.3478788551267041838265477441805600110467;
        diff_matrix(2, 6) = -0.03133923293520187620333693909408071632123;
        diff_matrix(2, 7) = 0.04098845955755862691072597869183962277781;
        diff_matrix(2, 8) = -0.005963188634687155197078928402509551508436;
        diff_matrix(2, 9) = 0.0003367341182936373038974376991292099082999;

        diff_matrix(3, 0) = 0.02915734641890708196910927068736798144670;
        diff_matrix(3, 1) = -0.1169665089768926152768236581512624861308;
        diff_matrix(3, 2) = -0.1112219092451476301503253995474190870412;
        diff_matrix(3, 3) = -0.7924486261248032107393766820001361351677;
        diff_matrix(3, 4) = 1.266650704820613624987450232358951199911;
        diff_matrix(3, 5) = -0.2899273290506621673153239836530375587273;
        diff_matrix(3, 6) = 0.002515684257201926199329020583484434062150;
        diff_matrix(3, 7) = 0.01329713961871764653006682056620518602804;
        diff_matrix(3, 8) = -0.001124464399630667352932212208930962568134;
        diff_matrix(3, 9) = 0.00006796268169601114882659136477742818715059;

        diff_matrix(4, 0) = -0.04582150000326981674750984653096293434777;
        diff_matrix(4, 1) = 0.2240986548857151482718685516611524323427;
        diff_matrix(4, 2) = -0.3246718493011818141660859125588209338018;
        diff_matrix(4, 3) = -0.3929792921782506986152017485694441380503;
        diff_matrix(4, 4) = 0.1166355818729375628072830916953646214341;
        diff_matrix(4, 5) = 0.3449626905957060254933930895775644438105;
        diff_matrix(4, 6) = 0.1430419813354607083034935179267283951745;
        diff_matrix(4, 7) = -0.07764802499372607792980458731991885121073;
        diff_matrix(4, 8) = 0.01332439335504217034559288889042994978834;
        diff_matrix(4, 9) = -0.0009426355684332077630290447720929851395193;

        diff_matrix(5, 0) = 0.003172814452954821196677290327889903944225;
        diff_matrix(5, 1) = 0.00001061446045061551877105554145609103530766;
        diff_matrix(5, 2) = -0.08747763580209736614983637747947172321794;
        diff_matrix(5, 3) = 0.3975827322299876034907453299884380895682;
        diff_matrix(5, 4) = -1.148835072393422871630425744497391344782;
        diff_matrix(5, 5) = 0.3583006649535242306065761818925080902380;
        diff_matrix(5, 6) = 0.5647665154270147564019144982190032455071;
        diff_matrix(5, 7) = -0.09698196887272109736153117076061707705561;
        diff_matrix(5, 8) = 0.008843905091972988427261446924164441884143;
        diff_matrix(5, 9) = 0.0006174304523363194998474898440202828786385;

        diff_matrix(6, 0) = -0.008639107540858839028043929986084287776394;
        diff_matrix(6, 1) = 0.04722773954485212324714352753530343274219;
        diff_matrix(6, 2) = -0.1008747537650261142294540111407681552350;
        diff_matrix(6, 3) = 0.08043834953845218736895768965086958762389;
        diff_matrix(6, 4) = 0.1295138674713300902982857323205417604553;
        diff_matrix(6, 5) = -0.7909424166489541737614153656634872155367;
        diff_matrix(6, 6) = 0.03807866847647628589685997987877954466259;
        diff_matrix(6, 7) = 0.7367055699548196242687865288427927434250;
        diff_matrix(6, 8) = -0.1480235854665196220062411065981933720158;
        diff_matrix(6, 9) = 0.01651566843542843794512095516024596165494;
        for (size_t i = 1; i < 7; ++i) {
          for (size_t j = 10; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // central points
        for (size_t i = 7; i < num_points - 7; ++i) {
          for (size_t j = 0; j < i - 3; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 3) = -1.0 / 60.0;
          diff_matrix(i, i - 2) = 0.15;
          diff_matrix(i, i - 1) = -0.75;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 0.75;
          diff_matrix(i, i + 2) = -0.15;
          diff_matrix(i, i + 3) = 1.0 / 60.0;
          for (size_t j = i + 4; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite sign.
        for (size_t i = std::max(num_points - 7, 7_st); i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else if (num_points >= 7) {
        for (size_t i = 0; i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }
        // D_{4-3}
        //
        // Left boundary points
        diff_matrix(0, 0) = -2.09329763466349871588733;
        diff_matrix(0, 1) = 4.0398572053206615302160;
        diff_matrix(0, 2) = -3.0597858079809922953240;
        diff_matrix(0, 3) = 1.37319053865399486354933;
        diff_matrix(0, 4) = -0.25996430133016538255400;
        for (size_t i = 5; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        diff_matrix(1, 0) = -0.31641585285940445272297;
        diff_matrix(1, 1) = -0.53930788973980422327388;
        diff_matrix(1, 2) = 0.98517732028644343383297;
        diff_matrix(1, 3) = -0.05264665989297578146709;
        diff_matrix(1, 4) = -0.113807251750624235013258;
        diff_matrix(1, 5) = 0.039879767889849911803103;
        diff_matrix(1, 6) = -0.0028794339334846531588787;

        diff_matrix(2, 0) = 0.13026916185021164524452;
        diff_matrix(2, 1) = -0.87966858995059249256890;
        diff_matrix(2, 2) = 0.38609640961100070000134;
        diff_matrix(2, 3) = 0.31358369072435588745988;
        diff_matrix(2, 4) = 0.085318941913678384633511;
        diff_matrix(2, 5) = -0.039046615792734640274641;
        diff_matrix(2, 6) = 0.0034470016440805155042908;

        diff_matrix(3, 0) = -0.01724512193824647912172;
        diff_matrix(3, 1) = 0.16272288227127504381134;
        diff_matrix(3, 2) = -0.81349810248648813029217;
        diff_matrix(3, 3) = 0.13833269266479833215645;
        diff_matrix(3, 4) = 0.59743854328548053399616;
        diff_matrix(3, 5) = -0.066026434346299887619324;
        diff_matrix(3, 6) = -0.0017244594505194129307249;

        diff_matrix(4, 0) = -0.00883569468552192965061;
        diff_matrix(4, 1) = 0.03056074759203203857284;
        diff_matrix(4, 2) = 0.05021168274530854232278;
        diff_matrix(4, 3) = -0.66307364652444929534068;
        diff_matrix(4, 4) = 0.014878787464005191116088;
        diff_matrix(4, 5) = 0.65882706381707471953820;
        diff_matrix(4, 6) = -0.082568940408449266558615;
        for (size_t i = 1; i < 5; ++i) {
          for (size_t j = 7; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // central points
        //
        // Coded up for completeness and so we can test different (low-order)
        // operators more easily. This won't always be used.
        for (size_t i = 5; i < num_points - 5; ++i) {
          for (size_t j = 0; j < i - 2; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 2) = 1.0 / 12.0;
          diff_matrix(i, i - 1) = -2.0 / 3.0;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 2.0 / 3.0;
          diff_matrix(i, i + 2) = -1.0 / 12.0;
          for (size_t j = i + 3; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite sign.
        for (size_t i = std::max(num_points - 5, 5_st); i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else if (num_points == 6) {
        // D_{4-2}
        //
        // Left boundary points
        diff_matrix(0, 0) = -24.0 / 17.0;
        diff_matrix(0, 1) = 59.0 / 34.0;
        diff_matrix(0, 2) = -4.0 / 17.0;
        diff_matrix(0, 3) = -3.0 / 34.0;
        for (size_t i = 4; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        diff_matrix(1, 0) = -1.0 / 2.0;
        diff_matrix(1, 1) = 0.0;
        diff_matrix(1, 2) = 1.0 / 2.0;
        for (size_t i = 3; i < num_points; ++i) {
          diff_matrix(1, i) = 0.0;
        }

        diff_matrix(2, 0) = 4.0 / 43.0;
        diff_matrix(2, 1) = -59.0 / 86.0;
        diff_matrix(2, 2) = 0.0;
        diff_matrix(2, 3) = 59.0 / 86.0;
        diff_matrix(2, 4) = -4.0 / 43.0;
        for (size_t i = 5; i < num_points; ++i) {
          diff_matrix(2, i) = 0.0;
        }

        diff_matrix(3, 0) = 3.0 / 98.0;
        diff_matrix(3, 1) = 0.0;
        diff_matrix(3, 2) = -59.0 / 98.0;
        diff_matrix(3, 3) = 0.0;
        diff_matrix(3, 4) = 32.0 / 49.0;
        diff_matrix(3, 5) = -4.0 / 49.0;

        // Coded up for completeness and so we can test different (low-order)
        // operators more easily. This won't always be used.
        for (size_t i = 6; i < num_points; ++i) {
          diff_matrix(3, i) = 0.0;
        }

        // central points
        //
        // Coded up for completeness and so we can test different (low-order)
        // operators more easily. This won't always be used.
        for (size_t i = 4; i < num_points - 4; ++i) {
          for (size_t j = 0; j < i - 2; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 2) = 1.0 / 12.0;
          diff_matrix(i, i - 1) = -2.0 / 3.0;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 2.0 / 3.0;
          diff_matrix(i, i + 2) = -1.0 / 12.0;
          for (size_t j = i + 3; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite sign.
        for (size_t i = std::max(num_points - 4, 4_st); i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else if (num_points > 1) {
        // D_{2-1}
        //
        diff_matrix(0, 0) = -1.0;
        diff_matrix(0, 1) = 1.0;
        for (size_t i = 2; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        for (size_t i = 1; i < num_points - 1; ++i) {
          for (size_t j = 0; j < i - 1; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 1) = -0.5;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 0.5;
          for (size_t j = i + 2; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite
        // sign.
        for (size_t i = num_points - 1; i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else {
        diff_matrix(0, 0) = 0.0;
        return diff_matrix;
      }
      const double inv_delta = 1.0 / (collocation_pts[1] - collocation_pts[0]);
      for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < num_points; ++j) {
          diff_matrix(i, j) *= inv_delta;
        }
      }
    } else if constexpr (BasisType == Spectral::Basis::Fourier) {
      diff_matrix = Fourier::differentiation_matrix(num_points);
    } else {
      const DataVector& bary_weights =
          detail::barycentric_weights<BasisType, QuadratureType>(num_points);
      for (size_t i = 0; i < num_points; ++i) {
        double& diagonal = diff_matrix(i, i) = 0.0;
        for (size_t j = 0; j < num_points; ++j) {
          if (LIKELY(i != j)) {
            diff_matrix(i, j) =
                bary_weights[j] /
                (bary_weights[i] * (collocation_pts[i] - collocation_pts[j]));
            diagonal -= diff_matrix(i, j);
          }
        }
      }
    }
    return diff_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct DifferentiationMatrixTransposeGenerator {
  Matrix operator()(const size_t num_points) const {
    Matrix diff_matrix_transpose =
        DifferentiationMatrixGenerator<BasisType, QuadratureType>{}.operator()(
            num_points);
    blaze::transpose(diff_matrix_transpose);
    return diff_matrix_transpose;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct ParityBasedDifferentiationMatrixGenerator {
  Matrix operator()(const size_t num_points, const Parity parity) const {
    // We intentionally do not instantiate "normal" derivatives for Zernike
    // bases due to their large errors
    if constexpr (BasisType == Spectral::Basis::ZernikeB1) {
      return Zernike<1>::differentiation_matrix(num_points, parity);
    } else if constexpr (BasisType == Spectral::Basis::ZernikeB2) {
      return Zernike<2>::differentiation_matrix(num_points, parity);
    } else if constexpr (BasisType == Spectral::Basis::ZernikeB3) {
      return Zernike<3>::differentiation_matrix(num_points, parity);
    } else {
      ERROR(
          "Calling ParityBasedDifferentiationMatrixGenerator with a "
          "non-Zernike basis: call non-parity generator");
    }
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct WeakFluxDifferentiationMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    if (BasisType != Basis::Legendre) {
      // We cannot use a `static_assert` because of the way our instantiation
      // macros work for creating matrices
      ERROR(
          "Weak differentiation matrix only implemented for Legendre "
          "polynomials.");
    }
    const DataVector& weights =
        quadrature_weights<BasisType, QuadratureType>(num_points);

    Matrix weak_diff_matrix =
        DifferentiationMatrixGenerator<BasisType, QuadratureType>{}.operator()(
            num_points);
    transpose(weak_diff_matrix);
    for (size_t i = 0; i < num_points; ++i) {
      for (size_t j = 0; j < num_points; ++j) {
        weak_diff_matrix(i, j) *= weights[j] / weights[i];
      }
    }
    return weak_diff_matrix;
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(differentiation_matrix, Matrix,
                              DifferentiationMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(differentiation_matrix_transpose, Matrix,
                              DifferentiationMatrixTransposeGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(weak_flux_differentiation_matrix, Matrix,
                              WeakFluxDifferentiationMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

PRECOMPUTED_SPECTRAL_QUANTITY_WITH_PARITY(
    differentiation_matrix, Matrix, ParityBasedDifferentiationMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY_WITH_PARITY

SPECTRAL_QUANTITY_FOR_MESH(differentiation_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(differentiation_matrix_transpose, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(weak_flux_differentiation_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const Matrix&
    differentiation_matrix<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const Matrix& differentiation_matrix<
    Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const Matrix&
    differentiation_matrix<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const Matrix&
    differentiation_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix&
    differentiation_matrix<Basis::Fourier, Quadrature::Equiangular>(size_t);
template const Matrix&
    differentiation_matrix<Basis::Legendre, Quadrature::Gauss>(size_t);
template const Matrix&
    differentiation_matrix<Basis::Legendre, Quadrature::GaussLobatto>(size_t);

template const Matrix& differentiation_matrix<
    Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t, Parity);
template const Matrix& differentiation_matrix<
    Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t, Parity);
template const Matrix& differentiation_matrix<
    Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t, Parity);

template const Matrix& weak_flux_differentiation_matrix<
    Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const Matrix& weak_flux_differentiation_matrix<
    Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix& weak_flux_differentiation_matrix<
    Basis::Legendre, Quadrature::Gauss>(size_t);
template const Matrix& weak_flux_differentiation_matrix<
    Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const Matrix& differentiation_matrix<Basis::FiniteDifference,
                                              Quadrature::CellCentered>(size_t);
}  // namespace Spectral
