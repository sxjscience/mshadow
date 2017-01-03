/*!
*  Copyright (c) 2016 by Contributors
* \file complex_hadamard_product.h
* \brief support for complex conjugate
* \author Xingjian Shi
*/
#ifndef MSHADOW_EXTENSION_COMPLEX_CONJUGATE_H_
#define MSHADOW_EXTENSION_COMPLEX_CONJUGATE_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
  namespace expr {
    /*!
    * \brief Complex hadamard product expression. Calculate res = A .* B where A and B are all complex 4D tensors
    * \tparam SrcExp type of source tensor expression, shape: (N, C, H, W*2)
    * \tparam DType the content data type
    */
    template<typename SrcExp, typename DType>
    struct ComplexConjugateExp :
      public Exp<ComplexConjugateExp<SrcExp, DType>, DType, type::kMapper> {
      /*! \brief source operand */
      const SrcExp &src_;
      /*! \brief constructor */
      ComplexConjugateExp(const SrcExp &src)
        : src_(src) {}
    };


    template<int dim, typename SrcExp, typename DType>
    struct ShapeCheck<dim, ComplexConjugateExp<SrcExp, DType> > {
      inline static Shape<dim>
        Check(const ComplexConjugateExp<SrcExp, DType> &t) {
        CHECK(dim == 4)
          << "ComplexConjugateExp: Dimension of the src tensor must be 4.";
        Shape<4> src_shape = ShapeCheck<4, SrcExp>::Check(t.src_);
        CHECK_EQ(src_shape[3] % 2, 0)
          << "ComplexConjugateExp: Size of the width channel must be even!";
        Shape<4> ret = src_shape;
        return ret;
      }
    };

    /*!
    * \brief complex_conjugate Calculate res = conj(A) where A is a complex 4D tensors
    * \param src left source, shape: (batch, channel, height, width*2)
    * \tparam e1 type of source expression
    */
    template<typename SrcExp, typename DType, int e1>
    inline ComplexConjugateExp<SrcExp, DType>
      complex_conjugate(const Exp<SrcExp, DType, e1> &src) {
      TypeCheckPass<ExpInfo<SrcExp>::kDim == 4>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
      return ComplexConjugateExp<SrcExp, DType>(src.self());
    }


    //----------------------
    // Execution plan
    //----------------------
    template<typename SrcExp, typename DType>
    struct Plan<ComplexConjugateExp<SrcExp, DType>, DType> {
    public:
      explicit Plan(const ComplexConjugateExp<SrcExp, DType> &e)
        : src_(MakePlan(e.src_)) {}
      MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
        using namespace std;
        DType res = src_.Eval(i, j);
        if (j % 2) {
          res = -res;
        }
        return res;
      }

    private:
      expr::Plan<SrcExp, DType> src_;
    };

    template<typename SrcExp, typename DType>
    inline Plan<ComplexConjugateExp<SrcExp, DType>, DType>
      MakePlan(const ComplexConjugateExp<SrcExp, DType> &exp) {
      return Plan<ComplexConjugateExp<SrcExp, DType>, DType>(exp);
    }


    template<typename SrcExp, typename DType>
    struct ExpInfo<ComplexConjugateExp<SrcExp, DType> > {
      static const int kDim = 4;
      static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
    };

  }  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_COMPLEX_CONJUGATE_H_
