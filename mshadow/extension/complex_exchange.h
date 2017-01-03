/*!
*  Copyright (c) 2016 by Contributors
* \file complex_exchange.h
* \brief support for exchanging of the real and imaginary part of a complex number
* \author Xingjian Shi
*/
#ifndef MSHADOW_EXTENSION_COMPLEX_EXCHANGE_H_
#define MSHADOW_EXTENSION_COMPLEX_EXCHANGE_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
  namespace expr {
    /*!
    * \brief Complex exchange expression. Calculate res = b + aj
    * \tparam SrcExp type of source tensor expression, shape: (N, C, H, W*2)
    * \tparam DType the content data type
    */
    template<typename SrcExp, typename DType>
    struct ComplexExchangeExp :
      public Exp<ComplexExchangeExp<SrcExp, DType>, DType, type::kMapper> {
      /*! \brief source operand */
      const SrcExp &src_;
      /*! \brief constructor */
      ComplexExchangeExp(const SrcExp &src)
        : src_(src) {}
    };


    template<int dim, typename SrcExp, typename DType>
    struct ShapeCheck<dim, ComplexExchangeExp<SrcExp, DType> > {
      inline static Shape<dim>
        Check(const ComplexExchangeExp<SrcExp, DType> &t) {
        CHECK(dim == 4)
          << "ComplexExchangeExp: Dimension of the src tensor must be 4.";
        Shape<4> src_shape = ShapeCheck<4, SrcExp>::Check(t.src_);
        CHECK_EQ(src_shape[3] % 2, 0)
          << "ComplexExchangeExp: Size of the width channel must be even!";
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
    inline ComplexExchangeExp<SrcExp, DType>
      complex_exchange(const Exp<SrcExp, DType, e1> &src) {
      TypeCheckPass<ExpInfo<SrcExp>::kDim == 4>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
      return ComplexExchangeExp<SrcExp, DType>(src.self());
    }


    //----------------------
    // Execution plan
    //----------------------
    template<typename SrcExp, typename DType>
    struct Plan<ComplexExchangeExp<SrcExp, DType>, DType> {
    public:
      explicit Plan(const ComplexExchangeExp<SrcExp, DType> &e)
        : src_(MakePlan(e.src_)) {}
      MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
        using namespace std;
        return src_.Eval(i, j - 2*(j%2) + 1);
      }

    private:
      expr::Plan<SrcExp, DType> src_;
    };

    template<typename SrcExp, typename DType>
    inline Plan<ComplexExchangeExp<SrcExp, DType>, DType>
      MakePlan(const ComplexExchangeExp<SrcExp, DType> &exp) {
      return Plan<ComplexExchangeExp<SrcExp, DType>, DType>(exp);
    }


    template<typename SrcExp, typename DType>
    struct ExpInfo<ComplexExchangeExp<SrcExp, DType> > {
      static const int kDim = 4;
      static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
    };

  }  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_COMPLEX_CONJUGATE_H_
