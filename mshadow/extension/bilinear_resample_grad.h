/*!
*  Copyright (c) 2016 by Contributors
* \file bilinear_resample.h
* \brief support for backpropagate gradient for bilinear resampling operation
* \author Xingjian Shi
*/
#ifndef MSHADOW_EXTENSION_BILINEAR_RESAMPLE_GRAD_H_
#define MSHADOW_EXTENSION_BILINEAR_RESAMPLE_GRAD_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
  namespace expr {
    /*!
    * \brief Bilinear resampling gradient expression. Take the output gradient and the ROIs as input, generate the sparse gradient w.r.t the input
    * \tparam SrcExp type of out_grad and in_data tensor expression, shape: (N_ROI, C, H_resampled, W_resampled) and (N_batch, C, H, W)
    * \tparam ROIExp type of ROI matrix expression, shape: (N, 5) or (N, 4). If each roi has 5 elements, these elements are (batch_ind, cx, cy, sx, sy). Otherwise, the elements are (cx, cy, sx, sy). All cx, cy, sx, sy have been normalized to be between 0.0f and 1.0f
    * \tparam DType the content data type
    */
    template<typename SrcExp, typename ROIExp, typename DType>
    struct BilinearResamplingGradExp :
      public Exp<BilinearResamplingGradExp<SrcExp, ROIExp, DType>, DType, type::kChainer> {
      /*! \brief out_grad operand */
      const SrcExp &out_grad_;
      /*! \brief in_data operand */
      const SrcExp &in_data_;
      /*! \brief roi operand */
      const ROIExp &roi_;
      /*! \brief in_data height */
      index_t dst_height_;
      /*! \brief in_data width */
      index_t dst_width_;
      /*! \brief out_grad height */
      index_t src_height_;
      /*! \brief out_grad width */
      index_t src_width_;
      /*! \brief channel size of the out_grad & in_data tensor*/
      index_t ch_;
      /*! \brief batch size of the in_data tensor*/
      index_t batch_size_;
      /*! \brief number of rois*/
      index_t nroi_;
      /*! \brief size of each roi*/
      index_t sroi_;
      /*! \brief spatial scale multiplier of sx and sz*/
      DType scale_;
      /*! \brief constructor */
      BilinearResamplingGradExp(const SrcExp &out_grad, const SrcExp &in_data, const ROIExp &roi, DType scale)
        : out_grad_(out_grad), in_data_(in_data), roi_(roi), scale_(scale){
        Shape<4> out_grad_shape = ShapeCheck<4, SrcExp>::Check(out_grad_);
        Shape<4> in_data_shape = ShapeCheck<4, SrcExp>::Check(in_data_);
        Shape<2> roi_shape = ShapeCheck<2, ROIExp>::Check(roi_);
        batch_size_ = in_data_shape[0];
        ch_ = in_data_shape[1];
        dst_height_ = in_data_shape[2];
        dst_width_ = in_data_shape[3];
        src_height_ = out_grad_shape[2];
        src_width_ = out_grad_shape[3];
        nroi_ = roi_shape[0];
        sroi_ = roi_shape[1];
      }
    };


    template<int dim, typename SrcExp, typename ROIExp, typename DType>
    struct ShapeCheck<dim, BilinearResamplingGradExp<SrcExp, ROIExp, DType> > {
      inline static Shape<dim>
        Check(const BilinearResamplingGradExp<SrcExp, ROIExp, DType> &t) {
        CHECK(dim == 4)
          << "BilinearResamplingGradExp: Dimension of the out_grad and in_data tensors must be 4.";
        Shape<4> out_grad_shape = ShapeCheck<4, SrcExp>::Check(t.out_grad_);
        Shape<4> in_data_shape = ShapeCheck<4, SrcExp>::Check(t.in_data_);
        Shape<2> roi_shape = ShapeCheck<2, ROIExp>::Check(t.roi_);
        CHECK(t.scale_ >= 0)
          << "BilinearResamplingGradExp: Scale must be >= 0 !";
        CHECK(t.dst_height_ >= 1 && t.dst_width_ >= 1)
          << "BilinearResamplingGradExp: Invalid in_data dimension detected!";
        CHECK(t.dst_height_ >= 1 && t.dst_width_ >= 1)
          << "BilinearResamplingGradExp: Size of the output tensor must be no smaller than 1X1.";
        CHECK(t.sroi_ == 5 || t.sroi_ == 4)
          << "BilinearResamplingGradExp: ROI must be of shape (N_roi, 5) or (N_batch, 4)";
        if (t.sroi_ == 4) {
          CHECK(t.nroi_ == t.batch_size_)
            << "BilinearResamplingGradExp: Number of ROIs must be equal to the batch size if not explicitly set the batch_ind.";
        }
        printf("%d, %d, %d, %d, %d\n", in_data_shape[0], in_data_shape[1], in_data_shape[2], in_data_shape[3], t.ch_);
        printf("%d, %d, %d, %d, %d\n", out_grad_shape[0], out_grad_shape[1], out_grad_shape[2], out_grad_shape[3], t.ch_);
        CHECK(in_data_shape[1] == out_grad_shape[1])
          << "BilinearResamplingGradExp: The out_grad and in_data must have the same channel numbers.";
        Shape<4> ret;
        ret[0] = t.batch_size_;
        ret[1] = t.ch_;
        ret[2] = t.dst_height_;
        ret[3] = t.dst_width_;
        return ret;
      }
    };

    /*!
    * \brief bilinear_resample_grad Put the gradient back to the input tensor. The gradient is taken with respect to the output after bilinear resampling.
    * \param out_grad the output gradient, shape: (n_roi, channel, height_resampled, width_resampled)
    * \param in_data the output gradient, shape: (batch_size, channel, height, width)
    * \param roi ROI data, shape: (n_roi, 5) or (batch, 4)
    * \param scale multiplier of the size
    * \return expression of in_grad
    * \tparam SrcExp out_grad and in_data expression (two tensors both have ndim = 4)
    * \tparam ROIExp roi expression
    * \tparam DType the source data (also content data and roi data) type
    * \tparam e1 type of SrcExp
    * \tparam e2 type of ROIExp
    */
    template<typename SrcExp, typename ROIExp, typename DType, int e1, int e2>
    inline BilinearResamplingGradExp<SrcExp, ROIExp, DType>
      bilinear_resample_grad(const Exp<SrcExp, DType, e1> &out_grad, const Exp<SrcExp, DType, e1> &in_data, const Exp<ROIExp, DType, e2> &roi, DType scale) {
      TypeCheckPass<ExpInfo<SrcExp>::kDim == 4 && ExpInfo<ROIExp>::kDim == 2>
        ::Error_Expression_Does_Not_Meet_Dimension_Req();
      return BilinearResamplingGradExp<SrcExp, ROIExp, DType>(out_grad.self(), in_data.self(), roi.self(), scale);
    }


    //----------------------
    // Execution plan
    //----------------------
    template<typename SrcExp, typename ROIExp, typename DType>
    struct Plan<BilinearResamplingGradExp<SrcExp, ROIExp, DType>, DType> {
    public:
      explicit Plan(const BilinearResamplingGradExp<SrcExp, ROIExp, DType> &e)
        : out_grad_(MakePlan(e.out_grad_)), in_data_(MakePlan(e.in_data_)), roi_(MakePlan(e.roi_)),
        dst_height_(e.dst_height_), dst_width_(e.dst_width_),
        src_height_(e.src_height_), src_width_(e.src_width_),
        ch_(e.ch_), batch_size_(e.batch_size_),
        nroi_(e.nroi_), sroi_(e.sroi_), scale_(e.scale_) {}
      MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
        using namespace std;
        const DType py = i % dst_height_;
        i /= dst_height_;
        const index_t c = i % ch_;
        const index_t b = i / ch_;
        const DType px = j;
        index_t roi_ind_start, roi_ind_end;
        const DType dst_height_f = static_cast<DType>(dst_height_);
        const DType dst_width_f = static_cast<DType>(dst_width_);
        const DType src_height_f = static_cast<DType>(src_height_);
        const DType src_width_f = static_cast<DType>(src_width_);
        if (sroi_ == 5) {
          roi_ind_start = 0;
          roi_ind_end = nroi_ - 1;
        }
        else {
          roi_ind_start = b;
          roi_ind_end = b;
        }
        DType res = 0;
        for (index_t roi_ind = roi_ind_start; roi_ind <= roi_ind_end; ++roi_ind) {
          index_t roi_b;
          DType roi_cx, roi_cy, roi_sx, roi_sy;
          if (sroi_ == 5){
            roi_b = static_cast<index_t>(roi_.Eval(roi_ind, 0));
            roi_cx = roi_.Eval(b, 1) * dst_width_f;
            roi_cy = roi_.Eval(b, 2) * dst_height_f;
            roi_sx = max(roi_.Eval(b, 3) * dst_width_f * scale_, 1.0f);
            roi_sy = max(roi_.Eval(b, 4) * dst_height_f * scale_, 1.0f);
          }
          else {
            roi_b = b;
            roi_cx = roi_.Eval(b, 0) * dst_width_f;
            roi_cy = roi_.Eval(b, 1) * dst_height_f;
            roi_sx = max(roi_.Eval(b, 2) * dst_width_f * scale_, 1.0f);
            roi_sy = max(roi_.Eval(b, 3) * dst_height_f * scale_, 1.0f);
          }
          
          if (roi_b != b) {
            continue;
          }
          const DType roi_y1 = roi_cy - roi_sy / 2.0;
          const DType roi_x1 = roi_cx - roi_sx / 2.0;
          // For the bilinear interpolation, Out(py, px) = In(y, x) * (1 - dy) * (1 - dx) + In(y + 1, x) * dy * (1 - dx) + In(y, x + 1) * (1 - dy) * dx + In(y, x + 1) * dy * dx
          for (index_t uy = 0; uy <= 1; ++uy) {
            for (index_t ux = 0; ux <= 1; ++ux) {
              const index_t src_y1 = static_cast<index_t>(min(max(floor((py - roi_y1 - static_cast<DType>(uy)) * src_height_f / roi_sy), 0.0f), src_height_f));
              const index_t src_x1 = static_cast<index_t>(min(max(floor((px - roi_x1 - static_cast<DType>(ux)) * src_width_f / roi_sx), 0.0f), src_width_f));
              const index_t src_y2 = static_cast<index_t>(min(max(ceil((py - roi_y1 - static_cast<DType>(uy) + 1.0f) * src_height_f / roi_sy), 0.0f), src_height_f));
              const index_t src_x2 = static_cast<index_t>(min(max(ceil((px - roi_x1 - static_cast<DType>(ux) + 1.0f) * src_width_f / roi_sx), 0.0f), src_width_f));
              printf("ux:%d, uy:%d, px:%g, py:%g, src_height_f:%g, src_width_f:%g, src_y1:%g, src_x1:%g, src_y2:%g, src_x2:%g\n", ux, uy, px, py, src_height_f, src_width_f, src_y1, src_x1, src_y2, src_x2);
              for (index_t x = src_x1; x <= src_x2; ++x) {
                for (index_t y = src_y1; y <= src_y2; ++y) {
                  const DType dst_y = static_cast<DType>(y)* roi_sy / src_height_f + roi_y1;
                  const DType dst_x = static_cast<DType>(x)* roi_sx / src_width_f + roi_x1;
                  if (y >= 0 && y < src_height_ && x >= 0 && x < src_width_
                    && (floor(dst_y) + static_cast<DType>(uy)) == py
                    && (floor(dst_x) + static_cast<DType>(ux)) == px) {
                    //printf("y=%d, x=%d, src_y:%d-%d, src_x:%d-%d, roi_height=%d, roi_width=%d, dst_height=%d, dst_width=%d, px=%d, roi_x_start=%d\n", y, x, src_y_lower, src_y_upper, src_x_lower, src_x_upper,
                    //  roi_height, roi_width, dst_height_i, dst_width_i, px, roi_x_start);
                    const DType dy = dst_y - floor(dst_y);
                    const DType dx = dst_x - floor(dst_x);
                    res += out_grad_.Eval((roi_b * ch_ + c) * src_height_ + y, x) * ((1.0f - dx) * (1.0f - ux) + dx * ux) * ((1.0f - dy) * (1.0f - uy) + dy * uy);
                  }
                }
              }
            }
          }
          
        }
        return res;
      }

    private:
      expr::Plan<SrcExp, DType> out_grad_, in_data_;
      expr::Plan<ROIExp, DType> roi_;
      const index_t dst_height_, dst_width_, src_height_, src_width_;
      const index_t ch_, batch_size_, nroi_, sroi_;
      const DType scale_;
    };

    template<typename SrcExp, typename ROIExp, typename DType>
    inline Plan<BilinearResamplingGradExp<SrcExp, ROIExp, DType>, DType>
      MakePlan(const BilinearResamplingGradExp<SrcExp, ROIExp, DType> &exp) {
      return Plan<BilinearResamplingGradExp<SrcExp, ROIExp, DType>, DType>(exp);
    }


    template<typename SrcExp, typename ROIExp, typename DType>
    struct ExpInfo<BilinearResamplingGradExp<SrcExp, ROIExp, DType> > {
      static const int kDim = 4;
      static const int kDevMask = ExpInfo<SrcExp>::kDevMask & ExpInfo<ROIExp>::kDevMask;
    };

  }  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_POOL_H_
