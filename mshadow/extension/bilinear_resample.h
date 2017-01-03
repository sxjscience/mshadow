/*!
 *  Copyright (c) 2016 by Contributors
 * \file bilinear_resample.h
 * \brief support for bilinear resampling
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_BILINEAR_RESAMPLE_H_
#define MSHADOW_EXTENSION_BILINEAR_RESAMPLE_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Bilinear resampling expression. Crop part of the input images and resize them to a given size. The cropping regions are specified by the ROIs. 
          Also, zero padding is used if the ROI goes outside the boundary.
 * \tparam SrcExp type of source tensor expression, shape: (N, C, H, W) 
 * \tparam ROIExp type of ROI matrix expression, shape: (N_R, 5) or (N, 4). If each roi has 5 elements, these elements are (batch_ind, cx, cy, sx, sy). 
                  Otherwise, the elements are (cx, cy, sx, sy). All the cx,cy,sx,sy are noramlized to be between (0.0, 1.0)
 * \tparam DType the content data type
 */
template<typename SrcExp, typename ROIExp, typename DType>
struct BilinearResamplingExp:
  public Exp<BilinearResamplingExp<SrcExp, ROIExp, DType>, DType, type::kChainer> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief roi operand */
  const ROIExp &roi_;
  /*! \brief destination height shape[1] */
  index_t dst_height_;
  /*! \brief destination width shape[0] */
  index_t dst_width_;
  /*! \brief source height shape[1] */
  index_t src_height_;
  /*! \brief source width shape[0] */
  index_t src_width_;
  /*! \brief channel size of the source tensor*/
  index_t ch_;
  /*! \brief batch size of the source tensor*/
  index_t batch_size_;
  /*! \brief number of rois*/
  index_t nroi_;
  /*! \brief size of each roi*/
  index_t sroi_;
  /*! \brief spatial scale multiplier of sx and sz*/
  DType scale_;
  /*! \brief constructor */
  BilinearResamplingExp(const SrcExp &src, const ROIExp &roi,
             index_t dst_height, index_t dst_width, DType scale)
      : src_(src), roi_(roi), dst_height_(dst_height), dst_width_(dst_width), scale_(scale) {
    Shape<4> src_shape = ShapeCheck<4, SrcExp>::Check(src_); 
    Shape<2> roi_shape = ShapeCheck<2, ROIExp>::Check(roi_);
    batch_size_ = src_shape[0];
    ch_ = src_shape[1];
    src_height_ = src_shape[2];
    src_width_ = src_shape[3];
    nroi_ = roi_shape[0];
    sroi_ = roi_shape[1];
  }
};


template<int dim, typename SrcExp, typename ROIExp, typename DType>
struct ShapeCheck<dim, BilinearResamplingExp<SrcExp, ROIExp, DType> > {
  inline static Shape<dim>
  Check(const BilinearResamplingExp<SrcExp, ROIExp, DType> &t) {
    CHECK(dim == 4)
      << "BilinearResamplingExp: Dimension of the src tensor must be 4.";
    Shape<4> src_shape = ShapeCheck<4, SrcExp>::Check(t.src_);
    Shape<2> roi_shape = ShapeCheck<2, ROIExp>::Check(t.roi_);
    CHECK(t.scale_ >= 0)
      << "BilinearResamplingExp: Scale must be >= 0 !";
    CHECK(t.dst_height_ >= 1 && t.dst_width_ >= 1)
      << "BilinearResamplingExp: Size of the new image must be no smaller than 1X1.";
    CHECK(t.sroi_ == 5 || t.sroi_ == 4)
      << "BilinearResamplingExp: ROI must be of shape (N_roi, 5) or (N_batch, 4)";
    if (t.sroi_ == 4) {
      CHECK(t.nroi_ == t.batch_size_)
        << "BilinearResamplingExp: Number of ROIs must be equal to the batch size if not explicitly set the batch_ind.";
    }
    Shape<4> ret;
    ret[0] = t.nroi_;
    ret[1] = t.ch_;
    ret[2] = t.dst_height_;
    ret[3] = t.dst_width_;
    return ret;
  }
};

/*!
 * \brief bilinear_resample Crop part of the input images and resize them to a given size.
 * \param src source image, shape: (batch, channel, height, width)
 * \param roi ROI data, shape: (n_roi, 5) or (batch, 4)
 * \param dst_height height after resampling
 * \param dst_width width after resampling
 * \param scale multiplier of the size
 * \return expression of bilinear interpolation result
 * \tparam SrcExp source expression
 * \tparam ROIExp roi expression
 * \tparam DType the source data (also content data and roi data) type
 * \tparam e1 type of source expression
 * \tparam e2 type of roi expression
 */
template<typename SrcExp, typename ROIExp, typename DType, int e1, int e2>
inline BilinearResamplingExp<SrcExp, ROIExp, DType>
bilinear_resample(const Exp<SrcExp, DType, e1> &src, const Exp<ROIExp, DType, e2> &roi,
     index_t dst_height, index_t dst_width, DType scale) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 4 && ExpInfo<ROIExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return BilinearResamplingExp<SrcExp, ROIExp, DType>(src.self(), roi.self(), dst_height, dst_width, scale);
}


//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename ROIExp, typename DType>
struct Plan<BilinearResamplingExp<SrcExp, ROIExp, DType>, DType> {
 public:
 explicit Plan(const BilinearResamplingExp<SrcExp, ROIExp, DType> &e)
      : src_(MakePlan(e.src_)), roi_(MakePlan(e.roi_)),
        dst_height_(e.dst_height_), dst_width_(e.dst_width_),
        src_height_(e.src_height_), src_width_(e.src_width_),
        ch_(e.ch_), batch_size_(e.batch_size_), 
        nroi_(e.nroi_), sroi_(e.sroi_), scale_(e.scale_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const DType py = static_cast<DType>(i % dst_height_);
    i /= dst_height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const DType px = static_cast<DType>(j);
    const DType dst_height_f = static_cast<DType>(dst_height_);
    const DType dst_width_f = static_cast<DType>(dst_width_);
    const DType src_height_f = static_cast<DType>(src_height_);
    const DType src_width_f = static_cast<DType>(src_width_);
    index_t src_b;
    DType roi_cx, roi_cy, roi_sx, roi_sy;
    if (sroi_ == 5) {
      src_b = static_cast<index_t>(roi_.Eval(b, 0));
      roi_cx = roi_.Eval(b, 1) * src_width_f;
      roi_cy = roi_.Eval(b, 2) * src_height_f;
      roi_sx = max(roi_.Eval(b, 3) * src_width_f * scale_, 1.0f);
      roi_sy = max(roi_.Eval(b, 4) * src_height_f * scale_, 1.0f);
    }
    else {
      src_b = b;
      roi_cx = roi_.Eval(b, 0) * src_width_f;
      roi_cy = roi_.Eval(b, 1) * src_height_f;
      roi_sx = max(roi_.Eval(b, 2) * src_width_f * scale_, 1.0f);
      roi_sy = max(roi_.Eval(b, 3) * src_height_f * scale_, 1.0f);
    }
    const DType roi_y1 = roi_cy - roi_sy / 2.0f;
    const DType roi_x1 = roi_cx - roi_sx / 2.0f;

    // The top-left position of the source image corresponding to the slice position is equal to [floor(py * old_h / new_h) + y0, floor(px * old_w / new_w) + x0]
    const DType src_y = floor(py * roi_sy / dst_height_f + roi_y1);
    const DType src_x = floor(px * roi_sx / dst_width_f + roi_x1);
    const DType dy = py * roi_sy / dst_height_f + roi_y1 - src_y;
    const DType dx = px * roi_sx / dst_width_f + roi_x1 - src_x;
    DType res = 0;

    for (index_t uy = 0; uy <= 1; ++uy) {
      for (index_t ux = 0; ux <= 1; ++ux) {
        const index_t x = static_cast<index_t>(src_x) + ux;
        const index_t y = static_cast<index_t>(src_y) + uy;
        if (y >= 0 && y < src_height_ && x >= 0 && x < src_width_) {
          res += src_.Eval((src_b * ch_ + c) * src_height_ + y, x) * ((1.0f - dx) * (1.0f - ux) + dx * ux) * ((1.0f - dy) * (1.0f - uy) + dy * uy);
        }
      }
    }
    return res;
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  expr::Plan<ROIExp, DType> roi_;
  const index_t dst_height_, dst_width_, src_height_, src_width_;
  const index_t ch_, batch_size_, nroi_, sroi_;
  const DType scale_;
};

template<typename SrcExp, typename ROIExp, typename DType>
inline Plan<BilinearResamplingExp<SrcExp, ROIExp, DType>, DType>
MakePlan(const BilinearResamplingExp<SrcExp, ROIExp, DType> &exp) {
  return Plan<BilinearResamplingExp<SrcExp, ROIExp, DType>, DType>(exp);
}


template<typename SrcExp, typename ROIExp, typename DType>
struct ExpInfo<BilinearResamplingExp<SrcExp, ROIExp, DType> > {
  static const int kDim = 4;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask & ExpInfo<ROIExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_POOL_H_
