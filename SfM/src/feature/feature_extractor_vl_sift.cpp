// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "feature_extractor_vl_sift.h"

namespace objectsfm {

VLSiftExtractor::VLSiftExtractor()
{
	imgheight_ = 0;
	imgwidth_  = 0;
	numlevel_  = 5;
	numoctave_ = 4;
	numpt_ = 	 0;
	numdims_ = 0;
}


VLSiftExtractor::~VLSiftExtractor()
{
	delete pimgdata;
}

inline void VLSiftExtractor::transpose_descriptor(vl_sift_pix* dst, vl_sift_pix* src)
{
	int const BO = 8 ;  /* number of orientation bins */
	int const BP = 4 ;  /* number of spatial bins     */
	int i, j, t ;

	for (j = 0 ; j < BP ; ++j) {
		int jp = BP - 1 - j ;
		for (i = 0 ; i < BP ; ++i) {
			int o  = BO * i + BP*BO * j  ;
			int op = BO * i + BP*BO * jp ;
			dst [op] = src[o] ;
			for (t = 1 ; t < BO ; ++t)
				dst [BO - t + op] = src [t + o] ;
		}
	}
}

void VLSiftExtractor::Run(cv::Mat & image, ListKeyPoint* keypoints, cv::Mat* descriptors)
{
	initialize(image.data, image.cols, image.rows);
	run_sift();

	keypoints->pts.resize(numpt_);
	*descriptors = cv::Mat(numpt_, numdims_, CV_32FC1, cv::Scalar(0));
	for (size_t i = 0; i < numpt_; i++)
	{
		keypoints->pts[i].pt.x = keypoints_[i][0];
		keypoints->pts[i].pt.y = keypoints_[i][1];

		float* ptr = (*descriptors).ptr<float>(i);
		for (size_t j = 0; j < numdims_; j++)
		{
			*ptr++ = descriptor_[i][j];
		}
	}
}

bool VLSiftExtractor::run_sift()
{
// parameter definition	
	int                O     =  numoctave_;
	int                S     =  numlevel_ ;
	int                o_min =   0 ;

	double             edge_thresh = 10 ;
	double             peak_thresh = 0 ;
	double             norm_thresh = -1 ;
	double             magnif      = 3 ;
	double             window_size = 2 ;


	VlSiftFilt        *filt ;
    vl_bool            first ;
    double            *frames = 0 ;
    void              *descr  = 0 ;
    int                nframes = 0, reserved = 0, i,j,q ;
	double            *ikeys = 0 ;
	int                nikeys = -1 ;
	vl_bool            force_orientations = 0 ;
	vl_bool            floatDescriptors = 0 ;

    /* create a filter to process the image */
	filt = vl_sift_new (imgwidth_,imgheight_, O, S, o_min) ;

    if (peak_thresh >= 0) vl_sift_set_peak_thresh (filt, peak_thresh) ;
    if (edge_thresh >= 0) vl_sift_set_edge_thresh (filt, edge_thresh) ;
    if (norm_thresh >= 0) vl_sift_set_norm_thresh (filt, norm_thresh) ;
    if (magnif      >= 0) vl_sift_set_magnif      (filt, magnif) ;
    if (window_size >= 0) vl_sift_set_window_size (filt, window_size) ;



    /* ...............................................................
     *                                             Process each octave
     * ............................................................ */
    i     = 0 ;
    first = 1 ;
    while (1) {
      int                   err ;
      VlSiftKeypoint const *keys  = 0 ;
      int                   nkeys = 0 ;

      /* Calculate the GSS for the next octave .................... */
      if (first) {
        err   = vl_sift_process_first_octave (filt, pimgdata) ;
        first = 0 ;
      } else {
        err   = vl_sift_process_next_octave  (filt) ;
      }

      if (err) break ;

      /* Run detector ............................................. */
      if (nikeys < 0) {
        vl_sift_detect (filt) ;

        keys  = vl_sift_get_keypoints  (filt) ;
        nkeys = vl_sift_get_nkeypoints (filt) ;
        i     = 0 ;
      } else {

		  nkeys = nikeys ;
      }

      /* For each keypoint ........................................ */
      for (; i < nkeys ; ++i) {
        double                angles [4] ;
        int                   nangles ;
        VlSiftKeypoint        ik ;
        VlSiftKeypoint const *k ;

        /* Obtain keypoint orientations ........................... */
        if (nikeys >= 0) {
          vl_sift_keypoint_init (filt, &ik,
                                 ikeys [4 * i + 1] - 1,
                                 ikeys [4 * i + 0] - 1,
                                 ikeys [4 * i + 2]) ;

          if (ik.o != vl_sift_get_octave_index (filt)) {
            break ;
          }

          k = &ik ;

          /* optionally compute orientations too */
          if (force_orientations) {
            nangles = vl_sift_calc_keypoint_orientations
              (filt, angles, k) ;
          } else {
            angles [0] = PI / 2 - ikeys [4 * i + 3] ;
            nangles    = 1 ;
          }
        } else {
          k = keys + i ;
          nangles = vl_sift_calc_keypoint_orientations
            (filt, angles, k) ;
        }

        /* For each orientation ................................... */
        for (q = 0 ; q < nangles ; ++q) {
          vl_sift_pix  buf [128] ;
          vl_sift_pix rbuf [128] ;

          /* compute descriptor (if necessary) */
            vl_sift_calc_keypoint_descriptor (filt, buf, k, angles [q]) ;
            transpose_descriptor (rbuf, buf) ;

          /* make enough room for all these keypoints and more */


          /* Save back with MATLAB conventions. Notice tha the input
           * image was the transpose of the actual image. */
          std::vector<float> curkeypoint;
		  curkeypoint.resize(4);
		  curkeypoint[0] = k -> x; 
		  curkeypoint[1] = k->  y;
		  curkeypoint[2] = k->sigma;
		  curkeypoint[3] = PI / 2 - angles [q];
		
		  keypoints_.push_back(curkeypoint);
			
		  std::vector<float> curdescrp;
		  curdescrp.resize(128);
		  for (j = 0 ; j < 128 ; ++j) {
			  curdescrp[j] = 512.0F * rbuf [j] ;
		  }
		 
		  descriptor_.push_back(curdescrp);

          ++ nframes ;
        } /* next orientation */
      } /* next keypoint */
    } /* next octave */

	vl_sift_delete(filt);
	numpt_ = keypoints_.size();
	numdims_ = 128;
	return true;
}

}