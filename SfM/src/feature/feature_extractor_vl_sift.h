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

#ifndef OBJECTSFM_FEATURE_EXTRACTOR_VLSIFT_H_
#define OBJECTSFM_FEATURE_EXTRACTOR_VLSIFT_H_

#ifndef PI
#define PI 3.1415926535897932
#endif

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "basic_structs.h"
#include "vl/sift.h"
#include "vl/generic.h"
#include "utils/basic_funcs.h"

namespace objectsfm {

class VLSiftExtractor
{	
public:
	VLSiftExtractor();
	~VLSiftExtractor();
	
	template <class T> bool initialize(T* imgdata, int imgwidth, int imgheight, int numoctave = 4, int numlevel = 5)
	{
		imgwidth_ = imgwidth;
		imgheight_ = imgheight;
		numoctave_ = numoctave;
		numlevel_ = numlevel;

		long long int npixel = imgheight_ * imgwidth_;
		pimgdata = new vl_sift_pix[npixel];

		// here I will do a stretch
		T maxva, minva, rangeva;
		math::get_max_min(imgdata, npixel, maxva, minva);
		rangeva = maxva - minva;
		float fmaxva = maxva;
		float fminva = minva;
		float frangeva = rangeva;

		for (long long int p = 0; p < npixel; p++)
		{
			*(pimgdata + p) = (vl_sift_pix)((float)*(imgdata + p) - fminva) / frangeva * 255;
		}

		// giving an estimate of all key points and descripors
		descriptor_.resize(0);
		keypoints_.resize(0);
		descriptor_.reserve(10000);
		keypoints_.reserve(10000);

		return true;
	}

	void Run(cv::Mat &image, ListKeyPoint* keypoints, cv::Mat* descriptors);

	bool run_sift();

	int numpt_, numdims_;
	std::vector<std::vector<float>> descriptor_; // n x 128 vector
	std::vector<std::vector<float>> keypoints_;  // n x 4 vector

private:
	int imgwidth_;
	int imgheight_;
	int numoctave_;
	int numlevel_;
	vl_sift_pix* pimgdata;

	inline void transpose_descriptor (vl_sift_pix* dst, vl_sift_pix* src);
};

}

#endif // OBJECTSFM_FEATURE_EXTRACTOR_VLSIFT_H_