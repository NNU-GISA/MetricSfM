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

#include "feature_extractor_cuda_sift.h"

#include "cudaSift/image.h"
#include "cudaSift/sift.h"
#include "cudaSift/utils.h"

namespace objectsfm {

CUDASiftExtractor::CUDASiftExtractor()
{
}

CUDASiftExtractor::~CUDASiftExtractor()
{
}

void CUDASiftExtractor::Run(cv::Mat & image, ListKeyPoint * keypoints, cv::Mat * descriptors)
{
	int devNum = 1;
	image.convertTo(image, CV_32FC1);
	unsigned int w = image.cols;
	unsigned int h = image.rows;

	float initBlur = 1.0f;
	float thresh = 1.5f;
	int noctave = 5; 
	int th_num_pts = 32768;

	cudaSift::InitCuda(devNum);

	cudaSift::SiftData *siftdata = new cudaSift::SiftData;
	cudaSift::InitSiftData(*siftdata, th_num_pts, true, true);
	siftdata->stream = 0;

	cudaSift::Image img;
	img.Allocate(w, h, cudaSift::iAlignUp(w, 128), false, NULL, (float*)image.data);
	img.stream = 0;
	img.Download();

	cudaSift::ExtractSift(*siftdata, img, noctave, initBlur, thresh, 0.0f, false);
	while (siftdata->numPts > th_num_pts)
	{
		thresh *= 2.0;
		cudaSift::ExtractSift(*siftdata, img, noctave, initBlur, thresh, 0.0f, false);
	}

	// convert
	int numpts = siftdata->numPts;
	int numdims = 128;
	keypoints->pts.resize(numpts);
	*descriptors = cv::Mat(numpts, numdims, CV_32FC1, cv::Scalar(0));

	cudaSift::SiftPoint *sift = siftdata->h_data;
	for (size_t i = 0; i < numpts; i++)
	{
		keypoints->pts[i].pt.x = sift[i].xpos;
		keypoints->pts[i].pt.y = sift[i].ypos;

		float* ptr_desp = (*descriptors).ptr<float>(i);
		float* ptr_sift = sift[i].data;
		for (size_t j = 0; j < numdims; j++)
		{
			*ptr_desp++ = *ptr_sift++;
		}
	}

	img.Destroy();
	cudaSift::FreeSiftData(*siftdata);
}

}