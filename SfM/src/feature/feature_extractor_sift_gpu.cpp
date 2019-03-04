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

#include "feature_extractor_sift_gpu.h"

#include "siftgpu/SiftGPU.h"
#include "gl/glew.h"

namespace objectsfm {

	SIFTGPUExtractor::SIFTGPUExtractor()
	{
	}

	SIFTGPUExtractor::~SIFTGPUExtractor()
	{
	}

	void SIFTGPUExtractor::Run(std::string img_path, ListKeyPoint * keypoints, cv::Mat * descriptors)
	{
		SiftGPU  *sift = new SiftGPU;
		if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
			return;
		}
		sift->RunSIFT(img_path.c_str());
		int numpts = sift->GetFeatureNum();
		int numdims = 128;

		std::vector<float> sift_descriptors(numdims * numpts);
		std::vector<SiftGPU::SiftKeypoint> sift_keys(numpts);
		sift->GetFeatureVector(&sift_keys[0], &sift_descriptors[0]);

		// convert
		keypoints->pts.resize(numpts);
		*descriptors = cv::Mat(numpts, numdims, CV_32FC1, cv::Scalar(0));
		float* ptr_desp = (float*)(*descriptors).data;
		float* ptr_sift = &sift_descriptors[0];
		for (size_t i = 0; i < numpts; i++)
		{
			keypoints->pts[i].pt.x = sift_keys[i].x;
			keypoints->pts[i].pt.y = sift_keys[i].y;

			for (size_t j = 0; j < numdims; j++)
			{
				*ptr_desp++ = *ptr_sift++;
			}
		}
		delete sift;
	}

	void SIFTGPUExtractor::Run(cv::Mat & image, ListKeyPoint * keypoints, cv::Mat * descriptors)
	{
		SiftGPU  *sift = new SiftGPU;
		if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
			return;
		}
		sift->_dog_level_num = 5;

		sift->RunSIFT(image.cols, image.rows, (void*)image.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
		int numpts = sift->GetFeatureNum();
		int numdims = 128;

		std::vector<float> sift_descriptors(numdims * numpts);
		std::vector<SiftGPU::SiftKeypoint> sift_keys(numpts);
		sift->GetFeatureVector(&sift_keys[0], &sift_descriptors[0]);

		// convert
		keypoints->pts.resize(numpts);
		*descriptors = cv::Mat(numpts, numdims, CV_32FC1, cv::Scalar(0));
		float* ptr_desp = (float*)(*descriptors).data;
		float* ptr_sift = &sift_descriptors[0];
		for (size_t i = 0; i < numpts; i++)
		{
			keypoints->pts[i].pt.x = sift_keys[i].x;
			keypoints->pts[i].pt.y = sift_keys[i].y;

			for (size_t j = 0; j < numdims; j++)
			{
				*ptr_desp++ = *ptr_sift++;
			}
		}
		delete sift;

	}

}