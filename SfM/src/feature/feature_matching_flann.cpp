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

#include "feature_matching_cuda_sift.h"

namespace objectsfm {

	bool FeatureMatchingCudaSift::Run(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches)
	{
		int ndims = 128;
		double minScore = 0.0;
		double second_best_ratio = 0.5;

		// convert data
		cudaSift::SiftData *siftdata1, *siftdata2;
		DataConvert(kp1, descriptors1, siftdata1);
		DataConvert(kp2, descriptors2, siftdata2);

		// cuda sift matching
		cudaSift::MatchSiftData(*siftdata1, *siftdata2);

		// geo-verification
		std::vector<cv::Point2f> pts1, pts2;
		std::vector<int> match_index;
		cudaSift::SiftPoint *ptr_data1 = siftdata1->h_data;
		cudaSift::SiftPoint *ptr_data2 = siftdata1->h_data;
		for (size_t i = 0; i < siftdata1->numPts; i++)
		{
			if (ptr_data1[i].score < minScore || ptr_data1[i].ambiguity > second_best_ratio)
			{
				continue;
			}

			int idx_match = ptr_data1[i].match;
			pts1.push_back(cv::Point2f(ptr_data1[i].xpos, ptr_data1[i].ypos));
			pts2.push_back(cv::Point2f(ptr_data2[idx_match].xpos, ptr_data2[idx_match].ypos));
			match_index.push_back(i);
		}

		std::vector<int> match_index_inliers;
		if (!GeoVerification(pts1, pts2, match_index_inliers))
		{
			return false;
		}

		for (size_t i = 0; i < match_index_inliers.size(); i++)
		{
			int idx1 = match_index[match_index_inliers[i]];
			int idx2 = ptr_data1[idx1].match;
			matches.push_back(std::pair<int, int>(idx1, idx2));
		}
		return true;
	}

	bool FeatureMatchingCudaSift::Run(cudaSift::SiftData * siftdata1, cudaSift::SiftData * siftdata2, std::vector<std::pair<int, int>> &matches)
	{
		double minScore = 0.0;
		double second_best_ratio = 0.95;

		// cuda sift matching
		cudaSift::MatchSiftData(*siftdata1, *siftdata2);

		// geo-verification
		std::vector<cv::Point2f> pts1, pts2;
		std::vector<int> match_index;
		cudaSift::SiftPoint *ptr_data1 = siftdata1->h_data;
		cudaSift::SiftPoint *ptr_data2 = siftdata2->h_data;
		for (size_t i = 0; i < siftdata1->numPts; i++)
		{
			if (ptr_data1[i].score < minScore || ptr_data1[i].ambiguity > second_best_ratio)
			{
				continue;
			}

			int idx_match = ptr_data1[i].match;
			pts1.push_back(cv::Point2f(ptr_data1[i].xpos, ptr_data1[i].ypos));
			pts2.push_back(cv::Point2f(ptr_data2[idx_match].xpos, ptr_data2[idx_match].ypos));
			match_index.push_back(i);
		}

		std::vector<int> match_index_inliers;
		if (!GeoVerification(pts1, pts2, match_index_inliers))
		{
			return false;
		}
		
		for (size_t i = 0; i < match_index_inliers.size(); i++)
		{
			int idx1 = match_index[match_index_inliers[i]];
			int idx2 = ptr_data1[idx1].match;
			matches.push_back(std::pair<int, int>(idx1, idx2));
		}
		return true;
	}

	void FeatureMatchingCudaSift::DataConvert(std::vector<cv::KeyPoint>& kp, cv::Mat & descriptors, cudaSift::SiftData * data)
	{
		int ndims = 128;

		data->stream = 0;
		cudaSift::InitSiftData(*data, 32768, true, true);

		//
		data->h_KernelParams->d_PointCounter[0] = 0;
		data->h_KernelParams->d_MaxNumPoints = data->maxPts;
		cudaMemcpyAsync(
			data->d_KernelParams->d_PointCounter,
			data->h_KernelParams->d_PointCounter,
			KPARAMS_POINT_COUNTER_SIZE_BYTES,
			cudaMemcpyHostToDevice,
			data->stream
		);
		cudaMemcpyAsync(
			&data->d_KernelParams->d_MaxNumPoints,
			&data->h_KernelParams->d_MaxNumPoints,
			KPARAMS_MAX_NUM_POINTS_SIZE_BYTES,
			cudaMemcpyHostToDevice,
			data->stream
		);

		data->numPts = kp.size();
		cudaSift::SiftPoint* ptr_h_data = data->h_data;
		for (size_t i = 0; i < data->numPts; i++)
		{
			ptr_h_data->xpos = kp[i].pt.x;
			ptr_h_data->ypos = kp[i].pt.y;
			float* ptr_h_data_data = ptr_h_data->data;
			float* ptr_desc = descriptors.ptr<float>(i);
			for (size_t j = 0; j < ndims; j++)
			{
				*ptr_h_data_data++ = *ptr_desc++;
			}
			ptr_h_data++;
		}

		if (data->d_data)
		{
			cudaMemcpyAsync(data->d_data, data->h_data, sizeof(cudaSift::SiftPoint)*data->numPts, cudaMemcpyHostToDevice, data->stream);
			cudaStreamSynchronize(data->stream);
		}
	}

	int FeatureMatchingCudaSift::ImproveHomography(cudaSift::SiftData & data, float * homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
	{
#ifdef MANAGEDMEM
		cudaSift::SiftPoint *mpts = data.m_data;
#else
		if (data.h_data == NULL)
			return 0;
		cudaSift::SiftPoint *mpts = data.h_data;
#endif
		float limit = thresh * thresh;
		int numPts = data.numPts;
		cv::Mat M(8, 8, CV_64FC1);
		cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
		double Y[8];
		for (int i = 0; i<8; i++)
			A.at<double>(i, 0) = homography[i] / homography[8];
		for (int loop = 0; loop<numLoops; loop++) {
			M = cv::Scalar(0.0);
			X = cv::Scalar(0.0);
			for (int i = 0; i<numPts; i++) {
				cudaSift::SiftPoint &pt = mpts[i];
				if (pt.score<minScore || pt.ambiguity>maxAmbiguity)
					continue;
				float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0f;
				float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
				float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
				float err = dx * dx + dy * dy;
				float wei = limit / (err + limit);
				Y[0] = pt.xpos;
				Y[1] = pt.ypos;
				Y[2] = 1.0;
				Y[3] = Y[4] = Y[5] = 0.0;
				Y[6] = -pt.xpos * pt.match_xpos;
				Y[7] = -pt.ypos * pt.match_xpos;
				for (int c = 0; c<8; c++)
					for (int r = 0; r<8; r++)
						M.at<double>(r, c) += (Y[c] * Y[r] * wei);
				X += (cv::Mat(8, 1, CV_64FC1, Y) * pt.match_xpos * wei);
				Y[0] = Y[1] = Y[2] = 0.0;
				Y[3] = pt.xpos;
				Y[4] = pt.ypos;
				Y[5] = 1.0;
				Y[6] = -pt.xpos * pt.match_ypos;
				Y[7] = -pt.ypos * pt.match_ypos;
				for (int c = 0; c<8; c++)
					for (int r = 0; r<8; r++)
						M.at<double>(r, c) += (Y[c] * Y[r] * wei);
				X += (cv::Mat(8, 1, CV_64FC1, Y) * pt.match_ypos * wei);
			}
			cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
		}
		int numfit = 0;
		for (int i = 0; i<numPts; i++) {
			cudaSift::SiftPoint &pt = mpts[i];
			float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0;
			float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
			float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
			float err = dx * dx + dy * dy;
			if (err<limit)
				numfit++;
			pt.match_error = sqrt(err);
		}
		for (int i = 0; i<8; i++)
			homography[i] = A.at<double>(i);
		homography[8] = 1.0f;
		return numfit;
	}

	bool FeatureMatchingCudaSift::GeoVerification(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int> &match_inliers)
	{
		// iter 1
		if (pt1.size() < 20)
		{
			return false;
		}

		cv::Mat HMatrix1 = cv::findHomography(pt1, pt2);
		if (std::abs(HMatrix1.at<double>(0, 0) - 0.995) < 0.01 &&
			std::abs(HMatrix1.at<double>(1, 1) - 0.995) < 0.01 &&
			std::abs(HMatrix1.at<double>(2, 2) - 0.995) < 0.01)
		{
			return false;
		}

		float th_epipolar1 = 3.0;
		std::vector<uchar> ransac_status1(pt1.size());
		cv::findFundamentalMat(pt1, pt2, ransac_status1, cv::FM_RANSAC, th_epipolar1);
		std::vector<cv::Point2f> pt1_, pt2_;
		std::vector<int> index;
		for (size_t i = 0; i < ransac_status1.size(); i++)
		{
			if (ransac_status1[i])
			{
				pt1_.push_back(pt1[i]);
				pt2_.push_back(pt2[i]);
				index.push_back(i);
			}
		}

		// iter 2
		if (pt1_.size() < 20)
		{
			return false;
		}

		cv::Mat HMatrix2 = cv::findHomography(pt1_, pt2_);
		if (std::abs(HMatrix2.at<double>(0, 0) - 0.995) < 0.01 &&
			std::abs(HMatrix2.at<double>(1, 1) - 0.995) < 0.01 &&
			std::abs(HMatrix2.at<double>(2, 2) - 0.995) < 0.01)
		{
			return false;
		}

		float th_epipolar2 = 1.0;
		std::vector<uchar> ransac_status2(pt1_.size());
		cv::findFundamentalMat(pt1_, pt2_, ransac_status2, cv::FM_RANSAC, th_epipolar2);
		for (size_t i = 0; i < ransac_status2.size(); i++)
		{
			if (ransac_status2[i])
			{
				match_inliers.push_back(index[i]);
			}
		}

		if (match_inliers.size() < 20)
		{
			return false;
		}

		return true;
	}


}