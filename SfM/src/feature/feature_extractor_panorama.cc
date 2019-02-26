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

#include "feature_extractor_panorama.h"

#include <omp.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/xfeatures2d.hpp>

namespace objectsfm {


void FeatureExtractorPanorama::Run(cv::Mat &image, std::string method, ListKeyPoint* keypoints, cv::Mat* descriptors, std::string path)
{
	//// convert the panorama image into cyclinder image
	//float fov_y = 90.0;
	//float fov_x = 30;
	//int f = 600;
	//cv::Mat img_cyclinder;
	//Sphere2Cyclinder(image, fov_x, fov_y, f, img_cyclinder);
	//cv::imwrite(path, img_cyclinder);

	//if (method != "SIFT" && method != "sift" && method != "SURF" && method != "surf")
	//{
	//	method = "SIFT";
	//}

	//cv::initModule_nonfree(); //if use SIFT or SURF
	//cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(method);
	//cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create(method);

	//std::vector<cv::KeyPoint> keypoints_cur;
	//cv::Mat descriptors_cur;
	//detector->detect(img_cyclinder, keypoints_cur);
	//descriptor->compute(img_cyclinder, keypoints_cur, descriptors_cur);

	//keypoints->pts = keypoints_cur;
	//*descriptors = descriptors_cur.clone();
}

void FeatureExtractorPanorama::Sphere2Cyclinder(cv::Mat &img_sphere, float fov_x, float fov_y, int f, cv::Mat &img_cyclinder)
{
	int n_zones = int(360.0 / fov_x);

	fov_x /= 180.0 / CV_PI;
	fov_y /= 180.0 / CV_PI;

	int height_zone = 2 * int(tan(fov_y / 2.0)*f);
	int width_zone  = 2 * int(tan(fov_x / 2.0)*f);

	float fov_y_half = fov_y / 2.0;
	float fov_x_half = fov_x / 2.0;

	// pre-calculation
	float xl = cos(fov_y_half)*sin(-fov_x_half);
	float xr = -xl;
	float yy = cos(fov_y_half)*cos(-fov_x_half);
	float zu = sin(fov_y_half);
	float zd = -zu;
	float dx = (xr - xl) / width_zone;
	float dz = (zd - zu) / height_zone;

	std::vector<std::vector<cv::Point2f>> zone_model(height_zone);
	float x = xl, y = yy, z = zu;
	for (size_t i = 0; i < height_zone; i++)
	{
		zone_model[i] = std::vector<cv::Point2f>(width_zone);
		for (size_t j = 0; j < width_zone; j++)
		{
			float N = sqrt(x * x + y * y + z * z);
			float lat = asin(z / N);
			float lon = asin(x / N / cos(lat));

			zone_model[i][j] = cv::Point2f(lon, lat);
			x += dx;
		}
		z += dz;
		x = xl;
	}

	// project each zone onto the plane image
	float kx = img_sphere.cols / (2 * CV_PI);
	float ky = img_sphere.rows / CV_PI;
	img_cyclinder = cv::Mat(height_zone, n_zones * width_zone, CV_8UC3, cv::Scalar(0, 0, 0));
	for (size_t i = 0; i < n_zones; i++)
	{
		for (size_t m = 0; m < height_zone; m++)
		{
			uchar* ptr_cyclinder = img_cyclinder.data + m * 3 * img_cyclinder.cols + i * 3 * width_zone;
			for (size_t n = 0; n < width_zone; n++)
			{
				float lon = (i + 0.5)*fov_x + zone_model[m][n].x;
				float lat = CV_PI / 2.0 - zone_model[m][n].y;
				
				float x_sphere = lon * kx;
				float y_sphere = lat * ky;

				cv::Scalar pixel;
				BilinearInterpolation(img_sphere, x_sphere, y_sphere, pixel);

				ptr_cyclinder[0] = pixel.val[0];
				ptr_cyclinder[1] = pixel.val[1];
				ptr_cyclinder[2] = pixel.val[2];
				ptr_cyclinder += 3;
			}
		}
	}
}

void FeatureExtractorPanorama::BilinearInterpolation(cv::Mat &img_sphere, float x, float y, cv::Scalar &pixel)
{
	int x1 = MAX(int(x), 0);
	int x2 = MIN(x1 + 1, img_sphere.cols);
	float dx = x - x1;

	int y1 = MAX(int(y),0);
	int y2 = MIN(y1 + 1, img_sphere.rows);
	float dy = y - y1;

	uchar* ptr11 = img_sphere.data + 3 * (y1*img_sphere.cols + x1);
	uchar* ptr12 = img_sphere.data + 3 * (y1*img_sphere.cols + x2);
	uchar* ptr21 = img_sphere.data + 3 * (y2*img_sphere.cols + x1);
	uchar* ptr22 = img_sphere.data + 3 * (y2*img_sphere.cols + x2);
	for (size_t i = 0; i < 3; i++)
	{
		pixel.val[i] = ptr11[i] * (1 - dx)*(1 - dy)
			+ ptr12[i] * dx*(1 - dy)
			+ ptr21[i] * (1 - dx)*dy
			+ ptr22[i] * dx*dy;
	}
}

}  // namespace objectsfm
