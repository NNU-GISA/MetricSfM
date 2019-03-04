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

#include <iostream>
#include <fstream>
#include "basic_structs.h"
#include "slam_gps.h"

int test(int argc, char *argv[])
{
	std::string fold_path = argv[1];
	std::cout << fold_path << std::endl;

	objectsfm::SLAMGPS slamgps;
	slamgps.Run(fold_path);

	return 1;
}

void convert(std::string file_in, std::string file_out, int idd)
{
	std::ifstream ifstr(file_in);
	std::ofstream ofstr(file_out);
	//ofstr << std::fixed << std::setprecision(8);
	while (1) {
		unsigned char uctmp;
		ifstr.read((char*)&uctmp, sizeof(unsigned char));
		ifstr.putback(uctmp);
		if (uctmp == '#') {
			char buffer[1024];      ifstr.getline(buffer, 1024);
			ofstr << buffer << std::endl;
		}
		else
			break;
	}

	int cnum, pnum;
	ifstr >> cnum >> pnum;
	ofstr << cnum << " " << pnum << std::endl;
	for (int c = 0; c < cnum; ++c) {
		double params[9];
		for (int i = 0; i < 3; ++i)
		{
			ifstr >> params[i];
			ofstr << params[i] << " ";
		}
		ofstr << std::endl;
		for (int i = 0; i < 9; ++i)
		{
			ifstr >> params[i];
			ofstr << params[i] << " ";
		}
		ofstr << std::endl;
		for (int i = 0; i < 3; ++i)
		{
			ifstr >> params[i];
			ofstr << params[i] << " ";
		}
		ofstr << std::endl;
	}

	double x, y, z;
	int r, g, b, num;
	for (int p = 0; p < pnum; ++p) {

		ifstr >> x >> y >> z >> r >> g >> b >> num;
		ofstr << x << " " << y << " " << z << " " << r << " " << g << " " << b << " " << num << std::endl;
		
		for (int i = 0; i < num; ++i) {
			int itmp;      
			ifstr >> itmp;
			ofstr << itmp - idd << " ";
			ifstr >> itmp;
			ofstr << itmp - idd << " ";

			// Based on the bundler version, the number of parameters here
			// are either 1 or 3. Currently, it seems to be 3.
			double dtmp;
			ifstr >> dtmp;
			ofstr << dtmp << " ";
			ifstr >> dtmp;
			ofstr << dtmp << std::endl;
		}
	}
	ifstr.close();
	ofstr.close();
}

void main(void)
{
	//std::string file_in = "F:\\Database\\GoPro\\12_11\\GX010007\\3\\cmvs3\\bundle.rd.out";
	//std::string file_out = "F:\\Database\\GoPro\\12_11\\GX010007\\3\\cmvs3\\bundle2.rd.out";
	//convert(file_in, file_out, 1554);

	objectsfm::SLAMGPS slamgps;

	//std::string fold = "F:\\Database\\GoPro\\11-20\\1114_HERO7_data\\front\\GX020005";
	//std::string fold = "L:\\G.D.A\\data\\GoPro\\data_for_processing\\11_20\\2";
	//std::string fold = "D:\\gopro\\11_20\\old\\2";
	//std::string fold = "F:\\Database\\GoPro\\12_11\\GX010007\\3";
	std::string fold = "F:\\Database\\GoPro\\12_11\\GX010007\\3";
	slamgps.Run(fold);
}	