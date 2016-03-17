#include "pdi_functions.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>

namespace pdi{
	void clamp(cv::Mat& mat, double lowerBound, double upperBound){
		cv::min(cv::max(mat, lowerBound), upperBound, mat);
	}

	void info(const cv::Mat &image, std::ostream &out){
		out << "Characteristics\n";
		out << "\tSize " << image.rows << 'x' << image.cols << '\n';
		out << "\tChannels " << image.channels() << '\n';
		out << "\tDepth ";
		out << '\t';
		switch(image.depth()){
			case CV_8U: out << "8-bit unsigned integers ( 0..255 )\n"; break;
			case CV_8S: out << "8-bit signed integers ( -128..127 )\n"; break;
			case CV_16U: out << "16-bit unsigned integers ( 0..65535 )\n"; break;
			case CV_16S: out << "16-bit signed integers ( -32768..32767 )\n"; break;
			case CV_32S: out << "32-bit signed integers ( -2147483648..2147483647 )\n"; break;
			case CV_32F: out << "32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )\n"; break;
			case CV_64F: out << "64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )\n"; break;
		}
	}

	void stats(const cv::Mat &image, std::ostream &out){
		double max, min;
		cv::Mat mean, std;
		cv::meanStdDev(image, mean, std);
		cv::minMaxLoc(image, &min, &max);
		out << "Stats\n";
		out << "\tarea " << image.size().area() << '\n';
		out << "\tminimum " << min << '\n';
		out << "\tmaximum " << max << '\n';
		out << "\tmean " << mean << '\n';
		out << "\tstd " << std << '\n';
	}

	void print(const cv::Mat &image, std::ostream &out){
		out << image;
	}

	void swap_copy(cv::Mat &a, cv::Mat &b){
		cv::Mat temp;
		a.copyTo(temp); //NO puede ser reemplazado por .clone()
		b.copyTo(a);
		temp.copyTo(b);
	}

	void centre(cv::Mat &image){
		int cx = image.cols/2, cy = image.rows/2;
		//cuadrantes
		cv::Mat
			top_left = cv::Mat(image, cv::Rect(0,0,cx,cy)),
			top_right = cv::Mat(image, cv::Rect(cx,0,cx,cy)),
			bottom_left = cv::Mat(image, cv::Rect(0,cy,cx,cy)),
			bottom_right = cv::Mat(image, cv::Rect(cx,cy,cx,cy));

		//intercambia los cuadrantes
		swap_copy(top_left, bottom_right);
		swap_copy(top_right, bottom_left);
	}

	cv::Mat histogram(const cv::Mat &image, int levels, const cv::Mat mask){
		const int channels = 0;
		float range[] = {0, 256};
		const float *ranges[] = {range};

		cv::MatND hist;
		cv::calcHist(
			&image, //input
			1, //sólo una imagen
			&channels, //de sólo un canal
			mask, //píxeles a considerar
			hist, //output
			1, //unidimensional
			&levels, //cantidad de cubetas
			ranges //valores límite
		);
		return hist;
	}

	void draw_graph(cv::Mat &canvas, const cv::Mat &data){
		for(int K=1; K<data.rows; ++K){
			cv::line(
				canvas,
				cv::Point( K-1, canvas.rows*(1-data.at<float>(K-1)) ),
				cv::Point( K, canvas.rows*(1-data.at<float>(K)) ),
				cv::Scalar::all(255)
			);
		}
	}

	cv::Mat optimum_size(const cv::Mat &image){
		cv::Mat bigger;
		cv::copyMakeBorder(
			image,
			bigger,
			0, cv::getOptimalDFTSize(image.rows)-image.rows,
			0, cv::getOptimalDFTSize(image.cols)-image.cols,
			cv::BORDER_CONSTANT
		);

		return bigger;
	}

	cv::Mat spectrum(const cv::Mat &image){
		cv::Mat fourier;
		cv::dft(image, fourier, cv::DFT_COMPLEX_OUTPUT);

		//calcula la magnitud
		std::vector<cv::Mat> xy;
		cv::Mat magnitud;
		cv::split(fourier, xy);
		cv::magnitude(xy[0], xy[1], magnitud);

		//logaritmo
		cv::log(magnitud+1, magnitud);
		cv::normalize(magnitud, magnitud, 0, 1, cv::NORM_MINMAX);

		//centrado
		centre(magnitud);

		return magnitud;
	}

	cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel){
		cv::Mat result;
		//same bits as the image, kernel centered, no offset
		cv::filter2D(image, result, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
		return result;
	}

	cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical ){
		cv::Mat big;
		if(vertical)
			cv::vconcat(a, b, big); //sin documentación
		else 
			cv::hconcat(a, b, big); //sin documentación

		return big;
	}

	std::vector<cv::Mat> load_colormap(const char *filename){
		std::ifstream input(filename);
		const size_t size = 256;

		std::vector<cv::Mat> rgb(3);
		for(size_t K=0; K<rgb.size(); ++K)
			rgb[K] = cv::Mat::zeros(1, size, CV_8U);

		for(size_t K=0; K<size; ++K)
			for(size_t L=0; L<3; ++L){
				float color;
				input>>color;
				rgb[L].at<byte>(K) = 0xff*color;
			}

		return rgb;
	}

	cv::Mat rotate(cv::Mat src, double angle)
	{
		cv::Mat dst;
		cv::Point2f pt(src.cols/2., src.rows/2.);    
		cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
		cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
		return dst;
	}

	cv::Mat filter(cv::Mat image, cv::Mat filtro_magnitud){
		//se asume imágenes de 32F y un canal, con tamaño óptimo
		cv::Mat transformada;

		//como la fase es 0 la conversión de polar a cartesiano es directa (magnitud->x, fase->y)
		cv::Mat x[2];
		x[0] = filtro_magnitud.clone();
		x[1] = cv::Mat::zeros(filtro_magnitud.size(), CV_32F);

		cv::Mat filtro;
		cv::merge(x, 2, filtro);

		cv::dft(image, transformada, cv::DFT_COMPLEX_OUTPUT);
		cv::mulSpectrums(transformada, filtro, transformada, cv::DFT_ROWS);

		cv::Mat result;
		cv::idft(transformada, result, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
		return result;
	}

}

