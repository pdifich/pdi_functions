#include <iostream>
#include <opencv2/opencv.hpp>
#include "pdi_functions.h"

void usage(const char *program) {
	std::cerr << program << " imagen\n";
	std::cerr << "Muestra el uso de las funciones de pdi_functions.h\n";
}

int main(int argc, char **argv){
	if(argc==0){
		usage(argv[0]);
	}
	//Información
	cv::Mat imagen = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	pdi::info(imagen);
	pdi::stats(imagen);

	//Histograma y gráficos
	cv::Mat histograma = pdi::histogram(imagen, 16);
	histograma /= imagen.size().area();
	cv::imshow("histograma", pdi::draw_graph(histograma, cv::Scalar(0, 255, 255)));

	//Filtrado en frecuencia
	imagen.convertTo(imagen, CV_32F, 1./255);
	imagen = pdi::optimum_size(imagen);
	cv::Mat filtro = pdi::filter_ideal(imagen.rows, imagen.cols, .1);
	cv::Mat filtrada = pdi::filter(imagen, filtro);

	cv::namedWindow("filtrado", CV_WINDOW_KEEPRATIO);
	cv::imshow("filtrado", pdi::mosaic(imagen, filtrada, false));
	cv::waitKey();
	cv::imshow("filtrado", pdi::mosaic(pdi::spectrum(imagen), pdi::spectrum(filtrada), false));
	cv::waitKey();

	return 0;
}

