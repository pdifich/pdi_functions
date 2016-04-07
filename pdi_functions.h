/*
 * PDI_functions
 */
#ifndef PDI_FUNCTIONS_H
#define PDI_FUNCTIONS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

/**Funciones auxiliares
 */
namespace pdi{
	typedef unsigned char byte;

	/**Modifica la matriz para que todos los valores queden dentro del rango,
	los valores fuera del rango se setean al extremo más cercano.
	*/
	void clamp(cv::Mat& mat, double lowerBound, double upperBound);

	/**Imprime información de tipo y tamaño.
	 */
	void info(const cv::Mat &m, std::ostream &out = std::cout);

	/**Imprime valores estadísticos: mínimo, máximo, media, desvío.
	BUG: max, min, funcionan sólo con imágenes de un canal.
	*/
	void stats(const cv::Mat &m, std::ostream &out = std::cout);

	/**Muestra en pantalla la matriz
	 */
	void print(const cv::Mat &image, std::ostream &out = std::cout);

	/**Realiza el intercambio al copiar elemento a elemento
	Deben ser matrices de igual tamaño
	*/
	void swap_copy(cv::Mat &a, cv::Mat &b);

	/**Desplaza la imagen de modo que el píxel central
	ocupe la esquina superior izquierda.
	Usado para visualizar la transformada de Fourier.
	*/
	void centre(cv::Mat &imagen);

	/** Histograma uniforme de una imagen de un canal de 8bits.
	\param mask permite seleccionar una región a procesar
	 */
	cv::Mat histogram(const cv::Mat &image, int levels, const cv::Mat &mask=cv::Mat());

	/**Dibuja un gráfico de líneas en el canvas.
	 * \param data vector con los valores a graficar,
	 * rango [0,1] para flotantess o [0,MAX] para enteros, que se mapean del borde inferior al superior.
	 */
	cv::Mat draw_graph(cv::Mat &canvas, const cv::Mat &data);

	/**Dibuja un gráfico de líneas en el canvas.
	 * wrapper para aceptar std::vector
	 */
	template<class T>
	cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data);

	/**Copia la imagen a una cuyas dimensiones hacen eficiente la fft
	 */
	cv::Mat optimum_size(const cv::Mat &image);

	/**Devuelve la magnitud logarítmica del espectro, centrada.
	Úsese para visualización.
	\param image debe ser de 32F
	*/
	cv::Mat spectrum(const cv::Mat &image);

	/**Devuelve la convolución.
	Kernel centrado, borde constante.
	*/
	cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel);


	/**Concatena las imágenes.
	 * Los tamaños deberían concordar.
	 */
	cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical=true );

	/**Devuelve los mapeos para rgb.
	 * Las matrices son de tipo 8U
	 */
	std::vector<cv::Mat> load_colormap(const char *filename);

	/**Rota la imagen en sentido antihorario
	\param angle en grados
	*/
	cv::Mat rotate(cv::Mat image, double angle);

	cv::Mat dft(cv::Mat image);
	cv::Mat idft(cv::Mat image);

	/**Realiza el filtrado en frecuencia
	\param image matriz 32F, un canal.
	\param filtro de magnitud, descentrado.
	*/
	cv::Mat filter(cv::Mat image, cv::Mat filtro);
}


//Implementation
#include <fstream>
namespace pdi{
	inline void clamp(cv::Mat& mat, double lowerBound, double upperBound){
		cv::min(cv::max(mat, lowerBound), upperBound, mat);
	}

	inline void info(const cv::Mat &image, std::ostream &out){
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

	inline void stats(const cv::Mat &image, std::ostream &out){
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

	inline void print(const cv::Mat &image, std::ostream &out){
		out << image;
	}

	inline void swap_copy(cv::Mat &a, cv::Mat &b){
		cv::Mat temp;
		a.copyTo(temp); //NO puede ser reemplazado por .clone()
		b.copyTo(a);
		temp.copyTo(b);
	}

	inline void centre(cv::Mat &image){
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

	inline cv::Mat histogram(const cv::Mat &image, int levels, const cv::Mat &mask){
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

	inline cv::Mat draw_graph(cv::Mat &canvas, const cv::Mat &data_){
		cv::Mat data = data_;
		switch(data_.depth()){
			case CV_8U: data.convertTo(data, CV_32F, 1./255, 0); break;
			case CV_8S: data.convertTo(data, CV_32F, 1./255, 0.5); break;
			case CV_16U: data.convertTo(data, CV_32F, 1./65535, 0); break;
			case CV_16S: data.convertTo(data, CV_32F, 1./65535, 0.5); break;
			case CV_32S: data.convertTo(data, CV_32F, 1./(2*2147483647u+1), 0.5); break;
			case CV_32F:
				 cv::normalize(data, data, 0, 1, CV_MINMAX);
				 break;
			case CV_64F:
				data.convertTo(data, CV_32F, 1);
				cv::normalize(data, data, 0, 1, CV_MINMAX);
				break;
		}

		for(int K=1; K<std::max(data.rows, data.cols); ++K){
			cv::line(
				canvas,
				cv::Point( K-1, canvas.rows*(1-data.at<float>(K-1)) ),
				cv::Point( K, canvas.rows*(1-data.at<float>(K)) ),
				cv::Scalar::all(255)
			);
		}

		return canvas;
	}

	template<class T>
	inline cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data){
		return draw_graph(canvas, cv::Mat(data));
	}

	inline cv::Mat optimum_size(const cv::Mat &image){
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

	inline cv::Mat spectrum(const cv::Mat &image){
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

	inline cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel){
		cv::Mat result;
		//same bits as the image, kernel centered, no offset
		cv::filter2D(image, result, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
		return result;
	}

	inline cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical ){
		cv::Mat big;
		if(vertical)
			cv::vconcat(a, b, big); //sin documentación
		else
			cv::hconcat(a, b, big); //sin documentación

		return big;
	}

	inline std::vector<cv::Mat> load_colormap(const char *filename){
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

	inline cv::Mat rotate(cv::Mat src, double angle){
		cv::Mat dst;
		cv::Point2f pt(src.cols/2., src.rows/2.);
		cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
		cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
		return dst;
	}

	inline cv::Mat filter(cv::Mat image, cv::Mat filtro_magnitud){
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

#endif /* PDI_FUNCTIONS_H */
