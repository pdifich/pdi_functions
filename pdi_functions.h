/*
 * PDI_functions
 */
#ifndef PDI_FUNCTIONS_H
#define PDI_FUNCTIONS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

/**Funciones auxiliares C++
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

	/**Desplaza la imagen de modo que el la componente f0 ocupe el píxel central
	Usado para visualizar la transformada de Fourier.
	*/
	cv::Mat fft_shift(const cv::Mat &image);
	/**Función inversa a fft_shift, equivalente a ésta si la dimensión es par
	 * Utilizado en la creación de filtros
	 */
	cv::Mat ifft_shift(const cv::Mat &image);

	/**Igual que fft_shift
	*/
	void centre(cv::Mat &imagen);

	/** Histograma uniforme de una imagen de un canal de 8bits.
	\param mask permite seleccionar una región a procesar
	 */
	cv::Mat histogram(const cv::Mat &image, int levels, const cv::Mat &mask=cv::Mat());

	/**Dibuja un gráfico de líneas en el canvas.
	 * \param data vector con los valores a graficar,
	 * rango [0,1] para flotantes o [0,MAX] para enteros, que se mapean del borde inferior al superior.
	 * \param color, color de las líneas
	 */
	cv::Mat draw_graph(
		const cv::Mat &data,
		cv::Scalar color = cv::Scalar::all(255)
	);

	/**Devuelve un gráfico de líneas comparativo.
	 */
	cv::Mat draw_graph(
		const std::vector<cv::Mat> &data,
		const std::vector<cv::Scalar> colour
	);


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

	/**Concatena las imágenes en el vector
	 * Los tamaños deberían concordar.
	 \param r: número de filas
	 */
	cv::Mat mosaic( const std::vector<cv::Mat> &images, size_t r=1);

	/**Devuelve los mapeos para rgb.
	 * Las matrices son de tipo 8U
	 */
	std::vector<cv::Mat> load_colormap(const char *filename);

	/**Rota la imagen en sentido antihorario
	\param angle en grados
	*/
	cv::Mat rotate(cv::Mat image, double angle);

	/**Realiza el filtrado en frecuencia
	\param image matriz 32F, un canal.
	\param filtro de magnitud, descentrado.
	*/
	cv::Mat filter(cv::Mat image, cv::Mat filtro);


	/**Devuelve una imagen descentrada de la magnitud de un filtro ideal
	\param corte frecuencia de corte relativa. 0.5 corresponde un círculo inscripto
	 */
	cv::Mat filter_ideal(size_t rows, size_t cols, double corte);

	/**Devuelve una imagen descentrada de la magnitud de un filtro butterworth
	 */
	cv::Mat filter_butterworth(size_t rows, size_t cols, double corte, size_t order);

	/**Devuelve una imagen descentrada de la magnitud de un filtro gaussiano
	 */
	cv::Mat filter_gaussian(size_t rows, size_t cols, double corte);


	/**Devuelve una matriz compleja dada su magnitud y fase
	 */
	cv::Mat polar_combine(const cv::Mat &magnitud, const cv::Mat &phase);

	/**Dada una matriz compleja, devuelve su magnitud
	 */
	cv::Mat magnitude(const cv::Mat &image);

	/**Dada una matriz compleja, devuelve su fase
	 */
	cv::Mat phase(const cv::Mat &image);

	/**Devuelve un filtro de desenfoque dadas las velocidades de desplazamiento horizontal y vertical
	 */
	cv::Mat motion_blur(cv::Size size, double a, double b);
}


//Implementation
#include <fstream>
#include <limits>
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

	inline cv::Mat fft_shift(const cv::Mat &image){
		cv::Mat result = cv::Mat::zeros(image.size(), image.type());
		int
			cx = image.cols/2,
			cy = image.rows/2,
			w = cx,
			h = cy;
		int
			off_x = image.cols%2,
			off_y = image.rows%2;

		//cuadrante 1 -> 3
		image(cv::Rect(0, 0, w+off_x, h+off_y)).copyTo(
			result(cv::Rect(cx, cy, w+off_x, h+off_y))
		);
		//cuadrante 2 -> 4
		image(cv::Rect(cx+off_x, 0, w, h+off_y)).copyTo(
			result(cv::Rect(0, cy, w, h+off_y))
		);
		//cuadrante 3 -> 1
		image(cv::Rect(cx+off_x, cy+off_y, w, h)).copyTo(
			result(cv::Rect(0, 0, w, h))
		);
		//cuadrante 4 -> 2
		image(cv::Rect(0, cy+off_y, w+off_x, h)).copyTo(
			result(cv::Rect(cx, 0, w+off_x, h))
		);

		return result;
	}

	inline cv::Mat ifft_shift(const cv::Mat &image){
		cv::Mat result = cv::Mat::zeros(image.size(), image.type());
		int
			cx = image.cols/2,
			cy = image.rows/2,
			w = cx,
			h = cy;
		int
			off_x = image.cols%2,
			off_y = image.rows%2;

		//cuadrante 1 -> 3
		image(cv::Rect(0, 0, w, h)).copyTo(
			result(cv::Rect(cx+off_x, cy+off_y, w, h))
		);

		//cuadrante 2 -> 4
		image(cv::Rect(cx, 0, w+off_x, h)).copyTo(
			result(cv::Rect(0, cy+off_y, w+off_x, h))
		);

		//cuadrante 3 -> 1
		image(cv::Rect(cx, cy, w+off_x, h+off_y)).copyTo(
			result(cv::Rect(0, 0, w+off_x, h+off_y))
		);

		//cuadrante 4 -> 2
		image(cv::Rect(0, cy, w, h+off_y)).copyTo(
			result(cv::Rect(cx+off_x, 0, w, h+off_y))
		);

		return result;
	}

	void centre(cv::Mat &imagen){
		imagen = fft_shift(imagen);
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

	namespace{
		/**Dibuja un gráfico de líneas en el canvas. Función auxiliar
		 * \param data vector con los valores a graficar,
		 * rango [0,1] para flotantess o [0,MAX] para enteros, que se mapean del borde inferior al superior.
		 */
		cv::Mat draw_graph(
			cv::Mat &canvas,
			const cv::Mat &data,
			cv::Scalar color = cv::Scalar::all(255),
			double scale = 1
		);

		/**Dibuja un gráfico de líneas en el canvas.
		 * wrapper para aceptar std::vector
		 */
		template<class T>
		cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data);
	}

	inline
	cv::Mat draw_graph(
		const std::vector<cv::Mat> &data,
		const std::vector<cv::Scalar> colour
	){
		cv::Mat canvas = cv::Mat::zeros(256, 256, CV_8UC(3) );
		//encontrar el máximo valor entre todos los arreglos
		double max = -std::numeric_limits<double>::max();
		for(size_t K=0; K<data.size(); ++K){
			double max_K;
			cv::minMaxLoc(data[K], NULL, &max_K);
			max = std::max(max, max_K);
		}

		//graficar
		for(size_t K=0; K<data.size(); ++K)
			draw_graph(canvas, data[K], colour[K], 1);

		return canvas;
	}


	namespace {
		inline cv::Mat draw_graph(cv::Mat &canvas, const cv::Mat &data_, cv::Scalar color, double scale){
			cv::Mat data = data_;
			//convert to range [0;1], type float
			switch(data_.depth()){
				case CV_8U: data.convertTo(data, CV_32F, 1./255, 0); break;
				case CV_8S: data.convertTo(data, CV_32F, 1./255, 0.5); break;
				case CV_16U: data.convertTo(data, CV_32F, 1./65535, 0); break;
				case CV_16S: data.convertTo(data, CV_32F, 1./65535, 0.5); break;
				case CV_32S: data.convertTo(data, CV_32F, 1./(2*2147483647u+1), 0.5); break;
				case CV_32F:
					 break;
				case CV_64F:
					data.convertTo(data, CV_32F, 1);
					break;
			}

			//stretch the graph to take all the columns in the canvas
			double stretch = double(canvas.cols-1)/(std::max(data.rows, data.cols)-1);
			for(int K=1; K<std::max(data.rows, data.cols); ++K){
				cv::line(
					canvas,
					cv::Point( (K-1)*stretch, canvas.rows*(1-scale*data.at<float>(K-1)) ),
					cv::Point( K*stretch, canvas.rows*(1-scale*data.at<float>(K)) ),
					color
				);
			}

			return canvas;
		}

		template<class T>
		inline cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data){
			return draw_graph(canvas, cv::Mat(data));
		}

	}

	inline cv::Mat draw_graph(const cv::Mat &data, cv::Scalar color){
		cv::Mat canvas = cv::Mat::zeros(256, 256, CV_8UC(3) );
		return draw_graph(canvas, data, color, 1);
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
		magnitud = fft_shift(magnitud);

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

	inline cv::Mat mosaic( const std::vector<cv::Mat> &images, size_t r){
		if(images.empty()) return cv::Mat();
		size_t c = images.size()/r + ((images.size()%r)?1:0);

		size_t rows = images[0].rows, cols = images[0].cols;
		cv::Mat big = cv::Mat::zeros(r*rows, c*cols, images[0].type()); //tamaño total

		for(size_t K=0; K<images.size(); ++K){
			cv::Rect submatrix ( (K%c)*cols, (K/c)*rows, images[K].cols, images[K].rows );
			cv::Mat region = cv::Mat(big, submatrix); //región donde pegar
			images[K].copyTo(region);
		}

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

	namespace{
		template <class T>
		inline T square(T x){
			return x*x;
		}
		double distance2( double x1, double y1, double x2, double y2 ){
			return square(x2-x1) + square(y2-y1);
		}
	}

	inline cv::Mat filter_ideal(size_t rows, size_t cols, double corte){
		cv::Mat
			magnitud = cv::Mat::zeros(rows, cols, CV_32F);
		cv::circle(
			magnitud,
			cv::Point(cols/2, rows/2), //punto central
			rows*corte, //radio
			cv::Scalar::all(1),
			-1 //círculo relleno
		);

		magnitud = ifft_shift(magnitud);
		return magnitud;
	}

	inline cv::Mat filter_butterworth(size_t rows, size_t cols, double corte, size_t order){
		//corte = w en imagen de lado 1
		//1 \over 1 + {D \over w}^{2n}
		cv::Mat
			magnitud = cv::Mat::zeros(rows, cols, CV_32F);

		corte *= rows;
		//corte *= corte;
		for(size_t K=0; K<rows; ++K)
			for(size_t L=0; L<cols; ++L){
				double d2 = distance2(K, L, rows/2, cols/2);
				magnitud.at<float>(K,L) = 1.0/(1 + std::pow(d2/(corte*corte), order) );
			}

		magnitud = ifft_shift(magnitud);
		return magnitud;
	}

	inline cv::Mat filter_gaussian(size_t rows, size_t cols, double corte){
		//corte es sigma en imagen de lado 1

		cv::Mat
			magnitud = cv::Mat::zeros(rows, cols, CV_32F);

		corte *= rows;
		//corte *= corte;
		for(size_t K=0; K<rows; ++K)
			for(size_t L=0; L<cols; ++L){
				double distance = distance2(K, L, rows/2, cols/2);
				magnitud.at<float>(K,L) = std::exp(-distance/(2*corte*corte));
			}

		magnitud = ifft_shift(magnitud);
		return magnitud;
	}

	inline cv::Mat polar_combine(const cv::Mat &magnitud, const cv::Mat &phase){
		cv::Mat x[2], result;
		cv::polarToCart(magnitud, phase, x[0], x[1]);
		cv::merge(x, 2, result);
		return result;
	}

	inline
	cv::Mat magnitude(const cv::Mat &image){
		cv::Mat planes[2];
		cv::split(image, planes);

		cv::Mat result;
		cv::magnitude(planes[0], planes[1], result);
		return result;
	}

	inline
	cv::Mat phase(const cv::Mat &image){
		cv::Mat phase, planes[2];
		cv::split(image, planes);
		cv::phase(planes[0], planes[1], phase);

		return phase;
	}

	//a = velocidad en x
	//b = velocidad en y
	inline
	cv::Mat motion_blur(cv::Size size, double a, double b){
		cv::Mat transformation =
			cv::Mat::zeros(size, CV_32FC(2));

		int
			rows = transformation.rows,
			cols = transformation.cols;

		const std::complex<float> I{0,1};
		//fase exp(j\pi (ua + vb))
		//magnitud \frac{ \sin(\pi(ua+vb)) }{ \pi (ua+vb) }
		for(int K=0; K<rows; ++K)
			for(int L=0; L<cols; ++L){
				double
					u = (L-cols/2)/(double)cols,
					v = (K-rows/2)/(double)rows;

				float pi_v = M_PI*(u*a+v*b);
				float mag = (pi_v)? sin(pi_v)/pi_v: 1; //lim{x->0} sin(x)/x
				transformation.at< std::complex<float> >(K,L) = mag*exp(I*pi_v);
			}

		transformation = ifft_shift(transformation);
		return transformation;
	}
}

#endif /* PDI_FUNCTIONS_H */
