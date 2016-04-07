/*
 * PDI_functions
 */
#ifndef PDI_FUNCTIONS_H
#define PDI_FUNCTIONS_H 
#include <iostream>
#include <opencv2/opencv.hpp>

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

	template<class T>
	cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data){
		return draw_graph(canvas, cv::Mat(data));
	}

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

#endif /* PDI_FUNCTIONS_H */
