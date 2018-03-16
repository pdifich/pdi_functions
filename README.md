# Conjunto de funciones auxiliares de OpenCV para C++ y Python


## Archivos
* `pdi_functions.h`: contiene la definición de las funciones dentro del namespace pdi
```c++
#include "pdi_functions.h"
using namespace pdi;
```
Al compilar deberá utilizarse el flag `-std=c++11`

* `build.sh`: es un script para facilitar la creación de ejecutables.
```sh
$ bash build.sh program.cpp
$ ./program.bin
```

* `opencv.gdb`: provee funciones para visualizar los `cv::Mat` dentro de `gdb`
```sh
#file ~/.gdb.init
source ~/.local/share/gdb/opencv.gdb
```
```sh
(gdb) pmat image float
```

## Documentación
Para generar un html con la documentación utilícese `doxygen`
