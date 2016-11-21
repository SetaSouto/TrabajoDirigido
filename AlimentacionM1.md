# Alimentación modelo 1

Comenzaremos con un estudio de como poder alimentar el M1 para la extracción de características.  

Primero veremos como son cargados los datos, por ejemplo, de *mnist*.  
Estos datos están en la librería, y se pueden cargar en formato numpy array haciendo:
```python
import anglepy.data.mnist as mnist
size = 28
train_x, train_y, valid_x, valid_y, test_x, test_y =
  mnist.load_numpy(size)
```

O sea a priori, debemos asignar sets de entranamiento, validación y testeo.  

Veamos que forma tienen y qué valores encontramos ahí.  

Con el script `AnalisisFormaMNIST.py` se pueden obtener valores que describen los arreglos, que representan las imágenes.  

Por ejemplo, el `train_x` es una matriz con $784$ filas y $50000$ columnas. Esto quiere decir que cada columna es un ejemplo. Son $784$ filas pues las imágenes son de $28\cdot 28$.  

Los valores de los vectores de entrenamiento rondan entre 0 y 1.

Ahora haremos un script para ver como están los datos que tenemos.  

#### Hyperspectral dataset

En el caso de las imágenes hiperespectrales los datos rondan entre 0 y (en un solo archivo) 65535. Lo que es demasiado grande, vamos a tener que normalizar los datos.  

Para ver caracteristicas de algun archivo se puede ejecutar el script `AnalisisDataset.py` que está en el disco duro externo, junto con los datos.  

Por ejemplo, analisando el archivo `ALH1599-17-labeled.hdf5` si ocupamos los datos de minerología y aceptamos los ejemplos "no etiquetados" tenemos 916182 ejemplos. Muchísimos.

#### Normalización necesaria?

Veamos como son ingresados los datos al modelo. Para evitar bugs, buscaremos alimentar el modelo ya configurado para mnist (y otros ejemplos) con nuestros datos, entonces vamos a querer que nuestro dataset tengo mas menos los mismos valores que se alcanzan en mnist para no tener que configurar nosotros los hiperparámetros.  

Para correr el modelo en ningún momento se hace un reformateo de los valores de mnist, asi que a priori serviría setear todos nuestros valores entre cero y uno nada más.  

Dividiendo por el máximo dejamos todo entre cero y uno. Tienen una desviación estándar (en el archivo que se menciona más arriba) de $0.213$, lo que anda relativamente cerca del mnist que tenía un std de $0.307$.  

#### Alimentación del modelo

Estoy escribiendo, o más bien adecuando, el gpulearn_z_x en un archivo aparte que se llama `1HYP_gpulearn_z_x.py`. Al final agregué un if al `gpulearn_z_x.py` normal no más.

En el caso de MNIST tenemos 50000 ejemplos para entrenar, 10000 de validación y 10000 también de test.  

A esta altura me surgen algunas dudas:
* Con cuantos ejemplos quedará bien informado nuestro modelo? Porque claro, con mnist solo hay 10 clases. En nuestro problema tenemos 100 clases distintas solo para minerología.  
En un solo archivo tenemos 916 mil ejemplos. Tenemos muchísimos, pero será bueno meterlos todos? Con cuantos de entrenamiento podría ser bueno? Hiperparámetros que se tendrán que ir viendo/conversando con Pablo.  

A priori, tomaremos 100 mil para entrenar.  

En realidad la extracción de los datos no es tan complicada pero hacerlo en un solo archivo es muy sucio. Escribiré una clase con funciones que permitan entregar los datos como se necesiten.  

Ya desarrollé la clase, es el archivo `hyperspectralData.py` que se encuentra dentro de anglepy.  

Ya se pueden importar los datos fácilmente.  
En el archivo ahora tenemos un problema: Cuando se setean las variables para mnist, colocan la variable "size" que indica la dimensión de la foto. En nuestro caso la dimensión no es un cuadrado perfecto (size*size) entonces tenemos que averiguar donde se utiliza esto y ver como lo podemos setear.  

Para nuestra suerte, esta variable solo se ocupa para pasar la "matemática a imágenes". Es decir mostrar visualmente que está saliendo de nuestra red.  

Ahora técnicamente el modelo debería correr (cosa que no pasa jaja) pero se ejecuta así:
`THEANO_FLAGS=floatX=float32 python run_gpulearn_z_x.py "hyperspectral"`

Ahora está funcionando, no estabas entregando los datos normalizados, deben ser menores a uno. Este bug fue resuelto en la clase hyperspectralData tomando los datos y dividiéndolos por el máximo más uno.  

El modelo corre, pero no puede pasar los datos a imágenes, porque no son cuadradas. Intentaré hacerla de alguna forma rectangular, veremos como nos va. La forma elegida es $(67,4)$ que es como factoriza $268$ que es la cantidad de datos por vector de ejemplo.  

Noticias buenas: El loglikelihood está bajando.
Noticias malas: Alcanzó el mínimo al tiro, se estancó en -110. Ahora estoy corriendo el mnist para ver a cuanto baja el loglikelihood de ellos.  

Corriendo el mnist... Lleva 30307 segundos (cerca de 8 horas y media) se estanca a ratos pero no pasa de 50 pasos estancado. Lleva 1350 pasos y tiene un likelihood sobre el validdataset de -97 prox.  

TODO: Entender la función de likelihood.  

Voy a terminar el mnist, va en el paso 1740 y sigue con loglikelihood cerca de -97, logró bajar a -96.8.

#### Retomando (18/10)

Vamos a entender que mide el log likelihood.  
TODO: Entender cada hiperparámetro.  

El modelo que estamos ocupando es el `GPUVAE_Z_X` que hereda de `GPUVAEModel`, que es el que tiene la función `est_loglik()` que permite calcular el loglikelihood sobre el validdataset.  

Qué hace `est_loglik()`?  
Toma los datos sobre los cuales se quiere calcular el "lower bound objective" del modelo. Retorna el lowbound promedio de todos los samples.  
Que es el lowbound acá?  
Este se define el `GPUVAE_Z_X`, en la definición de factors se indica que L(x) = logpx + logpz.  
`logpx` es la función de densidad del modelo, o sea es $\log p(x|z)$.  
`logpz` es el prior de Z más la entropia de $q(z|x)$, o sea es $\log p(z) + KL(q(z|x)||p(z))$.  
Esto se calcula a través de como 100 línea ocupando theano.  
Ojo, hay una leve inconsistencia según yo, lo que pasa es que después en la función que calcula el lower bound se hace logpx + logpz - logqz.  
Bueno, filo, la cuestión es que devuelve el lowebound objective del M1.  

Eché a correr de nuevo el script, cambié los hiperparámetros de qué densidades ocupar de "gaussianmarg" a "gaussian". También cambié la función no lineal, ahora estamos ocupando "tanh" e hice Bernoulli_x = False, este es el único que en verdad no se para que sirve, pero por ejemplo en frey faces ocupaban tanh y False en el bernoulli.  

#### Idea testeo:

En realidad no hay forma de convencerse de que el modelo está haciendo extracciones de características finas. En el mnist o en frey faces se pueden ver los números y caras generándose, en nuestro caso no tiene mucho sentido, no son imágenes donde tengamos mucho ojo.  

Entonces se puede hacer lo siguiente, ir comparando el lowerbound de mnist con las imágenes que producen. Por ejemplo la última vez se lograron buenos samples del modelo cuando mnist iba en los 1740 pasos (ahí lo paré ctrl+C).  

Entonces se podría ir escribiendo en un archivo el lowerbound para cada paso (o cada 10 pasos) y así poder comparar el "número del lower bound" con la imagen que está generando el modelo a esa altura. Así si nuestro modelo alcanza cierto nivel de lower bound, poder decir "aah ya pero igusal está haciendo extracción de características bien."  

#### Problema

Al cambiar los hiperparámetros está tirando totalmente otros valores en el lower bound. OJO ahí. Va a ser difícil determinar entonces si funciona bien, pues la idea anterior no estaría funcionando. Ahora por ejemplo está tirando valores positivos (que van aumentando) del orden de 731 en el paso 30.  

#### Output to csv

Ahora dejé corriendo el modelo (de nuevo) con los mismos hiperparámetros solo que ahora me escribe los resultados que printea en un archivo que se llama `results.txt` en estilo csv. Así después podríamos visualizar estos datos también.  

#### Comparación hiperparámetros:

En verdad resulta muy difícil medir cuales funcionan mejor, pues sacan lowerbounds distintos. No tengo claro cuales son las unidades de medida. Lo único que se me ocurre a priori es ver las fotos de output y la primera vez que lo corrí las fotos que se generaron en vdd no me daban nada de información, no tenía con qué compararlas. Ahora las que se están generando aún con los otros hiperparámetros se están pareciendo mucho a esas que se generaron con los primeros (y además en menos pasos). Igual es cuático, ahora las miré con más detención y las de la primera vez se parecieron al toque, pero es que al toque a las finales (que al final se estancaba), quizás después de todo, lo hacía la raja con esos hiperparámetros jajaja.   

Algo que se podría hacer es pasar a imágenes los datos que se tienen y compararlos no más. Ver cual genera imágenes más parecidas a los ejemplos.  

Otra idea es correr los dos también y sobre los dos hacer el M2 y ver con cual funciona mejor y punto.  

En todo caso se están guardando todas las variables y ahora se escribe el csv con los valores de cómo van evolucionando.  

#### Actualización

Detuve el que estaba corriendo con los hyperparámetros que estaban llevando sus fotos a lo que estaba haciendo desde el principio el otro. Dejé corriendo el que tiene las gaussianmarg.  

#### Actualización 27/10

Limpiaré el script de entrenamiento para el m1 para poder pasárselo a sergio. Listoco, ahora crearé el escript para sacar características, me basaré en el learn_yz_x_ss.py.
