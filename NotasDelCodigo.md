# Notas del código

Iré escribiendo notas acerca de lo que entiendo del código y de los algoritmos que utilizaron para hacer los modelos del paper *"Semi-supervised Learning with Deep Generative Models"*.  

También comentaré el código, para no estar esceribiendo todo acá.

## Learning M1+M2

Para correr el experimento de los dos modelos juntos se debe ejecutar el script `run_2layer_ssl.py`.

Al tiro notamos que para correr el modelo este script simplemente es para poner los argumentos, el que hace la pega es `learn_yz_x_ss` y se llama a si `main`.

En el código se menciona como *variational auto-encoder* siendo que en el paper no mencionan nada así. Se estudiará.

#### Argumentos del main

Al main se le pasan:
* `n_passes`  
  Cantidad de pasos para entrenar el modelo.
* `n_labeled`  
  Cantidad de ejemplos con labels a utilizar.
* `dataset`  
  El dataset a utilizar (mnist por ejemplo)
* `n_z`  
  El número de variables latentes $z$ a ocupar.

Por entender:
* `n_hideen`  
  A priori me imagino cantidad de nodos de la capa oculta.
* `seed`
* `alpha`:
  Debe ser el paso del optimizador.
* `n_minibatches`
  Tamaño del minibatch.

#### Inicializar los datos y cargar modelos

Al parecer estos wnes tienen los modelos en anglepy, que es una librería de ellos (podría ser buena si es que la podemos ocupar nosotros) e importan por ejemplo *variational auto-encoder* que lo llaman `VAE_Z_X`.  
La librería está escrita con *theano*.

### gpulearn_z_x

Me imagino debe estar optimizado para ser procesado en la GPU.

Lo primero que hace es setear de acuerdo al problema que se esté presentando las variables para el modelo:
* `x`: Dict que tiene de clave 'x' y de valor los datos de entrenamiento.
* `x_valid`: Análogo a 'x'.
* `x_test`: Análogo a 'x'.
* `L_valid`: ?
* `dim_input`: Dimensiones de la imagen del input.
* `n_x`: Tamaño del vector de características 'x'.
* `type_qz`: Distribución para el *recognition model* de las variables latentes.
* `type_pz`: Distribución para el *generative model* de las variables latentes.
* `nonlinear`: Función de activación (?).
* `type_px`: Distribución de (?)
* `n_train`: Número de elementos para entrenar.
* `n_batch`: Número de elementos que debe tener el batch para entrenar.
* `colorImg`: (boolean) Si la imagen es en color.
* `bernoulli_x`: (boolean) Si 'x' distribuye como una bernoulli (?).
* `byteToFloat`: (?)
* `weight_decay`: Parámetro para el optimizador.

Luego importa el modelo a ocupar `GPUVAE_Z_X`, que está en la librería `anglepy` (tiene todos los modelos).

Luego setea el optimizador (ocupan ADAM, versión del stochastic gradient descent pero con momentum) y setea el modelo (línea 256).

Después se define la función `hook` que va validando el modelo y guardando las variables de este cuando alcanza los mejores resultados (hasta ese momento) y además para el proceso cuando han pasado 100 iteraciones sin mejoras.  
Finalmente guarda imágenes que representan los valores del modelo.
Esta función entonces es la que sirve para ir viendo los valores y además genera unos samples de los datos con las valores que tiene.


Al final para optimizar el modelo se hace un loop con `loop_va(dostep, hook)` que loopea haciendo un paso del optimizador y va llamando al `hook`.

### GPUVAEModel

Se construye con un optimizador, llamado `get_optimizer`.


### GPUVAE_Z_X

Clase del modelo deep variational auto-encoder. La gracia es que por si solo, solo con los datos de ejemplos, extrae variables latentes que modelan los datos, las distribuciones de los datos se setean mediante MLP de las variables latentes $z$.

Hereda de GPUVAEModel.

# M1+M2

Vamos a ver como se implementa el que tiene mejores resultados.  
Para esto vamos al script que lo corre que es `run_2layer_ssl` el cual llama al main del archivo `learn_yz_x_ss`.  

Vamos a intentar correrlo primero y corregir errores de compatibilidad.

Lo que tenemos en el script `learn_yz_x_ss` es que se carga un modelo para la primera capa (el M1) el cual ya está entrenado, ya se hizo la extracción de características con aprendizaje no supervisado. El modelo ahí utilizado es el `VAE_Z_X`.  

...

# gpulearn_z_x

Volvamos a esto para entender bien como podemos ocupar el variational autoencoder para nuestros datos.
Vamos paso a paso.
Lo primero que hace es cargar los datos en diccionarios y setear algunas distribuciones. Pone por ejemplo (en mnist) la distribución aproximada como gaussiana igual que la verdadera (la $q$ y la $z$).  
Setea cuantos ejemplos son de entrenamiento y el tamaño del batch.  

Importa el modelo, que se llama `GPUVAE_Z_X`. Setea el optimizador que lo obtiene de la función `get_adam_optimizer`.  

No debería ser muy difícil poder ocupar lo mismo salvo cambiando el dataset.

viendo el modelo `GPUVAE_Z_X` hay demasiadas variables que no se que hacen específicamente, como las variables *hidden* o *var_smoothing* por ejemplo. No está nada documentado.
