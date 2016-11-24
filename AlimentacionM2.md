# Alimentación M2

Ya tenemos los v y w del primer modelo. Intentaremos alimentar el M2.
Para hacer el aprendizaje semi supervisado en el paper hacen el siguiente llamado:
`THEANO_FLAGS=floatX=float32 python run_2layer_ssl.py [n_labels] [seed]`  
Donde `[seed]` es el seed para el random de numpy.  

El archivo anterior lo único que hace es ejecutar el main del archivo learn_yz_x_ss.  

Está medio complicado entender que se hace en todo el código, iremos analizando por partes.  
Iré también comentando el código así no escribo todo acá, cuando sean ideas más globales escribiré acá.  

La vez pasada entrenamos un GPUVAE_Z_X, ahora se ocupa un VAE_Z_X. Espero no haya problemas.  
En esa sección, según yo, deberían ir los mismos hiperparámetros con los cuales se entrenó el modelo.  

A con tinuación viene una parte donde dice *Determines wich dimensions to keep*, aún no entiendo que hace.  
Llama a un atributo del modelo (sí, no es un método) que al parecer es una lista o diccionario. No está en VAE_Z_X si no que está en la clase padre que es VAEModel.  
Al parecer `dist_qz` son las distribuciones para qz (la distribución que hace q(z|x), que me devuelve los z generables por x, recuerda el paper).  
Para que sirve? La gracia es que el dist_qz sirve para generar los z (las variables latentes, que son como las características que va a tener el x), pero como los genera? En realidad al aplicar este dist_qz (que es una función de theano, por eso es atributo) me devuelve la media de los z y su varianza, entonces de esta forma el z será: q_mean + np.exp(0.5 * q_logvar) * z['eps'] donde el eps indica el ruido.  
Esta es la forma con la cual se genera la "extracción de características".  

Hay dos implementaciones, para svhn y para mnist, me da la idea (pura intuición) de que trabajar como lo hacen con svhn sea mejor, pues ahí las fotos tienen estructuras más complejas, de colores y formas.  

Bueno de esa forma se crean las media y las varianzas (parámetros para generar sus z) de los elementos del dataset.  

Luego inicializa los modelos para el aprendizaje con los z y con los $y$ (model) y el modelo de reconocimiento (model_qy).  

### TODO

Analizar como separa los datos en etiquetados y no etiquetados. Sería mejor crear un método en la clase para importar los datos que genere esta separación sencilla, que permita devolver los sets como se necesitan (etiquetados y no etiquetados por separado).  

Analicé los datos y se siguen entregando por columna y la cantidad de filas son cuantas característica (pixeles) tiene la foto.  
Ahora la clase se debe entregar como *one hot encoding*. Debe quedar de la forma (n_clases, x_examples).  

Estoy raja, dejé a medias la creación del método para retornar datos de entranmientos searados entre etiquetados y no etiquetados en HyperspectralData.

Esto está ready. El script está funcionando, es demasiado lento si. Lo estoy probando con 100 labeled y 100 unlabeled y loglikelihood baja al menos.

Ahora mi duda es que el validset error y el testset error son iguales para todos los pasos la wea no cambia. Voy a ver qué me está tirando el modelo.  

Qué pasó? El modelo me predice solo 39. Esto es porque dentro de los ejemplos que le estoy mostrando son casi todos 32.  

Haré un estudio más estadísticos de los datos que le estoy pasando.  
