# Deep generative models

Nos enfrentamos a un problema donde tenemos que los datos están etiquetados. Tenemos un vector de características $x_{i} \in R^{D}$ junto a su etiqueta $y_{i}$. Diremos que las observaciones (los $x_{i}$) tienen variables latentes que notaremos por $z_{i}$.  
Como nos enfrentamos al problema semi-supervisado quiere decir que no todos nuetros datos están etiquetados. Diremos que sobre los datos vamos a tener una distribución empírica
$$ p_{l}(x,y) $$
para los datos etiquetados (l: labeled) y
$$ p_{u}(x) $$
para los no etiquetados (u: unlabeled).  
Ahora veremos modelos para el aprendizaje semi-supervisado que explotan descripciones generativas de los datos para mejorar la clasificación que se será obtenida sólo de los que sí tienen etiquetas.

### Latent-feature discriminative model (M1)

Es típico ocupar un modelo que obtiene una representación característica de los datos. Usando estas características se entrena un clasificador.  
Esta extracción de características permite poder hacer un clustering de observaciones relativas en un *latent feature space* que permite clasificación más precisa, incluso con un número pequeño de etiquetas.  
En vez de ocupar un auto-encoder prefieren ocupar un modelo generativo profundo que les permite extraer de mejor manera estas *latent features*.

El modelo consiste en
$$ p(z) = N(z|0,I) $$
$$ p_{\theta}(x|z) = f(x;z,\theta) $$

donde $f$ es en realidad cualquier *likelihood* que queramos ocupar. Sus probabilidades se forman por una transformación no-lineal, con parámetros $\theta$, de un conjunto de variables latentes $z$.  
Esta tranformación no-lineal es la que permite obtener *higher moments* de los datos (me imagino que mayores niveles de abstracción de las características) y escogieron que estas transformaciones no lineales sean con *deep neural networks*.  

Ahora los samples aproximados del *posterior* $p(z|x)$ se usan como características para entrenar un clasificador para predecir la clase $y$. Haciendo esto entonces pueden hacer la clasificación en un espacio con menos dimensiones, porque se espera ocupar menos variables latentes que número de características tiene la obvservación.
Como estamos en un espacio con dimensiones menores y más encima nuestro *posterior* son gaussianas independientes cuyos parámetros se forman por las transformaciones no-lineales se obtiene que **los ejemplos son más fáciles de separar**. Este sencillo paso resulta en mejoras en la clasificación de las SVMs.

### Generative semi-supervised model (M2)

Se propone un modelo probabilístico que describe los datos como generados por una clase latente $y$ en adición a una variable latente $z$. Los datos son explicados por el modelo generativo

$$ p(y) = Cat(y|\pi) $$
$$ p(z) = N(z|0,I) $$
$$ p_{\theta}(x|y,z) = f(x;y,z,\theta) $$

donde $Cat(y|\pi)$ es la distribución [categorical](https://en.wikipedia.org/wiki/Categorical_distribution). La variable de las clases $y$ se trata como latente si no está disponible y $z$ son variables latentes adicionales. Estas variables latentes son marginalmente independientes, entonces permiten separar, en casos de lectura de dígitos por ejemplo, el dígito que se está leyendo (la clase $y$) del estilo de escritura (variable $z$).   
Como en el caso anterior el *likelihood* es seteado por una transformación no-lineal de las variables latentes. La transformación no lineal la hacen con *deep neural netowrks*.  
La clasificación entonces en este modelo se hace a través de inferencia. Para saber los labels que faltan se calculan los *posterior* $p_{\theta}(y|x)$.

### Stacked generative semi-supervised model (M1+M2)

Combinación de ambos modelos. Primero se obtienen variables latentes $z_{1}$ con M1 y entrenar el modelo M2 con esos *features* en vez de los datos brutos. El resultado es un modelo generativo con dos capas de variables estocásticas:
$$ p_{\theta}(x,y,z1,z2) = p(y)p(z_{2})p_{\theta}(z_{1}|y,z_{2})p_{\theta}(x|z_{1}) $$
donde los *priors* $p(y)$ y $p(z_{2})$ son igual que en el modelo anterior, las otras dos son parametrizadas con *deep neural networks*.

# Scalable variational inference

## Lower bound objective

En todos estos modelos la computación exacta de la distribución a posterior es intratable debido a la no-linealidad y las dependencias entre las variables aleatorias.  
Para los modelos descritos introduciremos una distribución $q_{\phi}(z|x)$ con parámetros $\phi$ que aproximan la distribución a posterior $p(z|x)$. Luego, siguiendo el [principio variacional](https://en.wikipedia.org/wiki/Variational_principle) obtendremos una cota inferior del *marginal likelihood*, lo que asegurará que la aproximación se parezca a la distribución real.  

 Se construye la distribución a posterior aproximada $q_{\phi}(\cdot)$ un *inference or recognition model* que se ha transformado en un método popular para inferencia variacional eficiente (cita al paper anterior, el de los DLGMs).  
 Usando una *inference network* nos saltamos la necesidad de calcular para cada punto los parámetros variacionales pero podemos obtener de igual forma los parámetros $\phi$ que son globales. Esto nos permite disminuir los costos de la inferencia generalizando la distribución a posterior estimada para todas las variables latentes a través de los parámetros de la red, lo que permite inferencia rápida tanto en entrenamiento como en testing.  

 Una *inference netowrk* es introducida para todas las variables latentes y son parametrizadas con *deep neural networks* cuyos outputs forman los parámetros de la distribución $q_{\phi}(\cdot)$.

 Para el *latent-feature discriminative model* (M1) se usa una *Gaussian inference network* $q_{\phi}(z|x)$ para las variables latentes $z$.

 Para el *generative semi-supervised model* (M2) se introduce un *inference model* para cada una de las variables latentes, $z$ e $y$, que se asumen tienen una forma factorizable $q_{\phi}(z,y|x) = q_{\phi}(z|x)q_{\phi}(y|x)$, que son distribuciones gaussianas y categorical (dice multinomial, raro igual) respectivamente.

 M1:

 $$ q_{\phi}(z|x) = \mathcal{N}(z | \mu_{\phi}(x), diag(\sigma^{2}_{\phi}(x))) $$

 M2:

 $$ q_{\phi}(z|y,x) = \mathcal{N}(z| \mu_{\phi}(y,x), diag(\sigma^{2}_{\phi}(x))) $$

 $$ q_{\phi}(y|x) = \mathrm{Cat}(y|\pi_{\phi}(x)) $$

 donde $\sigma_{\phi}(x)$ es un vector de desviaciones estándar, $\pi_{\phi}(x)$ es un vector de probabilidades, y las funciones $\mu_{\phi}(x)$, $\sigma_{\phi}(x)$ y $\pi_{\phi}(x)$ son representadas como [MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron).

### Latent feature discriminative model objective

El *variational bound* $\mathcal{J}(x)$ sobre el *marginal likelihood* para un solo punto es

$$ \log p_{\theta}(x) \geq \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \mathrm{KL}[q_{\phi}(z|x)||p_{\theta}(z)] = - \mathcal{J}(x) $$

La *inference network* $q_{\phi}(z|x)$ es usada durante el entrenamiento del modelo usando tanto los ejemplos etiquetados como los no etiquetados. Esta distribución a posterior aproximada es usada entonces como extractor de características de los ejemplos etiquetados y esas características entonces son usadas para entrenar un clasificador.

### Generative semi-supervised model objective

Para este modelo tenemos que considerar dos casos. El primer caso, el label correspondiente del dato se conoce y el *variational bound* es una simple extensión del bound puesto más arriba (el del modelo anterior):

$$\log p_{\theta}(x,y) \geq
\mathbb{E}_{q_{\phi}(z|x,y)}[\log p_{\theta}(x|y,z) + \log p_{\theta}(y) + \log p(z) - \log q_{\phi}(z|x,y)] = -\mathcal{L}(x,y) $$

Para el caso en que la etiqueta no la conozcamos, la clase es tratada como una variable latente donde se hace inferencia a posterior y la cota entonces es:

$$ \log p_{\theta}(x) \geq
\mathbb{E}_{q_{\phi}(y,z|x)}[\log p_{\theta}(x|y,z) + \log p_{\theta}(y) + \log p(z) -\log q_{\phi}(y,z|x)] $$

$$ = \sum_{y} q_{\phi}(y|x) (-\mathcal{L}(x,y)) + \mathcal{H}(q_{\phi}(y|x)) = -\mathcal{U}(x) $$

###### *Explicación de la igualdad:*
Ojo acá, que la esperanza si sabemos la etiqueta se calcula sobre $q_{\phi}(z|x,y)$ y si no sabemos la etiqueta se calcula sobre $q_{\phi}(y,z|x)$, pero habíamos dicho que

$$  q_{\phi}(z,y|x) = q_{\phi}(z|x)q_{\phi}(y|x)  $$

Entonces si notamos bien la esperanza se puede interpretar como que es el término $-\mathcal{L}(x,y)$ pero falta multiplicarlo por $q_{\phi}(y|x)$ y marginalizar sobre todas las clases. Además sobra un término: $-\log q_{\phi}(y|x)$ (al factorizar el $-\log q_{\phi}(y,z|x) = -\log q_{\phi}(y|x) - \log q_{\phi}(z|x)$) que al ser multiplicado por el $q_{\phi}(y|x)$ resulta la [entropía](https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution), entonces esta es sumada para mantener la igualdad. Es por esto que la esperanza que se muestra es igual a lo que se pone justo abajo.

###### *Concluyendo:*

Por lo tanto la cota para el marginal likelihood para todo el data set será

$$ \mathcal{J} = \sum_{(x,y)-p_{l}} \mathcal{L}(x,y) + \sum_{x - p_{u}} \mathcal{U}(x) $$

La distribución $q_{\phi}(y|x)$ para los ejemplos que no tienen etiqueta es tratada como un *discriminative classifier* y podemos usar este conocimiento para construir el mejor clasificador posible como nuestro *inference model*. Esta distribución es la que se ocupa en los test para predicciones en los datos que no se han visto.

En la función objetivo $\mathcal{J}$ la distribución predictiva de las etiquetas $q_{\phi}(y|x)$ contribuye sólo al segundo término, al de los que no tienen etiquetas, lo que es una propiedad no deseable si queremos usar esta distribución como nuestro clasificador. Idealmente, todo los parámetros (tanto del modelo como variacionales) deberían ser aprendidos en todos los casos. Para arreglar esto se agrega una *classification loss* cuya distribución $q_{\phi}(y|x)$ también aprende de los ejemplos etiquetados.

La función objetivo final resulta ser:

$$ \mathcal{J}^{\alpha} = \mathcal{J} + \alpha \cdot \mathbb{E}_{p_{l}(x,y)}[- \log q_{\phi}(y|x)] $$

Notemos que la esperanza se calcula sobre la distribución de los ejemplos etiquetados.

Acá el hiper-parámetro $\alpha$ controla el peso relativo entre *generative or purely discriminative learning*. En los experimentos usan un $\alpha = 0.1 \cdot N$.  

# Optimization

Los bounds encontrados para nuestros modelos proveen una función objetivo unificada (como las del los DLGMs) para la optimización de ambos parámetros, $\theta$ y $\phi$, del *generative* y del *inference model* respectivamente. Esta optimización se puede hacer en conjunto, sin tener que recurrir al algoritmo EM (expected maximization), una reparametrización determinística de las esperanzas en la función objetivo en conjunto con [Monte carlo approximation](https://en.wikipedia.org/wiki/Monte_Carlo_method).  

A continuación se describe las estrategias principales para el M1, pues las mismas se ocupan para el M2.

Cuando el prior $p(z)$ es una gaussiana esférica $p(z) = \mathcal{N}(z|0,I)$, y la distribución variacional $q_{\phi}(z|x)$ es una gaussiana (como la que habíamos mencionado más arriba) el término de la divergencia KL puede ser computado analíticamente (trivial (?) jajaja).

Ahora, el otro término, el del log-likelihood se puede reescribir como

$$ \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] =
\mathbb{E}_{\mathcal{N}(\epsilon|0,I)}[\log p_{\theta}(x| \mu_{\phi}(x) + \sigma_{\phi}(x) \cdot \epsilon)] $$

Donde el término $\cdot$ indica la multiplicación término a término. Esa transformación es la que mencionan en el paper de los DLGMs que dicen que cualquier gaussiana puede ser obtenida desde una esférica con esa transformación.

Esta esperanza aún no se puede resolver analíticamente (no lo vamos a hacer) pero sus gradientes respecto a los parámetros $\phi$ y $\theta$ pueden ser computados eficientemente como esperanzas de unos simples (?) gradientes:

$$ \nabla_{\{\phi, \theta\}} \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] =
\mathbb{E}_{\mathcal{N}(\epsilon|0,I)}[\nabla_{\{\phi,\theta\}} \log p_{\theta}(x| \mu_{\phi}(x) + \sigma_{\phi}(x)\cdot \epsilon)] $$

Esto es por los teoremas del '64 y '58 que se mete el gradiente a dentro no más, gooooool.

Los gradientes de la función de costo para el M2 pueden ser calculados por aplicación directa de la regla de la cadena y notando que el límite condicional $\mathcal{L}(x_{n},y)$ contiene los mismos términos.

Durante la optimización se usan estos gradientes estimados en conjunto con métodos de descenso del gradiente estocástico, como SGD, RMSprop, o ADAGrad. Esto resulta en actualización de parámetros del estilo $(\theta^{t+1}, \phi^{t+1}) \leftarrow (\theta^{t}, \phi^{t}) + \Gamma^{t}(g^{t}_{\theta}, g^{t}_{\phi})$ donde $\Gamma$ es una *diagonal preconditioning matrix* (?) que va adaptando el gradiente para minimización más rápida.

Ahora, los algoritmos a grandes rasgos se muestran en el paper.

# Manos al código

Ahora queda empezar a ver el código de los cabros y empezar a entender cómo chucha programar toda esta wea.
