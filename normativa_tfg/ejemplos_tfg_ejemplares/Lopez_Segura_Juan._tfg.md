# Análisis de las Redes Neuronales Recurrentes: Enfoque en las LSTM y GRU para predicción

**UNIVERSIDAD COMPLUTENSE DE MADRID**  
**FACULTAD DE MATEMÁTICAS**  
**GRADO EN INGENIERÍA MATEMÁTICA**

**TRABAJO DE FIN DE GRADO**

- **Alumno:** Juan López Segura  
- **Tutores:** Antonio López Montes, Ángel González Prieto, Antonio Martínez Raya y Teresa Benavent Merchán  
- **Curso académico:** 2022-23  
- **Convocatoria:** Junio

---

## Resumen

Este Trabajo de Fin de Grado aborda de forma resumida las redes neuronales, presentando una introducción a las mismas junto a algunas de sus aplicaciones, contexto histórico y enfoque matemático a los algoritmos de entrenamiento más relevantes, para posteriormente profundizar en las RNN (Redes Neuronales Recurrentes) y sus arquitecturas.

Tras eso, se describen exhaustivamente las dos configuraciones de celdas más destacadas de las RNN (LSTM y GRU), creadas para lidiar con los problemas derivados del gradiente, de cara a su posterior aplicación en diversos ámbitos. Se plantean ejemplos que permitan aplicar en un caso práctico los métodos matemáticos estudiados, tomando como ejemplo principal la predicción de precios trimestrales de 3 criptodivisas.

Este trabajo, por tanto, se basa en un estudio teórico-práctico de las RNN comentando sus características principales y estructuras, habiendo establecido de antemano el contexto necesario para comprenderlas.

## Abstract

This Bachelor's Thesis provides a concise overview of neural networks, offering an introduction to the field along with some of their applications, historical context, and a mathematical approach to the most relevant training algorithms. It then deepens into the Recurrent Neural Networks (RNNs) and explores their architectures.

Subsequently, the two most prominent cell configurations of RNNs (LSTM & GRU) are thoroughly described, created in order to deal with gradient-related issues, with a focus on their practical application in various domains. Several examples are presented that allow the mathematical methods studied to be applied in a practical case, with the primary example being the quarterly price prediction of three cryptocurrencies.

Therefore, this academic work is based on a theoretical-practical study to RNNs, discussing their main characteristics and structures, while establishing the necessary context for understanding them.

---

## Índice de contenidos

1. [Motivación](#1-motivación)
2. [Introducción a las redes neuronales](#2-introducción-a-las-redes-neuronales)
   - 2.1 [Historia](#21-historia)
   - 2.2 [Clasificación de redes neuronales](#22-clasificación-de-redes-neuronales)
3. [Estructura de una red neuronal](#3-estructura-de-una-red-neuronal)
   - 3.1 [Perceptrón](#31-perceptrón)
   - 3.2 [ADALINE](#32-adaline)
   - 3.3 [Perceptrón multicapa](#33-perceptrón-multicapa)
4. [Redes Neuronales Recurrentes](#4-redes-neuronales-recurrentes)
   - 4.1 [Arquitecturas de una RNN](#41-arquitecturas-de-una-rnn)
   - 4.2 [Celda LSTM](#42-celda-lstm)
     - 4.2.1 [Puerta de olvido](#421-puerta-de-olvido)
     - 4.2.2 [Puerta de entrada](#422-puerta-de-entrada)
     - 4.2.3 [Puerta de salida](#423-puerta-de-salida)
   - 4.3 [Celda GRU](#43-celda-gru)
   - 4.4 [Consideraciones finales](#44-consideraciones-finales)
5. [Problemas abordados](#5-problemas-abordados)
   - 5.1 [Problema 1](#51-problema-1)
     - 5.1.1 [Función sin x](#511-función-sin-x)
     - 5.1.2 [Función sin(x)/x](#512-función-sinxx)
   - 5.2 [Problema 2](#52-problema-2)
   - 5.3 [Problema 3](#53-problema-3)
     - 5.3.1 [Bitcoin](#531-bitcoin)
     - 5.3.2 [Ethereum](#532-ethereum)
     - 5.3.3 [Ripple](#533-ripple)
6. [Conclusiones y trabajo futuro](#6-conclusiones-y-trabajo-futuro)
- [A. Apéndices](#a-apéndices)
  - A.1 [Funciones de activación](#a1-funciones-de-activación)
  - A.2 [Explicación completa del algoritmo de Backpropagation](#a2-explicación-completa-del-algoritmo-de-backpropagation)
  - A.3 [Otros ejemplos](#a3-otros-ejemplos)
- [B. Anexos](#b-anexos)
  - B.1 [Código Ejemplo 1](#b1-código-ejemplo-1)
  - B.2 [Código Ejemplo 2](#b2-código-ejemplo-2)
  - B.3 [Código Ejemplo 3](#b3-código-ejemplo-3)
- [Bibliografía](#bibliografía)

---

## 1. Motivación

He realizado mi Trabajo de Fin de Grado sobre redes neuronales debido a las aplicaciones tan diversas que presentan, el desarrollo e inversión en las que se encuentran actualmente y el auge actual de las RNN y sus estructuras derivadas.

Más concretamente, estoy interesado en su aplicación a los modelos predictivos y el avance que puede suponer en infinidad de ámbitos, como la medicina, farmacología, economía, tráfico o estudio de poblaciones. En este trabajo se revisará la historia de las redes neuronales, la estructura de la neurona y su comparación con la neurona artificial, la arquitectura y las estructuras matemáticas en que se basan, centrando la atención en las redes LSTM (Long-Short Term Memory) y GRU (Gated Recurrent Unit), junto a diversos ejemplos que intenten esclarecer la relevancia de estas estructuras en problemas de predicción.

El trabajo se estructura conforme a 3 bloques: el primero se basa en los orígenes y utilidades de las redes neuronales, junto a las estructuras más básicas y algoritmos principales que formaron las bases del resto. Tras eso, en el segundo bloque se ahonda en las RNN, presentando sus principales ventajas, los problemas derivados, las estructuras creadas para solucionarlo y los fundamentos matemáticos detrás de ellas, junto a las arquitecturas principales. Por último, se abordan 3 ejemplos en orden creciente de profundidad, con los que se pretenden aplicar los conocimientos teóricos presentados en casos prácticos, extrayendo las debidas conclusiones.

---

## 2. Introducción a las redes neuronales

Una red neuronal es un método de inteligencia artificial basado en procesar datos de una manera simplificada pero inspirada en el funcionamiento del sistema nervioso. Se trata de un tipo de proceso de machine learning denominado aprendizaje profundo que se fundamenta en un conjunto de interconectado de unidades llamadas neuronas artificiales. Dichas neuronas se organizan en capas comunicadas entre sí mediante conexiones ponderadas, conocidas como sinapsis artificiales. Gracias a su estructura dividida en capas de entrada, ocultas y de salida, crean un sistema adaptable que se utiliza para aprender y mejorar de los errores cometidos.

Las ideas que subyacen detrás de este campo de estudio se basan en simular la forma en la que las neuronas biológicas procesan y transmiten la información. Las neuronas artificiales, mediante el uso de las capas mencionadas y de otros elementos como las funciones de activación, reciben una entrada, la procesan y transmiten la salida a las neuronas de la capa posterior. Según esta información fluye por la red, los pesos de las sinapsis se ajustan con la finalidad de mejorar el rendimiento y la precisión de la salida de la red.

Esta arquitectura le confiere a la red una característica o ventaja principal sobre el resto de procedimientos: su capacidad de aprendizaje. Gracias a su entrenamiento, las redes neuronales pueden reconocer patrones complejos, siendo tremendamente útiles para tareas como la clasificación, el reconocimiento de voz o el procesamiento del lenguaje natural, entre otras.

Entre la gran cantidad de aplicaciones y posibilidades detrás de las redes neuronales, algunas de las más relevantes son:

- **Clasificación de imágenes:** Para diagnósticos médicos, reconocimiento facial, identificación de logotipos, etc.
- **Predicciones de datos de diversa índole:** posibilidad o no de muerte según factores cardiacos, series temporales, predicciones financieras, etc.
- **Previsiones de demanda.**
- **Control de calidad.**
- **Identificación de compuestos químicos.**
- **Procesamiento del lenguaje natural** para la traducción automática, generación de texto, etc.
- **Reconocimiento visual en vehículos autónomos.**

En resumen, las redes neuronales se han convertido en una poderosa herramienta para el procesamiento y estudio de todo tipo de datos, con amplias aplicaciones gracias a conceptos como el del aprendizaje automático.

### 2.1. Historia

Desde las antiguas civilizaciones se ha estudiado el cerebro y el pensamiento, comenzando con pensadores como Platón y Aristóteles, para intentar comprender el funcionamiento detrás de un órgano tan importante.

Sin embargo, no fue hasta finales del siglo XIX cuando el científico español Santiago Ramón y Cajal descubre los diferentes tipos de neuronas en forma aislada, y otros conceptos esenciales como la división del sistema nervioso en neuronas individuales, comunicadas entre sí mediante la sinapsis. Tras esto, Alan Turing (1912-1954) fue el primero en estudiar el cerebro desde un punto de vista computacional, planteando un nuevo área que revolucionaría el mundo, la inteligencia artificial. Posteriormente, aparecen las personalidades de Warren McCulloch y Walter Pitts, neurobiólogo y estadístico respectivamente, quienes comenzaron en 1943 a trazar el inicio de las RNA (Redes Neuronales Artificiales) con la publicación de su artículo *"A logical calculus of the ideas immanent in nervous activity"* [20], junto con su primer modelo computacional de una red neuronal constituido por una arquitectura de red simple con circuitos eléctricos. Dicha red fue denominada "lógica umbral", ya que resolvía funciones elementales de encendido y apagado, simulando la respuesta lógica a los estados "A ∨ B", "A ∧ B" y "Ā".

La actualización de este primer modelo llega de la mano de Donald Hebb en 1949, quien explicó por primera vez los procesos del aprendizaje desde un punto de vista psicológico, cuyo fundamento se utiliza para las funciones de aprendizaje de la mayoría de redes neuronales. Desarrolló el nuevo tipo de regla para el aprendizaje no supervisado, que se conoce como el aprendizaje de Hebb, y sus trabajos [14] formaron las bases de la Teoría de las Redes Neuronales.

Tras ello, en 1958 Frank Rosenblatt construye una red neuronal asociada a un algoritmo de aprendizaje con reconocimiento de patrones la cual denomina perceptrón. Un perceptrón se basa en tomar varias entradas binarias (x₁, x₂, x₃, etc) y producir una salida binaria única, teniendo en cuenta unos determinados pesos (w₁, w₂, w₃, etc). La salida es 1 ó 0 dependiendo de si el valor final es mayor o menor que un umbral determinado, y es la estructura de red neuronal artificial más sencilla.

Tras esto, en 1965 aparece el perceptrón multicapa, el cual se fundamenta en una ampliación del anterior añadiendo capas de entrada, salida, y conceptos como el de capas ocultas, pretendiendo asemejarse a una neurona biológica simulando las dendritas (conexiones de entrada), el núcleo (elemento procesador o capas ocultas), y los axones (conexiones de salida). No obstante, los valores de entrada y salida siguen siendo binarios, y el valor de los pesos y el umbral sigue siendo asignado manualmente por el creador.

A continuación, en la década de 1980, se introduce un nuevo tipo de neuronas llamadas neuronas sigmoides para lograr que las redes aprendiesen solas (aprendizaje automático). En este caso las entradas pueden ser valores reales y aparecen parámetros denominados sesgos en ciertas capas. De esta forma, la salida que se consigue tiene la forma d(w·x+b), donde d es una función sigmoide d(z) = 1/(1+e⁻ᶻ), siendo ésta la primera función de activación.

En 1986 aparece el algoritmo de backpropagation, y gracias a ello se consigue entrenar redes neuronales de múltiples capas de forma supervisada. Con dicho algoritmo se calcula el error de la salida y se hacen ajustes pequeños en capas previas, ayudando a que clasifique las entradas de mejor forma para minimizar errores.

Por último, antes de llegar a las LSTM, aparecen también las redes neuronales convolucionales, cuya arquitectura es especialmente útil para el procesamiento de imágenes, y se basa en varias capas que extraen características y clasifican.

### 2.2. Clasificación de redes neuronales

En el caso de las redes neuronales, hay múltiples clasificaciones distintas, aunque suele haber una división generalizada conforme a dos criterios: según la topología de la red (arquitectura) y según el método de aprendizaje.

**1. Topología de la red**

- **Red neuronal monocapa:** Corresponde a la estructura más sencilla, formada únicamente por una capa de entrada y una de salida. Ejemplo: Perceptrón.
- **Red neuronal multicapa:** Consiste en la aparición de capas intermedias entre la entrada y la salida llamadas capas ocultas. Ejemplo: Perceptrón multicapa.
- **Red neuronal convolucional:** La diferencia principal con respecto al perceptrón multicapa radica en la existencia de filtros, reduciendo el número de neuronas necesarias y la complejidad computacional.
- **Red neuronal recurrente:** En este caso no hay una estructura de capas definidas, sino que se permiten conexiones arbitrarias entre neuronas, siendo posible la creación de ciclos y dotando a la estructura de temporalidad, es decir, que la red tenga memoria.
- **Red de base radial:** Las redes de base radial se basan en el cálculo de la salida de la función dependiendo de la distancia al punto denominado centro, aplicando combinaciones lineales de funciones de activación radiales.

**2. Método de aprendizaje**

- **Supervisado:** Este tipo de aprendizaje se caracteriza por disponer de los valores de las respuestas, siendo posible una supervisión y corrección de los resultados de la red neuronal, pretendiendo que se acerquen lo máximo posible al valor deseado sin incurrir en overfitting[^1] ni underfitting[^2].
- **No supervisado o autosupervisado:** Su principal diferencia con respecto al aprendizaje supervisado se basa en que no necesita influencia externa para el ajuste de pesos, dividiéndose a su vez en aprendizaje hebbiano y aprendizaje competitivo y comparativo.
- **Por refuerzo:** Este tipo de aprendizaje se fundamenta en un mecanismo de probabilidades que adecúa los pesos sinápticos en función de si un dato es aceptable o no.
- **Híbrido:** Utiliza tanto aprendizaje supervisado como no supervisado, aplicando cada algoritmo a algunas de sus capas.

[^1]: Sobreajuste: falta de generalización.
[^2]: Subajuste: falta de información.

---

## 3. Estructura de una red neuronal

En una red se diferencian tres niveles: la capa de entrada, las ocultas y la de salida. La primera es la que recibe las señales de entrada y recoge el vector de información, mientras que la última es la que transmite la respuesta final al medio externo o a la siguiente neurona, tras recibir la información de la última capa oculta. Las capas ocultas (pueden ser cero o el número decidido por el programador) no tienen contacto con el exterior y se encargan de realizar las transformaciones pertinentes mediante conexiones diversas.

### 3.1. Perceptrón

El perceptrón o perceptrón de una sola capa (Single Layer Perceptron o SLP) fue el primer y más sencillo modelo de una red neural artificial. Su arquitectura es simple, ya que está formada por varias entradas con pesos y una salida, haciendo uso de una función de activación binaria o bipolar. No tiene capas ocultas, por lo que es un ejemplo de una red monocapa.

Utilizando como componentes el vector de entrada **x** = (x₁, ..., x_N) ∈ ℝᴺ, los pesos **w** = (w₀, ..., w_N) ∈ ℝᴺ⁺¹ y la salida y ∈ ℝ, las transformaciones son:

$$h = \sum_{n=1}^{N} w_n x_n + w_0; \quad h \in \mathbb{R}$$

$$y = f(h) = \begin{cases} 1 & \text{si } h \geq 0 \\ 0 & \text{si } h \leq 0 \end{cases}$$

Siendo en este caso el sesgo o umbral w₀. En cuanto a su aprendizaje, éste se basa en calcular los vectores de salida correspondientes y_p ∈ ℝ dados diversos conjuntos de vectores de entrada. El error cometido para cada patrón de entrenamiento es:

$$e_p = (d_p - y_p)^2$$

Los pesos se actualizan mediante la fórmula siguiente, donde α es el factor de aprendizaje:

$$w_n(k+1) = w_n(k) + \alpha(d_p - y_p) \cdot x_{pn}$$
$$w_0(k+1) = w_0(k) + \alpha(d_p - y_p)$$

Mediante estas ecuaciones, que constituyen el **Algoritmo de Gradiente Descendente**, se pretenden minimizar los errores hasta que sean suficientemente pequeños o se llegue a un número de iteraciones determinadas.

### 3.2. ADALINE

La red ADALINE o *Adaptative Linear Neuron* es muy similar al perceptrón. En este caso la función de activación es lineal, por lo que la salida es un número real, y puede tener más de una neurona en su capa de procesamiento, dando lugar a un sistema con N entradas y M salidas.

El algoritmo de aprendizaje se basa en una variante del Algoritmo de Gradiente Descendente llamada **Regla Delta**. Se define el error total producido por los P patrones de entrenamiento:

$$J = \frac{1}{2} \sum_{p=1}^{P} \sum_{m=1}^{M} (d_{m,p} - y_{m,p})^2 = \frac{1}{2} \sum_{p=1}^{P} e_p$$

Los pesos se actualizan como:

$$w_{n,m}(k+1) = w_{n,m}(k) + \alpha(d_{m,p} - y_{m,p}) \cdot x_n$$
$$w_{0,m}(k+1) = w_{0,m}(k) + \alpha(d_{m,p} - y_{m,p})$$

### 3.3. Perceptrón multicapa

Un perceptrón multicapa presenta una estructura similar al perceptrón, pero contiene una o más capas ocultas y su modelo de entrenamiento es el de retropropagación. La primera fase del algoritmo es la **fase de avance**, donde las activaciones se propagan desde la capa de entrada hasta la salida; la segunda fase o **fase de retroceso**, el error se propaga en sentido inverso para modificar los pesos y valores umbrales de las capas anteriores.

El **algoritmo de backpropagation** selecciona los vectores de datos de entrenamiento, calcula la salida de las neuronas ocultas y propaga los valores hasta calcular la salida final, para posteriormente determinar el error, modificar los pesos dependiendo del error de la capa de salida y de la estimación del error en las capas ocultas. El proceso se repite hasta que se verifique la condición de parada. La explicación completa de este algoritmo se encuentra en el apéndice A.2.

**Otros ejemplos:** Además de las estructuras comentadas, existen muchas otras arquitecturas diferentes con propósitos muy específicos, que se pueden encontrar brevemente explicadas en el apéndice A.3.

---

## 4. Redes Neuronales Recurrentes

Las redes neuronales recurrentes o **RNN** se basan en un trabajo de David Rumelhart [27], el cual se basó a su vez en un estudio de John Hopfield en 1982.

En las redes neuronales vistas hasta ahora, el flujo de activación va en una sola dirección. Sin embargo, en las RNN existen conexiones hacia atrás que permiten que en cada instante t la neurona reciba la información de entrada en ese paso de tiempo y de salida del anterior. Además, son capaces de analizar sucesiones, lo que es de gran utilidad si los datos tienen temporalidad o su ordenación es de importancia.

Cada elemento del vector de entrada x_t, junto con el estado oculto de la red en el paso de tiempo anterior h_{t-1}, dan lugar a una salida y_t y un estado oculto nuevo h_t utilizado en el paso siguiente. Para crear el nuevo estado oculto a partir del estado anterior y la entrada, hace uso de una transformación lineal y de una función de activación. Tras eso, devuelve la predicción haciendo uso del nuevo estado oculto y aplicando una función **softmax** (función exponencial normalizada, alternativa de la función sigmoidea).

Sin embargo, esto conduce al principal problema de las RNN: su **memoria a corto plazo**. Las fórmulas que reflejan este fenómeno son:

$$y_3 = \text{softmax}(W_{y,h} \cdot h_3)$$
$$h_n = \tanh(W_{h,h} \cdot h_{n-1})$$
$$h_3 = \tanh(W_{h,h} \cdot \tanh(W_{h,h} \cdot \tanh(W_{h,h} \cdot h_0)))$$

La anidación de funciones de activación desemboca en el **desvanecimiento de gradiente**: el efecto del estado oculto inicial es menor en cada etapa, llevando a una red con memoria de corto plazo. También puede ocurrir el efecto contrario: la **explosión de gradiente**.

Para mitigar este problema surgen dos arquitecturas de celdas específicas: la arquitectura **LSTM** y **GRU**, que incluyen mecanismos de puertas que permiten controlar el flujo de información y recordar información a largo plazo.

En las RNN no se aplica backpropagation de la forma habitual. En su lugar, se entrenan con una adaptación conocida como **retropropagación en el tiempo** (*backpropagation through time*), "desenrollando" la red recurrente para convertirla en una red feedforward y aplicar el algoritmo de propagación hacia atrás.

### 4.1. Arquitecturas de una RNN

- **Uno a Uno:** No presenta secuencias; es análogo a una red neuronal tradicional aplicada a un único paso de tiempo.
- **Uno a muchos:** Basándose en una única entrada, se generan diversas secuencias.
- **Muchos a uno:** Teniendo en cuenta diversas entradas en forma de secuencia, se genera una salida. Usado en modelos de predicción de series temporales.
- **Muchos a muchos:** Se encuentran tanto secuencias de entrada como de salida. Pueden ser directas (cada salida procede de la entrada respectiva) o indirectas (se codifica la secuencia de entrada en un vector de estado para descodificar la secuencia de salida). Ejemplo: traductores de texto.

### 4.2. Celda LSTM

Las entradas de la celda LSTM son:
- El valor de la secuencia en el paso correspondiente x_t (entrada de la red en ese instante)
- La memoria de la celda anterior s_{t-1}
- El estado oculto de la red del paso anterior h_{t-1}

El elemento adicional s se conoce como **canal de memoria** o **celda de estado**, clave para la memoria de largo plazo. Una red LSTM puede aprender dependencias entre puntos alejados y entre puntos cercanos gracias al control del flujo de la información, que se realiza mediante **3 compuertas**. Cada una de ellas está formada por una red neuronal, una función de activación y un elemento multiplicador.

#### 4.2.1. Puerta de olvido

La puerta de olvido (*forget gate*) escoge la parte de la memoria que debe permanecer en la red. Calcula un vector f_t:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Donde W_f es la matriz de parámetros para la puerta de olvido, y σ es la función sigmoidal. Si uno de los valores del vector es cercano a 0 la red eliminará esa porción de información; si es cercano a 1 se mantendrá.

#### 4.2.2. Puerta de entrada

La puerta de entrada controla la información que se añade a la memoria de la red:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

Junto con este cálculo, se crea un vector de candidatos a formar parte de la memoria:

$$\tilde{s}_t = \tanh(W_s \cdot [h_{t-1}, x_t] + b_s)$$

A continuación se actualiza el estado de la red mediante el producto y suma de los vectores obtenidos (⊙ denota el producto de Hadamard):

$$s_t = f_t \odot s_{t-1} + i_t \odot \tilde{s}_t$$

#### 4.2.3. Puerta de salida

La puerta de salida (*output gate*) calcula el nuevo estado oculto y la nueva salida de la celda:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(s_t)$$

Por último, la salida de la celda se obtiene mediante una transformación del nuevo estado oculto:

$$y_t = \text{softmax}(W_{y,h} \cdot h_t)$$

### 4.3. Celda GRU

Las celdas GRU (*Gated Recurrent Unit*) son una simplificación de las LSTM con mayor eficiencia computacional, propuestas por Kyunghyun Cho en 2014 [7]. Han demostrado dar rendimientos muy similares a las LSTM y son altamente recomendables en conjuntos de datos no muy grandes.

En esta celda simplificada se juntan los dos vectores de estado (corto y largo plazo) en uno único s. Las compuertas utilizadas son únicamente la **puerta de reset** y la **puerta de actualización**.

La **puerta de reset** selecciona la información de la memoria utilizada en un paso concreto:

$$r_t = \sigma(W_r \cdot [s_{t-1}, x_t] + b_r)$$

La **puerta de actualización** condensa las puertas de olvido y entrada de una LSTM. Primero calcula:

$$z_t = \sigma(W_z \cdot [s_{t-1}, x_t] + b_z)$$

Luego se calcula la información candidata a añadirse a la memoria:

$$\tilde{s}_t = \tanh(W_s \cdot [r_t \odot s_{t-1}, x_t] + b_s)$$

Por último, se actualiza el estado de la red mediante una media ponderada entre el estado anterior y la información calculada:

$$s_t = (1 - z_t) \odot s_{t-1} + z_t \odot \tilde{s}_t$$

### 4.4. Consideraciones finales

La ventaja de este tipo de red es la capacidad de almacenar información pasada durante un largo periodo de tiempo gracias a la celda de estado. El control se realiza de forma difusa (las puertas pueden tomar cualquier valor real entre 0 y 1), pudiendo dejar pasar y almacenar información sólo en parte.

Otra gran ventaja es evitar los problemas derivados de la desaparición y explosión del gradiente, ya que al no necesitar multiplicarse consigo misma un número elevado de veces, evita estos inconvenientes de las RNN simples.

---

## 5. Problemas abordados

A continuación se presenta la parte práctica del trabajo, centrada en la aplicación de redes LSTM, comparándolas en ocasiones con celdas GRU. Todos los códigos han sido implementados en **MATLAB**. Como indicadores de precisión se usan el MSE (Mean Squared Error) y el RMSE (Root Mean Squared Error). En todos los casos se dividen los datos en subconjuntos de entrenamiento y test.

### 5.1. Problema 1

En este primer ejemplo se abordan un par de modelos sencillos para ilustrar las capacidades, limitaciones y resultados de una red LSTM. Se tratan las funciones:

$$f_1(x) = \sin x \qquad f_2(x) = \frac{\sin x}{x}$$

#### 5.1.1. Función sin x

Se crea un vector de 200 elementos de entrada, normalizado. Posteriormente, se crean los vectores de entrenamiento y el de test de 300 elementos. Se entrena para intentar predecir el valor de x basado en el de x − 1. La red tiene 200 capas ocultas y 250 épocas.

El RMSE obtenido es de **0.0034**. Los resultados demuestran que la arquitectura LSTM capta por completo la estructura de los datos y la reproduce a la perfección para un conjunto de datos mayor incluso al de entrenamiento, únicamente teniendo en cuenta el valor anterior. Según avanzan las iteraciones el error aumenta ligeramente, pero los resultados son excepcionales.

#### 5.1.2. Función sin(x)/x

Se utilizan los mismos parámetros que en el caso anterior. Los resultados muestran que la red también captura la tendencia de los datos pero la pierde más fácilmente, ya que se atenúa demasiado rápido. Esto indica que tener en cuenta sólo el dato anterior es demasiado básico para este ejemplo.

Al realizar el mismo ejercicio con los datos de entrada de los **7 valores anteriores**, se obtiene mucha mayor precisión, capturando toda una onda completa. Al probar con una proporción de 300 entrenamiento y 200 test, se observa que no pierde prácticamente la estructura de los datos, lo que demuestra la potencia y utilidad de estas redes neuronales.

### 5.2. Problema 2

En el segundo problema se plantea un ejemplo considerablemente más complejo. Se utiliza un conjunto de datos de Kaggle [26] sobre el clima y el nivel de contaminación cada hora durante varios años en la embajada de EE. UU. en Beijing, China. Se utilizan únicamente 4 años y se hace una división 75/25 en entrenamiento y test.

Los predictores son meteorológicos (presión, temperatura, punto de rocío, etc.). La polución[^3] tiene cierta volatilidad, ya que el valor de desviación típica es muy similar a la media.

Con configuraciones cercanas a las 500 capas ocultas e iteraciones parece sobreajustar los datos, y con menos de 200 subajusta. Se obtienen buenos resultados para **250 iteraciones y capas**.

Cuando se entrena únicamente con los datos del día anterior (24 datos anteriores), los resultados son relativamente pobres: el modelo sólo mantiene una forma general. Al probar con **predictores disponibles**, los resultados mejoran considerablemente, capturando la forma general de los datos de una manera mucho más precisa. Esto demuestra la utilidad del uso de predictores en conjuntos de datos con mucha variabilidad.

[^3]: Los valores corresponden a niveles de concentración PM2.5.

### 5.3. Problema 3

Para el tercer y último problema, se utilizan datos de precios de 3 criptoactivos (Bitcoin, Ethereum y Ripple) [11] desde 2015 hasta 2018, intentando predecir su precio mediante el uso de distintas LSTM y redes GRU. Este caso es el más complejo de predecir por su extrema volatilidad y falta de datos históricos.

La metodología aplicada a cada uno de los 3 activos es:
- Red LSTM y red GRU para un salto **diario**
- Red LSTM y red GRU para un salto **semanal**
- Red LSTM y red GRU para un salto **mensual**
- Red LSTM y red GRU para un salto **trimestral**

El número de capas ocultas y de épocas es **250** para cada una.

#### 5.3.1. Bitcoin

Se utilizan los datos de 2015 hasta 2017 para predecir los precios del primer trimestre de 2018.

**Etapa diaria.** Los datos diarios no tienen suficiente profundidad para predecir el precio al día siguiente. Los resultados son pobres tanto para LSTM como para GRU.

**Etapa semanal.** Los resultados siguen sin ser decentes. En el caso de la LSTM no tiene capacidad suficiente para reproducir la tendencia, y la red GRU se vuelve caótica, fruto seguramente de un sobreajuste.

**Etapa mensual.** Se obtienen mejores resultados. Los de la celda LSTM ajustan bien para los primeros días. Los resultados de la celda GRU son mucho mejores, siendo capaz de captar la fuerte bajada después de haberse entrenado principalmente con subidas.

**Etapa trimestral.** Sin duda el apartado que consigue mejores resultados. La celda LSTM ajusta los precios relativamente bien en corto plazo, consiguiendo el RMSE más bajo de todos. La celda GRU sí captura y predice la bajada, aunque de forma retardada, con valores más lógicos.

**Conclusión Bitcoin:** Es necesaria cierta complejidad y profundidad temporal. Con valores mensuales o trimestrales los resultados no son excesivamente malos. Las celdas GRU aportan unos resultados ligeramente mejores en las etapas mensual y trimestral.

#### 5.3.2. Ethereum

Se utilizan los datos de 2015-2017 para predecir los precios del primer trimestre de 2018.

**Etapas diaria y semanal.** Ninguna de las dos redes es capaz de realizar una predicción adecuada.

**Etapa mensual.** Los resultados son mejores. La celda GRU es la mejor, captando muy bien la subida y la resistencia en un corto periodo de tiempo. La celda LSTM es más plana, aunque con un error bastante bajo.

**Etapa trimestral.** Los resultados trimestrales de la celda LSTM no son especialmente buenos. Sin embargo, se obtiene un resultado impresionantemente bueno en la configuración de celda GRU, captando casi a la perfección la tendencia del precio.

**Conclusión Ethereum:** Donde hay menos datos y mayor volatilidad, son mejores opciones las celdas GRU, dando un resultado excepcionalmente bueno para la tendencia trimestral.

#### 5.3.3. Ripple

Para el último activo se procede de manera análoga, teniendo el mismo número de datos que el primero.

**Etapas diaria y semanal.** Ninguna de las dos redes predice de forma adecuada el precio de XRP.

**Etapa mensual.** La configuración de celdas GRU no es muy precisa, siendo mejor la predicción de las LSTM. A pesar de que las predicciones han mejorado notablemente, el resultado sigue sin tener un error menor a uno, aunque predice perfectamente la resistencia y su posterior bajada.

**Etapa trimestral.** Los resultados son excepcionales. Al contrario que con el ETH, las celdas GRU devuelven resultados menos fiables, mientras que las LSTM lo predice casi a la perfección, en especial los valores más lejanos y el soporte de la bajada.[^4]

**Conclusión Ripple:** Los resultados de tendencia diaria y semanal son claramente insuficientes. Las predicciones mensuales y trimestrales captan y predicen una bajada fuerte detrás de todo un histórico general de subidas. El caso de las celdas LSTM es relativamente preciso en ambos casos, siendo el trimestral el mejor.

[^4]: Los valores negativos se truncan a 0 en las predicciones.

---

## 6. Conclusiones y trabajo futuro

A modo de conclusión, se han podido observar en el trabajo las ventajas principales de las RNN tanto desde un punto de vista teórico como práctico, y las arquitecturas de celdas LSTM y GRU. Gracias a la memoria de la red, y a la falta de formato necesario en los datos de entrada, son altamente usadas en todo tipo de predicciones. No obstante, también tienen claras limitaciones, ya que en cuanto los datos no son periódicos y aumenta la incertidumbre, las predicciones se vuelven menos acertadas.

Se ha observado cómo, si los resultados gozan de una periodicidad o tendencia muy marcada y relativamente simple (como el caso de funciones seno, coseno, etc.), sus resultados son excepcionales. Al aumentar la incertidumbre, la volatilidad y la profundidad, se vuelven mucho más complejas de calibrar. El uso de predictores suele ser muy beneficioso, aportando más información que la tendencia en conjuntos de datos con mucha variabilidad.

Por otro lado, estas no son las únicas aplicaciones disponibles para estas redes neuronales. También son ampliamente utilizadas en el modelado del lenguaje, reconocimiento de voz, generación de texto, etc., aplicados en proyectos tan relevantes como Google Translate, "Talk to Transformer" de OpenAI o "DeepMoji". Con la llegada del deep learning en 2006, se han hecho grandes avances que permiten la creación de proyectos tan complejos como **ChatGPT** (arquitectura Transformer). Éste es similar a una RNN, con la diferencia de que utilizan atención autoregresiva y paralelismo para capturar las relaciones contextuales en el texto en vez de retroalimentación secuencial.

Esto demuestra que actualmente es un ámbito que se encuentra en el vórtice de una expansión sin precedentes, lo que sin duda conllevará a avances tecnológicos y sociales que transformarán la sociedad, siendo por tanto un área de investigación muy prometedora.

---

## A. Apéndices

### A.1. Funciones de activación

Las funciones de activación se utilizan para evitar que la red neuronal únicamente sea el paso de unos nodos a otros aplicando combinaciones lineales de los datos de entrada.

**Función de paso binario**

$$f(x) = \begin{cases} 1 & \text{si } x \geq \alpha \\ 0 & \text{si } x \leq \alpha \end{cases}$$

La función de paso binario o función escalón utiliza el parámetro de sesgo *bias* para tomar el valor 0 o 1 según se requiera (también se puede permutar el 0 por -1, siendo un paso bipolar). El parámetro α es el valor umbral de la función de activación.

**Función lineal**

$$f(x) = \beta \cdot x$$

Permite generar combinaciones lineales de las entradas.

**Función sigmoide**

$$f(x) = \frac{1}{1 + e^{-x/\rho}}$$

La función sigmoide o logística es muy utilizada ya que refleja muy bien la curva de aprendizaje de una red, penalizando resultados cercanos a 0 y 1. Depende de un parámetro ρ que determina la suavidad de la curva.

**Función tangente hiperbólica**

$$f(x) = \tanh x = \frac{\sinh x}{\cosh x} = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

Al igual que la anterior, devuelve valores en el intervalo (−1, 1) y refleja muy bien la curva de aprendizaje del modelo.

**Función rectificadora**

$$f(x) = \max(0, x)$$

La función rectificadora o ReLU (*Rectified Linear Unit*) es similar a la lineal, pero teniendo en cuenta únicamente la parte positiva. Es la función más usada debido a su bajo coste computacional.

**Función SoftMax**

$$f(x)_j = \frac{e^{x_j}}{\sum_{k=1}^{K} e^{x_k}} \quad \forall j = 1, ..., K$$

La función SoftMax o exponencial normalizada devuelve valores en el intervalo (0, 1) cuya suma es 1, siendo ampliamente usada para asignar probabilidades.

Dependiendo de las utilidades que se quieran dar, puede tener una mayor o menor utilidad una de ellas: la función SoftMax es aconsejable para representaciones en forma de probabilidades; la tangente hiperbólica tiene buen rendimiento en RNN; y la función ReLU, en redes neuronales convolucionales. Tanto la función SoftMax como la tangente hiperbólica y la sigmoide tienen buen rendimiento en las últimas capas.

### A.2. Explicación completa del algoritmo de Backpropagation

El primer paso del entrenamiento es matemáticamente similar al perceptrón o a la red Adaline, teniendo una red de N entradas y M salidas completamente conectada. En el patrón de entrenamiento p-ésimo se tiene un vector **x**_p ∈ ℝᴺ de entrada y un vector **d**_p ∈ ℝᴹ de salida. La notación utilizada para la salida de la capa l es **i**ˡ_p, siendo l una de las L capas de la red.

El error cuadrático se define como:

$$E_p = \frac{1}{2} \sum_{m=1}^{M} (d_{m,p} - y_{m,p})^2$$

El error cuadrático total tras los P patrones de entrenamiento es:

$$E = \frac{1}{P} \sum_{p=1}^{P} E_p$$

**Notación:**

- Las capas ocultas van desde l = 1 hasta l = L − 1; l = 0 es la capa de entrada y l = L la capa de salida.
- Los pesos se definen como $w^l_{j_{l-1}, j_l}$, correspondiendo al paso de la componente j_{l-1}-ésima de la capa oculta l-1 a la componente j_l-ésima de la capa oculta l.
- $\text{Neta}^l_{j_l, p}$ se define como la suma ponderada de las entradas $i^{l-1}_{j_{l-1}, p}$ por los pesos $w^l_{j_{l-1}, j_l}$, sumado al factor umbral $\theta^l_{j_l}$.

Las salidas de las sucesivas capas se calculan como:

$$\text{Neta}^1_{j_1, p} = \sum_{n=1}^{N} w^1_{n, j_1} x_{n,p} + \theta^1_{j_1}$$

$$\text{Neta}^l_{j_l, p} = \sum_{j_{l-1}=1}^{J_{l-1}} w^l_{j_{l-1}, j_l} i^{l-1}_{j_{l-1}, p} + \theta^l_{j_l}$$

$$i^l_{j_l, p} = f^l_{j_l}(\text{Neta}^l_{j_l, p})$$

Para minimizar la ecuación de coste, se calculan las derivadas parciales respecto a los pesos de la última neurona:

$$\frac{\partial E_p}{\partial w^L_{j_{L-1}, m}} = \frac{\partial E_p}{\partial f^L_m} \cdot \frac{\partial f^L_m}{\partial \text{Neta}^L_{m,p}} \cdot \frac{\partial \text{Neta}^L_{m,p}}{\partial w^L_{j_{L-1}, m}}$$

Calculando cada derivada y definiendo el término de error imputado a la neurona m-ésima:

$$\delta^L_{m,p} = (d_{m,p} - y_{m,p})(f^L_m)'(\text{Neta}^L_{m,p})$$

Las ecuaciones de modificación de pesos quedan:

$$w^L_{j_{L-1}, m}(k+1) = w^L_{j_{L-1}, m}(k) + \alpha \cdot \delta^L_{m,p} \cdot i^{L-1}_{j_{L-1}, p}$$

El error imputado a la neurona j_{L-1}-ésima de la capa L-1 es:

$$\delta^{L-1}_{j_{L-1}, p} = \delta^L_{m,p} \cdot w^L_{j_{L-1}, m} \cdot (f^{L-1}_{j_{L-1}})'$$

Esta forma se puede generalizar para calcular las ecuaciones parciales de la función de costes de cualquier neurona de cualquier capa, incluidas las ocultas, haciendo posible el entrenamiento de redes multicapa tipo perceptrón.

### A.3. Otros ejemplos

**Redes neuronales convolucionales**

Las redes neuronales convolucionales o CNN (*Convolutional Neural Network*) fueron diseñadas para simular la estructura de la corteza visual animal, por lo que su uso principal se encuentra en el procesamiento de imágenes y su visualización. Su arquitectura se basa en neuronas colocadas a lo largo de las 3 dimensiones (ancho, alto y profundidad), y dichas neuronas se conectan en cada capa a una región de la anterior.

**Red neuronal de Hopfield**

Esta red fue propuesta en 1982, con una arquitectura distinta al perceptrón y a la red tipo Adaline. A diferencia de las redes feedforward (FNN), este nuevo tipo de arquitectura planteaba la posibilidad de enviar la salida de una neurona determinada como nueva entrada de datos a todas las neuronas de la red, salvo a dicha neurona, aplicando el proceso de recursividad. Se suele utilizar en aprendizaje no supervisado al ser una red autoasociativa.

**Red de base radial**

La red de base radial (RBF) es una arquitectura específica de red creada en 1985 que calcula la salida en función de la distancia a un punto determinado llamado centro, siendo la salida una combinación lineal de funciones de activación radiales. No presenta mínimos locales donde la retropropagación pueda quedarse bloqueada, y es un tipo de red multicapa unidireccional con aprendizaje híbrido. Tiene un alto grado de generalización que, sumado a la velocidad y simpleza, hacen de esta arquitectura una de las más usadas en reconocimiento y clasificación de patrones.

**Máquinas de Boltzmann restringidas**

Las Máquinas de Boltzmann Restringidas (RBM) son un tipo de red neuronal recurrente estocástica. Presenta un tipo de modelo binario de Markov, con varias capas de variables aleatorias ocultas y una red de unidades binarias estocásticas acopladas simétricamente. Se utilizan en el reconocimiento de objetos o de voz. No tienen conexión entre unidades de la misma capa.

**Redes de creencias profundas**

La red de creencias profundas (DBN) es un tipo sofisticado de red neuronal que puede utilizar aprendizaje supervisado o no supervisado. Son similares a las RBM, salvo por el hecho de que la capa oculta de cada subred es la capa visible para la siguiente. Esta arquitectura es, a grandes rasgos, un modelo gráfico generativo compuesto por diversas capas de variables latentes con conexiones entre capas, pero no entre las unidades de cada capa individual.

---

## B. Anexos

### B.1. Código Ejemplo 1

**Primer caso (sin x):**

```matlab
clc
clear all
close all

%% Predicción de datos futuros

x=1:201;
ENTRADA= sin(x);
dataTrain=ENTRADA;
Numero_de_predicciones=300;

%media y varianza
mu = mean(dataTrain);
sig = std(dataTrain);

%datos estandarizados de entrenamiento
dataTrainStandardized = (dataTrain - mu) / sig;

%preparar los datos para la predicción
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

y = 201:200+Numero_de_predicciones;
YTest = sin(y);

% RED ARQUITECTURA LSTM

inputSize = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer ];

opts = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',150, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress');

%entrenamiento de la red
net = trainNetwork(XTrain,YTrain,layers,opts);

%predicciones
net = predictAndUpdateState(net,XTrain);

YPred(:,1)=YTrain(end);

for i = 2:Numero_de_predicciones
 [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1), ...
 'ExecutionEnvironment','cpu');
end

%desestandarizamos
YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2))

hold on
plot(ENTRADA(1:end),'blue')
plot(length(ENTRADA):(length(ENTRADA)+length(YTest)-1),YTest, 'green')
plot(length(ENTRADA):(length(ENTRADA)+length(YPred)-1),YPred,'red')
legend(["Observed (Train)", "Observed (Test)", "Forecast"], 'FontSize', 14)
ylabel("Cases", 'FontSize', 14)
title("Forecast", 'FontSize', 14)
hold off

figure,
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Observed" "Forecast"], 'FontSize', 14)
subplot(2,1,2)
stem(YPred-YTest)
ylabel("Error", 'FontSize', 14)
title("RMSE="+rmse, 'FontSize', 14)
```

> **Nota:** Para el segundo caso (sin x/x con 1 valor anterior), cambiar `sin(x)` por `sin(x)./x`. Para el tercer caso (sin x/x con 7 valores anteriores), usar el código siguiente:

**Tercer caso (sin(x)/x con 7 valores anteriores):**

```matlab
clc
clear all
close all

%% Predicción de datos futuros

x=1:201;
ENTRADA= sin(x)./x;
dataTrain=ENTRADA;
Numero_de_predicciones=300;

mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = vertcat(dataTrainStandardized(1:end-7), ...
        dataTrainStandardized(2:end-6), dataTrainStandardized(3:end-5), ...
        dataTrainStandardized(4:end-4), dataTrainStandardized(5:end-3), ...
        dataTrainStandardized(6:end-2), dataTrainStandardized(7:end-1));

YTrain = dataTrainStandardized(8:end);

y = 201:200+Numero_de_predicciones;
YTest = sin(y)./y;

inputSize = 7;
numResponses = 1;
numHiddenUnits = 200;

layers = [ sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer ];

opts = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',150, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,opts);

net = predictAndUpdateState(net,XTrain);
YPred(:,1) = YTrain(end);
[net,YPred(:,2)] = predictAndUpdateState(net,YTrain(end-6:end)');
[net,YPred(:,3)] = predictAndUpdateState(net,horzcat(YTrain(end-5:end), YPred(:,2))');
[net,YPred(:,4)] = predictAndUpdateState(net,horzcat(YTrain(end-4:end), YPred(:,2:3))');
[net,YPred(:,5)] = predictAndUpdateState(net,horzcat(YTrain(end-3:end), YPred(:,2:4))');
[net,YPred(:,6)] = predictAndUpdateState(net,horzcat(YTrain(end-2:end), YPred(:,2:5))');
[net,YPred(:,7)] = predictAndUpdateState(net,horzcat(YTrain(end-1:end), YPred(:,2:6))');

for i = 8:Numero_de_predicciones
 [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-7:i-1)', ...
 'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))

% [gráficas análogas al primer caso]
```

### B.2. Código Ejemplo 2

**Segundo caso (24 datos anteriores):**

> **Nota:** Para el primer caso con sobreajuste, cambiar `numHiddenUnits` y `MaxEpochs` a 500.

```matlab
clc
clear all
close all

data = readtable('LSTM-Multivariate_pollution.csv');
pollution = data.pollution;

mu = mean(pollution);
sig = std(pollution);
dataSt = (pollution - mu) / sig;

train_ratio = 0.75;
train_size = floor(train_ratio * length(dataSt));

num_h = 24;
XTrain = [];

for i = 1:num_h
    columna = dataSt(i:train_size-num_h+i-1);
    XTrain = [XTrain columna];
end
XTrain = XTrain';

YTrain = dataSt(25:train_size)';
YTest = pollution(train_size+1:end)';

inputSize = 24;
numResponses = 1;
numHiddenUnits = 250;

layers = [ sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer ];

opts = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,opts);

net = predictAndUpdateState(net,XTrain);
YPred(:,1) = YTrain(end);
[net,YPred(:,2)] = predictAndUpdateState(net,YTrain(end-23:end)');

for i = 3:num_h
    inputSequence = horzcat(YTrain(end-25+i:end), YPred(:, 2:(i-1)))';
    [net, YPred(:, i)] = predictAndUpdateState(net, inputSequence);
end

Numero_de_predicciones = length(YTest);

for i = num_h+1:Numero_de_predicciones
 [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-24:i-1)');
end

YPred = sig*YPred + mu;
YTrain = sig*YTrain + mu;

rmse = sqrt(mean((YPred-YTest).^2))

% [gráficas de resultados]
```

**Tercer caso (con predictores):**

```matlab
clc
clear all
close all

data = readtable('LSTM-Multivariate_pollution.csv');
datos = table2array(data(:, 2:8));
pollution = data.pollution;

mu = mean(datos);
sig = std(datos);

dataSt = datos;
for i = 1:7
    dataSt(:,i) = (datos(:,i) - mu(i)) / sig(i);
end

train_ratio = 0.75;
train_size = floor(train_ratio * length(pollution));

XTrain = dataSt(1:train_size, 1:6)';
YTrain = dataSt(1:train_size, 7)';
XTest = dataSt(train_size+1:end, 1:6)';
YTest = datos(train_size+1:end, 7)';

inputSize = 6;
numResponses = 1;
numHiddenUnits = 500;

layers = [ sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer ];

opts = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,opts);

YPred = predict(net,XTest);
YPred = YPred*sig(7) + mu(7);

rmse = sqrt(mean((YPred-YTest).^2))

% [gráficas de resultados]
```

### B.3. Código Ejemplo 3

A continuación se muestra el código para Bitcoin (BTC). El código para Ethereum (ETH) y Ripple (XRP) es análogo, cambiando el archivo de datos y el `train_ratio`.

**BTC** (`train_ratio = 0.9245`, archivo `bitcoin_usd_gwa.csv`):  
**ETH** (`train_ratio = 0.907`, archivo `ethereum_usd_gwa.csv`):  
**XRP** (`train_ratio = 0.9245`, archivo `ripple_usd_gwa.csv`)

```matlab
clc
clear all
close all

data = readtable('bitcoin_usd_gwa.csv');
datos = table2array(data(:, 3:9));
datos = datos(end:-1:1,:);

mu = mean(datos);
sig = std(datos);

train_ratio = 0.9245;
train_size = floor(train_ratio * length(datos(:,1)));

datos_train = datos(1:train_size,:);
datos_test = datos(train_size+1:end,:);

dataSt = datos_train;
for i = 1:7
    dataSt(:,i) = (datos_train(:,i) - mu(i)) / sig(i);
end

cierre = dataSt(:,4);

% Crear conjuntos de entrenamiento para 1, 7, 30 y 90 días anteriores
XTrain1 = cierre(1:end-1)';

for num_dias = [7, 30, 90]
    XTrain_tmp = [];
    for i = 1:num_dias
        columna = cierre(i:end-num_dias+i-1);
        XTrain_tmp = [XTrain_tmp columna];
    end
    % asignar a XTrain2, XTrain3, XTrain4 respectivamente
end

YTrain1 = cierre(2:end)';
YTrain2 = cierre(8:end)';
YTrain3 = cierre(31:end)';
YTrain4 = cierre(91:end)';
YTest = datos_test(:, 4)';
Numero_de_predicciones = length(YTest);

% Parámetros comunes
numResponses = 1;
numHiddenUnits = 250;
opts_base = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',150, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress');

% CASO 1: 1 día - LSTM
layers_1 = [ sequenceInputLayer(1)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer ];

net = trainNetwork(XTrain1, YTrain1, layers_1, opts_base);
net = predictAndUpdateState(net, XTrain1);
YPred11(:,1) = YTrain1(end);

for i = 2:Numero_de_predicciones
 [net,YPred11(:,i)] = predictAndUpdateState(net,YPred11(:,i-1));
end

YPred11 = sig(4)*YPred11 + mu(4);
rmse = sqrt(mean((YPred11-YTest).^2))

% CASO 1: 1 día - GRU (análogo cambiando lstmLayer por gruLayer)

% CASO 2: 7 días - LSTM
layers_7 = [ sequenceInputLayer(7)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer ];

net = trainNetwork(XTrain2, YTrain2, layers_7, opts_base);
net = predictAndUpdateState(net, XTrain2);
YPred21(:,1) = YTrain2(end);
[net,YPred21(:,2)] = predictAndUpdateState(net,YTrain2(end-6:end)');

for i = 3:7
    inputSequence = horzcat(YTrain2(end-8+i:end), YPred21(:, 2:(i-1)))';
    [net, YPred21(:, i)] = predictAndUpdateState(net, inputSequence);
end
for i = 8:Numero_de_predicciones
 [net,YPred21(:,i)] = predictAndUpdateState(net,YPred21(:,i-7:i-1)');
end

YPred21 = sig(4)*YPred21 + mu(4);
rmse = sqrt(mean((YPred21-YTest).^2))

% Los casos de 30 días (XTrain3/YTrain3) y 90 días (XTrain4/YTrain4)
% siguen la misma estructura adaptando el tamaño de la ventana.
% Para cada caso se ejecuta también la versión GRU (gruLayer en vez de lstmLayer).
```

---

## Bibliografía

[1] Sergio Esteban Altamirano Pontigo. *Aplicación de redes neuronales recurrentes y modelos de series de tiempo bayesianos a la predicción de rentabilidad de fondos de pensiones*. Tesis de la Universidad de Concepción, 2020.

[2] Amazon Web Services. *What is a neural network?*

[3] A. Bosch Rué, J. Casas-Roma, and T. Lozano Bagén. *Deep learning: principios y fundamentos*. Editorial UOC, Barcelona, primera edición en lengua castellana, 2019.

[4] Eduardo Francisco Caicedo Bravo. *Una aproximación práctica a las redes neuronales artificiales*. Universidad del Valle, 2009.

[5] Fernando Bueno Pascual. *Redes neuronales: Entrenamiento y comportamiento*. Trabajo de fin de grado, Universidad Complutense de Madrid, 2019.

[6] Alfredo Canziani. *Arquitectura de las RNNs y modelos LSTM*, 2020.

[7] Kyunghyun Cho et al. *Learning phrase representations using RNN encoder–decoder for statistical machine translation*. In Proceedings of EMNLP 2014, pages 1724–1734, Doha, Qatar, October 2014.

[8] Instituto Tecnológico de Nuevo Laredo. *Redes neuronales*, 2007.

[9] Diego Calvo. *Clasificación de redes neuronales artificiales*, 2017.

[10] Diego Calvo. *Función de activación en redes neuronales*, 2018.

[11] Gorgia. *Cryptocurrencies dataset*, 2018.

[12] Victor Grau Moreso. *Adaptación de algoritmos de aprendizaje automático para su ejecución sobre GPUs*, 2015.

[13] The Simulation Guy. *Deep learning using LSTM network to predict/forecast future values in MATLAB*. Vídeo de Youtube, 2019.

[14] Donald Hebb. *The organization of behavior: A neuropsychological theory*. Wiley, 1949.

[15] César Hernández Rodríguez. *Predicción y clasificación de series temporales bursátiles mediante redes neuronales recurrentes*. Trabajo de Fin de Grado, Universidad de Valladolid, 2020.

[16] Andrés Jiménez. *Ejemplo de una red neuronal convolucional*, 2016.

[17] MATLAB & PYTHON Deep Learning jitectechnologies. *Data prediction using deeplearning RNN (LSTM) - own data*. Vídeo de Youtube, 2020.

[18] Juan Ignacio Bagnato. *Breve historia de las redes neuronales artificiales*, 2018.

[19] Jorge López Melchor. *Implementación hardware de una red neuronal Long Short-Term Memory*. Trabajo de Fin de Grado, Universidad Complutense de Madrid, 2020.

[20] Warren S. McCulloch and Walter Pitts. *A logical calculus of the ideas immanent in nervous activity*. The Bulletin of Mathematical Biophysics, 5(4):115–133, 1943.

[21] ML4A. *Neural networks*.

[22] Ricardo Ocampo. *Ejemplo de una red de base radial*, 2014.

[23] Jesús Pérez Guerrero. *Redes recurrentes*. Trabajo de Fin de Grado, Universidad de Sevilla, 2020.

[24] Diego Andrés Restrepo Leal, Julie Pauline Viloria Porto, and Carlos Arturo Robles Algarín. *El camino a las redes neuronales artificiales*. Editorial Unimagdalena, Santa Marta, 2021.

[25] Frank Rosenblatt. *The perceptron: a probabilistic model for information storage and organization in the brain*. Psychological Review, 65(6):386–408, 1958.

[26] Rupak Bob Roy. *Air pollution forecasting - LSTM multivariate*, 2022.

[27] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. *Learning representations by back-propagating errors*. Nature, 323(6088):533–536, 1986.

[28] David E. Rumelhart and James L. McClelland. *Parallel distributed processing: explorations in the microstructure of cognition. Volume 1. Foundations*. 1986.

[29] Juan Miguel Sierra Ramos. *Introducción a las redes neuronales artificiales*. Trabajo Fin de Grado, Universidad Complutense de Madrid, 2022.

[30] Miguel Sotaquirá. *Introducción a las redes neuronales recurrentes*, 2019.

[31] Miguel Sotaquirá. *¿Qué son las redes LSTM?*, 2019.

[32] Rishabh Upadhyay. *Ejemplo de una red de creencias profundas*, 2017.

[33] Wikipedia. *Red neuronal artificial — Wikipedia, la enciclopedia libre*, 2023.
