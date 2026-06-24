# Desarrollo de aplicación basada en modelos de clasificación para diagnóstico de enfermedades cardiovasculares

**Universidad Complutense de Madrid — Facultad de Matemáticas**
**Trabajo de Fin de Grado**

Supervisor: Ana Carpio

**Sergio Gil Gavela**
Doble grado en Matemáticas e Ingeniería Informática
Curso académico 2020-21

---

## Resumen

Según la OMS, las enfermedades cardiovasculares son la primera causa de muerte en el mundo. Para las personas que padecen enfermedades cardiovasculares, es crucial la detección precoz de la existencia de la enfermedad o de un infarto cardíaco. Con este propósito A. Janosi et al. crearon una base de datos en la que se recogían 14 parámetros médicos de pacientes a su entrada en el hospital. Algunos de estos pacientes finalmente presentaron una enfermedad cardiovascular y algunos resultaron estar sanos.

El propósito de este trabajo es, a partir de la base de datos de A. Janosi et al., crear un modelo que prediga la probabilidad de padecer una enfermedad cardíaca. Para ello se evaluará el rendimiento de los siguientes modelos de clasificación y regresión sobre la base de datos de A. Janosi et al.:

- Naive Bayes
- KNN
- Regresión linear
- Regresión logística
- Random forest

Estos modelos se evaluarán atendiendo principalmente a dos indicadores de la bondad de un modelo:

- AUC
- Valor F1

Tras la evaluación se determinará cuál es el modelo que mejores resultados ofrece y se desarrollará una interfaz gráfica que integre dicho modelo. Esta interfaz gráfica en forma de calculadora permitirá al personal sanitario introducir los parámetros médicos e imprimirá por pantalla la probabilidad de que el paciente padezca una enfermedad cardiovascular.

**Palabras clave:** Modelos de clasificación, modelos de regresión, enfermedades cardiovasculares, Naive Bayes, KNN, regresión linear, regresión logística, random forest, AUC, valor F1.

---

## Abstract

According to WHO, cardiovascular diseases are the primary cause of death in the world. It is crucial for people who suffer from heart diseases an early diagnosis of the disease or of a heart attack. For this purpose, A. Janosi et al. developed a data set based on 14 medical parametres measured when patients entered hospital. Some of these patients finally suffered from a heart disease and some were healthy.

This project purpose is to develop a model based on A. Janosi et al. data set which predicts the probability of suffering a heart disease. To this end, several classification and regression models will be tested:

- Naive Bayes
- KNN
- Linear regression
- Logistic regression
- Random forest

This models will be evaluated accordingly to two metrics which indicate how good a model is:

- AUC
- F1 score

After evaluating the models it will be determined which model is the one that performed best. The best model will be integrated into a graphical interface. The graphical interface will be a calculator in which health professionals will be able to input the parameters and it will display the probability that a patient suffers from a heart disease.

**Key words:** Classification models, regression models, cardiovascular diseases, Naive Bayes, KNN, linear regression, logistic regression, random forest, AUC, F1 score.

---

## Índice

1. [Introducción](#1-introducción)
   - 1.1. [Objetivos](#11-objetivos)
2. [Fundamentos teóricos](#2-fundamentos-teóricos)
   - 2.1. [Qué es un modelo de aprendizaje automático](#21-qué-es-un-modelo-de-aprendizaje-automático)
   - 2.2. [Introducción a los modelos de clasificación y regresión](#22-introducción-a-los-modelos-de-clasificación-y-regresión)
3. [Presentación de los modelos](#3-presentación-de-los-modelos)
   - 3.1. [Clasificador Naive Bayes](#31-clasificador-naive-bayes)
   - 3.2. [KNN](#32-knn)
   - 3.3. [Regresión linear](#33-regresión-linear)
   - 3.4. [Regresión logística](#34-regresión-logística)
   - 3.5. [Random Forest](#35-random-forest)
4. [Presentación de la base de datos](#4-presentación-de-la-base-de-datos)
5. [Métricas](#5-métricas)
6. [Metodología](#6-metodología)
7. [Marcos de trabajo](#7-marcos-de-trabajo)
8. [Resultados](#8-resultados)
   - 8.1. [Naive Bayes](#81-naive-bayes)
   - 8.2. [KNN](#82-knn)
   - 8.3. [Regresión linear](#83-regresión-linear)
   - 8.4. [Regresión logística](#84-regresión-logística)
   - 8.5. [Random Forest](#85-random-forest)
   - 8.6. [Conclusiones de la evaluación de modelos](#86-conclusiones-de-la-evaluación-de-modelos)
9. [Manual de la interfaz gráfica](#9-manual-de-la-interfaz-gráfica)
   - 9.1. [Introducción](#91-introducción)
   - 9.2. [Cómo descargar la carpeta](#92-cómo-descargar-la-carpeta)
   - 9.3. [Contenido de la carpeta](#93-contenido-de-la-carpeta)
   - 9.4. [Antes de ejecutar la aplicación](#94-antes-de-ejecutar-la-aplicación)
   - 9.5. [Ejecutar la aplicación](#95-ejecutar-la-aplicación)
   - 9.6. [Introducción de datos](#96-introducción-de-datos)
10. [Conclusiones](#10-conclusiones)
11. [Referencias](#referencias)

---

## 1. Introducción

Según la OMS [1], las enfermedades cardiovasculares son la primera causa de muerte en el mundo (ver figura 1), se estima que anualmente mueren 17,5 millones de personas por esta causa, aproximadamente un 30% de las defunciones globales. Las enfermedades cardiovasculares son un conjunto de trastornos del corazón y de los vasos sanguíneos que se clasifican en distintos tipos como la hipertensión arterial, la insuficiencia cardíaca o las miocardiopatías entre otros.

> **Figura 1:** Principales causas de muerte en el año 2016 [3].

Existe un porcentaje de la población considerada de alto riesgo de infarto cardiovascular. Esto se debe a la presencia de uno o más factores de riesgo como la hipertensión, la obesidad, el tabaquismo, el alcoholismo o colesterol alto en sangre. No obstante, en la mayoría de los casos, las enfermedades cardiovasculares pueden ser prevenidas incidiendo sobre algunos hábitos diarios como el consumo de tabaco, la dieta, la obesidad, el consumo de alcohol o el sedentarismo [2].

Es crucial para las personas con enfermedades cardiovasculares o que pertenecen a un grupo de riesgo la detección precoz de un infarto cardíaco. Con este propósito, A. Janosi et al. [4] crearon en 1988 una base de datos para estudiar a los pacientes ingresados en el hospital clínico de Cleaveland. Esta base de datos recoge 75 parámetros médicos de 303 pacientes de entre 29 y 77 años de edad a su entrada en el hospital. Para su base de datos, A. Janosi et al. seleccionaron parámetros médicos que potencialmente pueden contribuir a la presencia de una enfermedad cardiovascular. Finalmente, algunos de los pacientes padecían una enfermedad cardiovascular mientras que otros resultaron estar sanos.

### 1.1. Objetivos

El propósito de este trabajo es analizar la base de datos creada por A. Janosi et al. y crear un modelo predictivo para calcular la probabilidad de padecer una enfermedad cardiovascular. Con estos fines, hemos marcado los siguientes objetivos para el trabajo:

- Identificar qué grado de correlación existe entre cada uno de los parámetros recogidos en la base de datos y el riesgo de infarto cardiovascular.
- Elaborar varios modelos predictivos procesando los datos mediante técnicas de aprendizaje automático como modelos de regresión y modelos de clasificación. En concreto se usarán los siguientes modelos:
  - Naive Bayes
  - KNN
  - Regresión linear
  - Regresión logística
  - Random forest

  Para ello se elaborarán scripts en Python que implementen los modelos y optimicen sus parámetros.

- Determinar qué modelo ofrece mejores resultados. Para ello nos fijaremos principalmente en dos métricas[^1] que servirán para comparar los modelos y decidir cuál es el mejor:
  - Valor F1
  - AUC

- Crear una interfaz gráfica en la que un profesional sanitario pueda introducir los datos médicos de un paciente y obtenga el riesgo de que el paciente sufra una enfermedad cardiovascular. Para implementar esta interfaz gráfica se usará el modelo con mejores resultados.

[^1]: En este trabajo la palabra métrica se empleará como sinónimo de indicador para medir la bondad de un modelo. En ningún caso tendrá que ver con el concepto matemático de distancia.

---

## 2. Fundamentos teóricos

### 2.1. Qué es un modelo de aprendizaje automático

Según Microsoft [5] un modelo de aprendizaje automático es un archivo que ha sido entrenado para reconocer determinados tipos de patrones. Un modelo se entrena sobre un conjunto de datos que ya han sido desglosados e interpretados. El propósito es que una vez entrenado sea capaz por sí mismo de desglosar e interpretar nuevos datos que no ha visto antes basándose en todo lo aprendido de los datos que le fueron suministrados durante el entrenamiento (ver figura 2).

> **Figura 2:** Esquema del funcionamiento de un modelo de machine learning [6].

Los datos que se suministran a un algoritmo encargado de entrenar un modelo pueden ser de muy diversos tipos, desde indicadores financieros relacionados con la bolsa a imágenes como en los coches autónomos, o sonidos como en herramientas de procesamiento de lenguaje natural. La idea siempre es la misma: generar un modelo que pueda interpretar datos por sí mismo. Esta herramienta es muy potente cuando se utiliza para predecir lo que va a suceder en un futuro basándose en los datos pasados. Por ejemplo, es común en los bancos emplear modelos para predecir impagos o para calcular el riesgo al conceder un crédito.

Existen muchos algoritmos distintos de machine learning pero la clasificación más general es la que hace Recuero [7]:

- **Algoritmos de aprendizaje supervisado:** Estos algoritmos trabajan con datos que han sido etiquetados e intentan encontrar una función que dadas unas variables de entrada, les asigne la etiqueta de salida adecuada. Este tipo de algoritmos se entrena con una base de datos que utiliza para predecir los valores de salida. Este tipo de algoritmos se utilizan en problemas de regresión y de clasificación, que son los que se afrontarán en este trabajo.

- **Algoritmos de aprendizaje no supervisado:** En este caso los algoritmos no disponen de datos etiquetados sino que solamente conocen los datos de entrada. Como no existen datos de salida el algoritmo tiene que intentar explorar los datos de entrada y agruparlos basándose en sus similitudes. Estas agrupaciones, al ser de carácter exploratorio, no siempre tienen un significado o utilidad. Los algoritmos de clustering son el exponente más conocido de algoritmos de aprendizaje no supervisado.

### 2.2. Introducción a los modelos de clasificación y regresión

Como se mencionaba en la anterior sección, tanto los modelos de clasificación como los de regresión son métodos de aprendizaje automático supervisado. Esto significa que aprenden a partir de datos que han sido etiquetados, es decir, se tienen las variables de entrada y se conoce la salida. Como se verá en esta sección, los modelos de clasificación y de regresión tienen muchas similitudes y habitualmente se utilizan para solucionar problemas similares.

Los modelos de clasificación se utilizan cuando la salida deseada es una etiqueta discreta, es decir, cuando la salida pertenece a un conjunto finito de resultados posibles. Cuando el conjunto está formado solamente por dos elementos (sí o no), se habla de clasificación binaria. Un ejemplo de problema de clasificación binaria es el que afrontan las plataformas de correo electrónico para clasificar el e-mail como spam o no-spam.

Por otra parte, los modelos de regresión son útiles cuando la salida esperada es continua, esto quiere decir que la salida es un valor numérico calculado mediante una función que toma como entrada los datos que se han suministrado al algoritmo. Un ejemplo de problema de regresión es cuanto va a tardar un vehículo en llegar a su destino; aplicaciones como Google Maps emplean datos de viajes anteriores en los que se tiene en cuenta a qué velocidad viajaba el vehículo, el tiempo meteorológico, la concurrencia de la carretera o la hora del día para predecir al usuario cuanto va a tardar en realizar su viaje.

> **Figura 3:** Los problemas de clasificación tienen como salida valores discretos mientras que los problemas de regresión tienen valores continuos [8].

En la imagen 3 se puede observar que el objetivo de un clasificador binario es establecer una línea de separación entre las dos clases de datos, pero no siempre esto es posible. Algunos algoritmos, como el "perceptron", no convergen si las clases no pueden separarse por una frontera lineal (ver figura 4). Además, el tipo de datos, el número de características disponibles o el número de clases son parámetros que influyen en el comportamiento de los algoritmos de clasificación. Por este motivo, en la práctica, es habitual comparar el comportamiento de diferentes algoritmos para poder escoger el que más precisión ofrece para un problema concreto.

> **Figura 4:** Algunos algoritmos de clasificación ofrecen los mejores resultados para conjuntos de datos separables linealmente pero no convergen si el conjunto no es separable [9].

Estos dos tipos de algoritmos, a pesar de sus diferencias, se emplean para resolver problemas de clasificación. Esto se debe a que la mayoría de algoritmos de clasificación binaria dan los resultados en forma de probabilidades (el correo es spam con 0.92 de probabilidad) y por su parte los algoritmos de regresión se pueden acotar para que el resultado esté entre 0 y 1. En la práctica ambos tipos de algoritmos están siendo usados para predecir una probabilidad. Este precisamente es el problema que nos atañe en este trabajo.

Cuando se aborda un problema de probabilidad como los problemas de clasificación, es importante fijar un límite a partir del cual se considerar que se ha acertado. El límite deberá ser más alto cuanto más alto sea el coste de equivocarse comparado con el beneficio de obtener la respuesta correcta. Existen dos tipos de errores: cuando el modelo estima un positivo que en realidad es un negativo y cuando el modelo estima un negativo que es un positivo. El coste de cometer un error de un tipo u otro no es habitualmente el mismo como se puede ver en los ejemplos del cuadro 1.

| | Filtrado de Spam | Diagnóstico médico |
|---|---|---|
| **Falso negativo** | Podría entrar un virus en el ordenador | Se le dice al paciente que no está enfermo pero sí lo está |
| **Falso positivo** | El usuario podría quedarse sin leer un e-mail | Se trata al paciente con medicamentos que no necesita |

**Cuadro 1:** El coste de un falso negativo y un falso positivo no siempre es el mismo [10].

En la sección 5 se verá en más profundidad los tipos de errores y cómo se utilizan para calcular las métricas que evalúan cómo de bueno es un modelo. Además, se verá que cuantos más datos se utilicen para entrenar al clasificador, siempre que los datos sean buenos, menor cantidad de estos tipos de errores se cometen. No obstante, por muchos que sean los datos suministrados nunca dejará de existir un error mínimo en los modelos de clasificación.

---

## 3. Presentación de los modelos

En esta sección se hace una presentación detallada de los modelos de clasificación y regresión que van a ser entrenados con los datos de la base de datos médica creada por A. Janosi et al. [4]. Todos los modelos tienen como objetivo ser lo más precisos posible para clasificar nuevos datos pero para ello emplean principios y mecanismos diferentes. Por este motivo, algunos modelos se comportan mejor para algunos tipos de conjuntos de datos. Entender los modelos puede ofrecer una visión general sobre cómo se va a comportar frente a los datos, pero en la práctica, como se detalla en la sección 2.2, hay que probar todos los modelos y analizar cuál es el que mejor rendimiento ha ofrecido.

Antes de exponer los modelos conviene tener claras algunas definiciones.

> **Definición 1 (Característica)** Se denomina característica a cada uno de los parámetros recogidos en una base de datos de los cuales potencialmente depende el resultado. Cada característica se corresponde con una columna de una base de datos menos la última.

> **Definición 2 (Vector de características)** Se denomina vector de características al conjunto de todas las características.

> **Definición 3 (Clase)** Se denomina clase a cada uno de los posibles resultados. La clase se corresponde con la última columna de una base de datos.

De esta manera si se tiene una base de datos para predecir qué clase de producto va a comprar un usuario en función de su edad, género y productos anteriores comprados; edad, género y productos anteriores comprados serían las características del modelo; y juguete, libro y ordenador podrían ser las clases del modelo.

### 3.1. Clasificador Naive Bayes

El clasificador Naive Bayes o Bayes ingenuo [12] es uno de los algoritmos más simples que existen para clasificación, no obstante, también es uno de los más potentes para abordar conjuntos de datos muy grandes. Este algoritmo se fundamenta en el Teorema de Bayes:

> **Teorema 1 (Bayes)** Sean $\{A_1, A_2, \ldots, A_i, \ldots, A_n\}$ un conjunto de sucesos mutuamente excluyentes y exhaustivos tales que la probabilidad de cada uno es distinta de cero ($P[A_i] \neq 0$ para $i = 1, 2, \ldots, n$). Si $B$ es un suceso cualquiera del que se conocen las probabilidades condicionales $P(B|A_i)$, entonces la probabilidad $P(A_i|B)$ viene dada por:
>
> $$P(A_i|B) = \frac{P(B|A_i)\,P(A_i)}{P(B)} \tag{1}$$
>
> donde $P(A_i)$ es la probabilidad del suceso $A_i$ a priori, $P(B|A_i)$ es la probabilidad condicionada de $B$ por $A_i$, y $P(A_i|B)$ son las probabilidades a posteriori [11].

El clasificador Naive Bayes asume que el efecto de una característica particular en el resultado es independiente del resto de características. Por ejemplo, si se supone que un solicitante de préstamo es deseable o no en función de sus ingresos, edad, historial de préstamos y ubicación; el clasificador Naive Bayes asume que estas características son independientes incluso si es evidente que no lo son como en el caso de los ingresos y la edad. Por este motivo a este clasificador se le denomina ingenuo.

Condensando todo lo anterior se puede obtener la fórmula que utiliza el clasificador Naive Bayes para determinar la probabilidad de una clase determinada:

$$P(C|F_1, \ldots, F_n) = \frac{P(C)\,P(F_1, \ldots, F_n|C)}{P(F_1, \ldots, F_n)} \tag{2}$$

donde $C$ es la clase de la cual se está calculando la probabilidad y $(F_1, \ldots, F_n)$ es el vector de características. Se puede observar que el numerador es una probabilidad compuesta:

$$P(C, F_1, \ldots, F_n) = P(C)\,P(F_1|C)\,P(F_2|C, F_1) \cdots P(F_n|C, F_1, \ldots, F_{n-1}) \tag{3}$$

y si se asume que $F_i$ es independiente de $F_j$ para $i \neq j$, es decir, $P(F_i|C, F_j) = P(F_i|C)$, se tiene que:

$$P(C, F_1, \ldots, F_n) = P(C)\prod_{i=1}^{n} P(F_i|C) \tag{4}$$

por lo que la distribución condicional de una clase sobre el vector de características se puede expresar como:

$$P(C, F_1, \ldots, F_n) = \frac{1}{Z}\,p(C)\prod_{i=1}^{n} P(F_i|C) \tag{5}$$

donde $Z$ es un factor que depende únicamente de $F_1, \ldots, F_n$ [12].

La simplificación que hace el clasificador Naive Bayes respecto a la independencia de las características le confiere una serie de ventajas y desventajas:

**Ventajas:**
- El clasificador Naive Bayes se basa en la teoría de la probabilidad clásica, por lo que tiene una base matemática sólida y resultados estables.
- Funciona muy bien en conjuntos de entrenamiento a gran escala cuando el número de características no es excesivamente grande.
- Funciona muy bien para tareas de clasificación múltiple, es decir, cuando hay más de dos clases.
- Funciona muy bien para características categóricas, es decir, características que toman valores en un conjunto finito.

**Desventajas:**
- Al suponer que las características son independientes, en caso de no ser así y ser fuertemente dependientes, el modelo no será bueno. [13]

### 3.2. KNN

KNN (K-Nearest Neighbours) [14] es un modelo de clasificación por vecindad, esto quiere decir que está basado en la búsqueda de los $k$ elementos más cercanos al nuevo elemento que se quiere clasificar. Esto plantea algunas dudas:

- ¿Cómo se van a ubicar los datos en el espacio?
- ¿Qué definición de distancia se va a usar?

La primera respuesta es sencilla, para ubicar un dato en el espacio se utiliza su vector de características. De esta manera, si en una base de datos los vectores de características tienen $n$ elementos, se necesitará un espacio $n$-dimensional para ubicar los datos. En cuanto a la distancia, lo más sencillo es emplear la distancia euclídea:

$$d_E(P, Q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2} \tag{6}$$

donde $(p_1, \ldots, p_n)$ y $(q_1, \ldots, q_n)$ son los vectores de características de los datos de entrada $P$ y $Q$. No obstante, a veces se ponderan las distancias de cada característica de manera que determinadas características influyan más sobre la clasificación final que otras o se utilizan definiciones alternativas de distancia como la distancia Manhattan.

El siguiente ejemplo ilustra cómo se comporta KNN. Se tiene una base de datos con dos características que son utilizadas para clasificar los datos en cuadrados y triángulos. En la figura 5 se han ubicado correspondientemente los datos en el espacio. Ahora se quieren utilizar estos datos para clasificar el círculo verde, que representa un nuevo dato para el cual se tienen las características pero no si es un cuadrado o un triángulo. Para clasificarlo se deben de tomar los $k$ puntos más cercanos y en base a ellos elaborar la predicción. Por ejemplo si se tomase $k = 3$, entonces sería clasificado como un triángulo ya que de sus tres vecinos más cercanos, dos son triángulos y solo uno es un cuadrado. Sin embargo, si se tomase $k = 5$ sería clasificado como un cuadrado.

> **Figura 5:** El círculo de menor diámetro corresponde a $k = 3$ mientras que el de mayor diámetro corresponde a $k = 5$ [15].

Este ejemplo pone de manifiesto una importante cuestión, ¿cuál es el mejor valor para $k$? Una vez más, la mejor elección de $k$ depende de los datos concretos del problema. En general, valores pequeños para $k$ reducen el coste computacional pero su efectividad está condicionada a la homogeneidad dentro de las clases. Por otro lado, valores grandes de $k$ reducen el efecto de las clases no homogéneas pero crean límites entre clases parecidas además de aumentar el coste computacional.

Para elegir un buen $k$ habitualmente se prueban varios valores y se selecciona el más óptimo. No obstante, el valor más comúnmente usado es $k = \sqrt{N}$ donde $N$ es el número de datos en el conjunto de entrenamiento (ver demostración en [14]).

**Ventajas:**
- Ofrece muy buen rendimiento cuando el conjunto de datos es grande.
- Funciona muy bien para tareas de clasificación múltiple, es decir, cuando hay más de dos clases.
- Las distancias pueden ponderarse para valorar más algunas características que otras a la hora de calcular la proximidad entre dos datos.

**Desventajas:**
- Hay que calcular el mejor valor de $k$, un proceso de prueba y error en el que hay que ejecutar numerosas veces el algoritmo.
- La función de distancia debe de ser representativa para que el algoritmo sea preciso; si se ponderan mal las distancias de cada característica se obtiene un mal modelo.
- Computacionalmente es un algoritmo muy costoso ya que hay que calcular la distancia de cada dato a todo el resto de datos de la base de datos, esto es un orden de $N^2$. [16]

### 3.3. Regresión linear

La regresión lineal [24], [25], [26] intenta modelar la relación entre varias variables dependientes y una o más variables independientes. En el caso de este trabajo se utilizará la regresión linear para modelar la relación de múltiples características y una variable dependiente o variable de respuesta; a este tipo de regresión linear se la denomina múltiple.

Sea

$$(X|y) = \{x_{i1}, x_{i2}, \ldots, x_{ip}, y_i\}_{i=1}^{n} \tag{7}$$

una base de datos, donde $\{x_{ij}\}_{i=1}^{n}$ representa la columna correspondiente a la característica $j$ y $\{y_i\}_{i=1}^{n}$ es la columna objetivo. La regresión linear asume que existe una relación linear entre los $p$ vectores asociados a las columnas de cada característica y la columna objetivo, es decir,

$$y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i = \mathbf{x}_i^T \boldsymbol{\beta} + \varepsilon_i, \quad i = 1, \ldots, n \tag{8}$$

donde $\varepsilon_i$ se denomina variable de error y se comporta como una variable aleatoria que añade ruido a la relación linear entre las características y la variable dependiente. Es habitual expresar la ecuación 8 en su forma matricial:

$$\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon} \tag{9}$$

El objetivo final de una regresión linear es hallar el valor de los elementos del vector $\boldsymbol{\beta}$. Para ello la regresión linear hace algunas suposiciones:

1. **Linealidad** entre las variables independientes $\{x_1, x_2, \ldots, x_p\}$ y la variable objetivo $y$.
2. **Homocedasticidad:** Esto quiere decir que la varianza del error de la variable de respuesta es constante a lo largo de las observaciones (ver figura 6).
3. **Independencia** entre las observaciones, es decir, el vector de características de cada fila de la base de datos es independiente del resto.
4. **Normalidad:** La variable dependiente $y$ y los términos de error tienen una distribución normal.

> **Figura 6:** La gráfica superior corresponde a una regresión linear, la inferior muestra la distribución del error, cuya varianza se mantiene constante. Se puede afirmar que el modelo de la gráfica superior tiene la propiedad de homocedasticidad [22].

No obstante, las anteriores suposiciones no son suficiente para hallar los valores de $\boldsymbol{\beta}$, sino que estos han de ser aproximados mediante técnicas de estimación. Con este propósito se han desarrollado múltiples técnicas que difieren en sus complejidades algorítmicas o las suposiciones que añaden a las cuatro suposiciones básicas de la regresión linear. Nos vamos a centrar en la técnica más extendida para aproximar $\boldsymbol{\beta}$: la aproximación por mínimos cuadrados.

Sea $\vec{x}_i = [x_{i1}, \ldots, x_{ip}]$ la variable independiente que no queda fijada y $\vec{\beta} = [\beta_0, \ldots, \beta_p]$ los parámetros del modelo. Se sabe que la predicción del modelo es:

$$y_i \approx \beta_0 + \sum_{j=1}^{p} \beta_j \times x_{ij} = \vec{\beta}\,\vec{x}_i \tag{10}$$

En la aproximación por mínimos cuadrados los parámetros óptimos son los que minimizan la suma de la diferencia de cuadrados:

$$\left\|\vec{\beta} - \vec{v}\right\|^2 = \sum_{i=1}^{p}(\beta_i - v_i)^2 \tag{11}$$

sea mínima. Dicho de otra manera:

$$\vec{v} = \arg\min_{\vec{\beta}}\, L(\vec{\beta}) = \arg\min_{\vec{\beta}} \sum_{i=1}^{p}\left(\vec{\beta}\,\vec{x}_i - y_i\right)^2 \tag{12}$$

donde $L(\vec{\beta})$ es la función de pérdida. Si se expresa la función de pérdida en forma matricial:

$$L(\vec{\beta}) = \left\|X\vec{\beta} - \mathbf{y}\right\|^2 = \mathbf{y}^T\mathbf{y} - \mathbf{y}^T X\vec{\beta} - \vec{\beta}^T X^T \mathbf{y} + \vec{\beta}^T X^T X\vec{\beta} \tag{13}$$

Como la función de pérdida es convexa (ver demostración en [23]) alcanzará su mínimo en el punto en el que el gradiente sea 0:

$$\frac{\partial L(\vec{\beta})}{\partial \vec{\beta}} = -2X^T\mathbf{y} + 2X^TX\vec{\beta} = 0 \tag{14}$$

con lo que nos queda:

$$\vec{v} = (X^TX)^{-1}X^T\mathbf{y} \tag{15}$$

### 3.4. Regresión logística

La regresión logística [27], al contrario de lo que su nombre indica, es un modelo de clasificación y no de regresión, es decir, su objetivo es clasificar nuevos datos basándose en los datos de la base de datos que ya están clasificados. Esto quiere decir que la regresión logística asigna clases de un conjunto finito a los nuevos datos. Existen dos tipos de regresión logística:

- **Binaria:** Los datos son clasificados en dos clases.
- **Multinomial:** Los datos son clasificados en más de dos clases.

En este trabajo se usará la regresión logística binaria ya que el objetivo es clasificar los datos en "paciente con enfermedad cardiaca" y "paciente sano". Por este motivo la explicación se centrará en la regresión logística binaria.

La regresión logística modela la probabilidad de una clase utilizando una función similar a la regresión linear pero acotando la función de error entre 0 y 1. Para ello utiliza la función sigmoide:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \tag{16}$$

La forma en la que se define matemáticamente el modelo de regresión logística es:

$$\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p \tag{17}$$

donde $p$ es la probabilidad del evento de interés, que en nuestro caso será "tener una enfermedad cardiaca". Para hacer la explicación más comprensible vamos a suponer que la variable dependiente depende de una única variable independiente:

$$\log\frac{p}{1-p} = \beta_0 + \beta_1 x \tag{18}$$

si despejamos $p$ en la ecuación nos queda:

$$p(x) = \sigma(\beta_0 + \beta_1 x) = \frac{1}{1 + e^{\beta_0 + \beta_1 x}} \tag{19}$$

Si se dibujase la anterior función para ciertos valores $\beta_0$ y $\beta_1$, se obtendría una gráfica similar a la de la figura 7. La gráfica se interpreta de la siguiente manera: supongamos que las cruces verdes son pacientes sanos y los rombos rojos son pacientes enfermos. Los pacientes han sido ubicados en el eje $x$ en función del valor que tomaba la variable independiente $x$ para cada uno de ellos (esta variable podría ser nivel de glucosa en sangre). A partir de estos valores se ha calculado $\beta_0$ y $\beta_1$, y se ha dibujado una función sigmoide. Si ahora se nos proporcionan nuevos datos, representados por círculos azules, basta con ubicarlos sobre el eje $x$ atendiendo a su nivel de glucosa en sangre y observar que valor toma la función sigmoide para cada uno de ellos. Ese valor representa la probabilidad de que el paciente esté sano. Es habitual determinar un umbral, por ejemplo $t = 0.6$, a partir del cual los datos serán clasificados en una clase o en otra, es decir, como sanos o como enfermos.

> **Figura 7:** Las cruces verdes representan datos pertenecientes a la clase 1 y los rombos rojos a la clase 2. Los círculos azules son nuevos datos [27].

Evidentemente la clave de la regresión logística es aproximar lo mejor posible los parámetros $\beta = \beta_0, \ldots, \beta_p$ para que el modelo haga buenas predicciones, o lo que es lo mismo, minimizar la función de pérdida al igual que hicimos con la regresión linear. La función de pérdida de la regresión logística es:

$$L(p(x), y) = \begin{cases} -\log(p(x)) & \text{si } y = 1 \\ -\log(1 - p(x)) & \text{si } y = 0 \end{cases} \tag{20}$$

donde $y$ representa la variable dependiente, es decir, las clases del modelo.

Para calcular el mínimo de esta función se utiliza el método del gradiente descendente (ver [28]), que en el fondo es la base de aprendizaje de numerosas técnicas de machine learning.

**Ventajas:**
- Es una técnica simple y que no requiere de grandes recursos computacionales.
- Los resultados son altamente interpretables, no es una caja negra. Tras elaborar una predicción es fácil observar qué peso ha tenido cada una de las características.

**Desventajas:**
- Es importante eliminar las características que muestren gran correlación entre ellas y las que no estén relacionadas con la salida.
- La variable objetivo debe de ser linealmente separable (ver figura 4); de no ser así el modelo no clasificará correctamente (en el caso de este trabajo sí lo es).

### 3.5. Random Forest

El último de los modelos que se usarán para procesar la base de datos es "Random Forest" [30], [32], [33], también conocido por su traducción al castellano "Bosques aleatorios". "Random Forest" forma parte de un conjunto de algoritmos utilizados en machine learning llamados árboles de decisión. En un árbol de decisión se pueden identificar los siguientes elementos (ver figura 8):

- **Nodo:** Cada nodo define el momento en el que se debe de tomar una decisión entre varias alternativas.
- **Hoja:** Cada una de las hojas del árbol está asociada a una clase, que es la solución dada por el árbol al problema de clasificación.

> **Figura 8:** En un árbol de decisión los nodos representan decisiones y las hojas soluciones [29].

La idea detrás de un árbol de decisión es aplicar recursivamente los siguientes pasos:

1. Del conjunto de características elegir la mejor utilizando ASM (Attribute Selection Measure)[^2] para dividir el conjunto de datos.
2. Convertir esa característica en un nodo de decisión de manera que se divida el conjunto de datos en varios subconjuntos.

[^2]: Algunos de los ASM más utilizados son Ganancia de información, Índice de Gini o Ratio de ganancia.

Estos pasos se repiten recursivamente para cada subconjunto de datos hasta que:

- Todos los datos pertenecen al mismo valor de un atributo.
- No quedan atributos.
- No quedan elementos en la base de datos.

Los árboles de decisión son ampliamente utilizados ya que son muy robustos frente a características irrelevantes; no obstante, si el árbol es demasiado profundo, es decir, se toman demasiadas decisiones se arriesga a incurrir en overfitting. Se dice que se ha incurrido en overfitting cuando el modelo se adapta excesivamente a un determinado conjunto de datos y por tanto puede fallar a la hora de clasificar instancias que no pertenezcan a ese conjunto de datos.

Se puede decir que los bosques de decisión aúnan los esfuerzos de muchos árboles de decisión para obtener un desempeño mejor que un único árbol de decisión. El algoritmo de entrenamiento de Random Forest aplica una técnica denominada bootstrap aggregating o bagging [31]. Dado un conjunto de entrenamiento con sus vectores de características y sus respectivas clases $(X|y) = \{x_{i1}, x_{i2}, \ldots, x_{ip}, y_i\}_{i=1}^{n}$, bagging selecciona una muestra aleatoria con remplazamiento de la base de datos y elabora un árbol de decisión que modele esa muestra. Este proceso se repite $B$ veces de manera que se obtienen $B$ árboles de decisión. De esta manera, para $b = 1, \ldots, B$ se extraen muestras aleatorias $(X_b|Y_b)$ y se obtiene un árbol de decisión $f_b$. Si ahora se quiere hacer una predicción de clase para una instancia no clasificada $x'$ se puede:

- Para un problema de regresión calcular el promedio de los resultados de los $B$ árboles:

$$\hat{f} = \frac{1}{B}\sum_{b=1}^{B} f_b(x') \tag{21}$$

- Para un problema de clasificación tomar la clase predicha por cada árbol de decisión y clasificar $x'$ en la clase que haya sido predicha por más árboles (ver figura 9).

> **Figura 9:** En un problema de clasificación el resultado de un bosque de decisión es la clase con más votos [30].

La principal ventaja de bootstrap aggregating es que disminuye la varianza del modelo[^3] sin aumentar el sesgo[^4]. En otras palabras, las predicciones hechas sobre un único árbol de decisión son más sensibles al ruido que pueda haber en el conjunto de entrenamiento que el promedio de varios árboles siempre que estos no estén correlacionados.

[^3]: La varianza de un modelo mide la diferencia entre el conjunto de entrenamiento y otra muestra. Es deseable que sea baja para no incurrir en overfitting.
[^4]: El sesgo mide lo lejos que se encuentra el valor estimado al real de la población.

Hay que remarcar que el número de árboles $B$ que compondrá el bosque es un parámetro libre. Es habitual que este valor oscile entre varias centenas y varios millares; no obstante la mejor forma de encontrar su valor óptimo es ir probando y aproximando.

El procedimiento anterior describe el algoritmo bootstrap aggregating; el algoritmo Random Forest va un paso más allá y para cada árbol del bosque selecciona un conjunto de características que se usarán en el entrenamiento de ese árbol de decisión excluyendo al resto. Esta selección se lleva a cabo con el fin de evitar la correlación entre árboles. Habitualmente para un problema de clasificación en el que intervienen $p$ características se seleccionan $\sqrt{p}$ características para cada árbol, mientras que si el problema es de regresión se recomienda seleccionar $p/3$. Sin embargo, una vez más el valor óptimo de este parámetro depende de cada problema y se puede optimizar probando.

**Ventajas:**
- Se considera uno de los algoritmos de clasificación más precisos para conjuntos grandes de datos.
- Es capaz de manejar un número muy grande de características y es capaz de determinar cuáles son importantes y cuáles no.

**Desventajas:**
- Se ha observado que se desajusta en conjuntos de datos con tareas de clasificación o regresión ruidosas.
- Hay que tener cuidado con el número de árboles y el número de características por árbol para obtener un modelo fiable sin incurrir en overfitting.
- Al contrario que un árbol de decisión al uso, los resultados de los bosques de decisión son complicados de interpretar.

---

## 4. Presentación de la base de datos

El propósito de esta sección es hacer una presentación detallada de la base de datos que se va a utilizar para la experimentación. Esta base de datos basada en los informes médicos de 303 pacientes del hospital de Cleaveland contiene 14 parámetros médicos que potencialmente pueden estar relacionados con una enfermedad cardiovascular. La base de datos creada por Janosi et al. [4] inicialmente tenía 75 características distintas pero todos los experimentos publicados hasta la fecha referencian solamente el uso de 14 de ellas. Los nombres originales de las características son abreviaturas en inglés de difícil entendimiento. A fin de que sean más comprensibles han sido renombradas para este trabajo.

- **Edad**

- **Sexo**
  - 0: Mujer.
  - 1: Varón.

- **Tipo de dolor pectoral:**
  - 1: Angina pectoral típica
  - 2: Angina pectoral atípica
  - 3: Dolor no producido por una angina
  - 0: Asintomático

- **Tensión en reposo:** Tensión arterial en reposo medida en mmHg tomada en el momento de ingreso en el hospital.

- **Colesterol:** Nivel de colesterol en sangre medido en mg/dL.

- **Glucemia en ayunas:** Nivel de glucosa en sangre en ayunas.
  - 0: $\leq 120$ mg/dL
  - 1: $> 120$ mg/dL

- **Electrocardiograma:** Resultados del electrocardiograma en reposo.
  - 0: Normal
  - 1: Onda ST-T anormal (Inversión de la onda T y/o elevación o depresión del segmento ST $> 0.05$ mV)[^5]
  - 2: Muestra hipertrofia del ventrículo izquierdo según el criterio de Romhilt-Estes (ver criterio de Romhilt-Estes en [18]).

[^5]: La onda T y el segmento ST son partes de un electrocardiograma (ver figura 10). La existencia de alteraciones en el patrón del electrocardiograma puede ser indicativo de algún tipo de problema cardiovascular.

> **Figura 10:** Las anomalías en el patrón de un electrocardiograma pueden indicar problemas cardiovasculares [17].

- **Ppm máximas:** Máximas pulsaciones por minuto alcanzadas.

- **Angina inducida:** Angina inducida por ejercicio. La angina es un tipo de dolor de pecho causado por la reducción del flujo de sangre al corazón.
  - 0: Ausencia de angina
  - 1: Presencia de angina

- **Depresión ST:** Depresión en el segmento ST (mm).

- **Pendiente:** Pendiente del segmento ST.
  - 1: Pendiente ascendente
  - 2: Pendiente plana
  - 3: Pendiente descendente

- **Nº vasos mayores:** Número de vasos mayores coloreados mediante fluoroscopia[^6].

[^6]: La fluoroscopia es una técnica de imagen usada en medicina para obtener imágenes en tiempo real usando rayos X de las estructuras internas de los pacientes [19].

- **Thal:** La thalassemia es una enfermedad sanguínea que se caracteriza por un decremento en la producción de hemoglobina [20].
  - 1: Normal
  - 2: Defecto corregido
  - 3: Defecto reversible

La última columna de la base de datos, llamada columna objetivo, hace referencia a la presencia (1) o ausencia (0) de una enfermedad cardiaca en el paciente. Esta última columna será la que los modelos que se entrenen intentarán predecir en función de los valores de todas las características anteriores. En el cuadro 2 se muestran a modo de ejemplo las primeras cinco filas de la base de datos.

| | edad | sexo | tipo de dolor pectoral | ... | nº vasos mayores | thal | C. objetivo |
|---|---|---|---|---|---|---|---|
| 0 | 63 | 1 | 3 | ... | 0 | 1 | 1 |
| 1 | 37 | 1 | 2 | ... | 0 | 2 | 1 |
| 2 | 41 | 0 | 1 | ... | 0 | 2 | 1 |
| 3 | 56 | 1 | 1 | ... | 0 | 2 | 1 |
| 4 | 57 | 0 | 0 | ... | 0 | 2 | 1 |

**Cuadro 2:** Primeras cinco filas de la base de datos.

Para tener una idea de los perfiles de los pacientes que están recogidos en la base de datos, en la figura 11 se puede ver los porcentajes de varones y mujeres, las edades de los pacientes y si sufrían algún tipo de enfermedad cardiaca. Se puede observar que las edades están en su mayoría comprendidas entre 38 y 71 años; además la base de datos está bien balanceada en cuanto a sexo y pacientes que sufrían una enfermedad cardiaca. Este hecho es muy importante ya que si solo hubiese pacientes varones el modelo predictivo puede que no funcione para mujeres, o si únicamente hubiese pacientes mayores de 70 años el modelo podría no funcionar para pacientes de menor edad.

> **Figura 11:** Gráficos de las características edad y sexo, y la columna objetivo de la base de datos.

Los modelos de aprendizaje automático están diseñados para ponderar positivamente aquellas características que más influyen sobre el resultado y desechar las que no. No obstante, uno se puede hacer una idea general de qué características son más influyentes elaborando un cuadro de interdependencias como el que se puede ver en la figura 12. En este gráfico, el valor en la posición $(C_i, C_j)$ representa el coeficiente de correlación de Pearson entre la columna $i$ y la columna $j$, es decir, la correlación de Pearson entre la característica $i$ y la característica $j$. El coeficiente de correlación de Pearson se define a través de la desviación típica.

> **Definición 4 (Desviación típica)** Para un conjunto de observaciones $X$ se define la desviación típica como:
>
> $$s_x = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2} \tag{22}$$
>
> donde $\bar{x}$ representa la media aritmética.

> **Definición 5 (Coeficiente de correlación de Pearson)** Sea $z_x = \dfrac{x - \bar{x}}{s_x}$ la variable $X$ normalizada, el coeficiente de correlación de Pearson se define como:
>
> $$r = \frac{\sum z_x z_y}{N-1} \tag{24}$$

El sentido de que el coeficiente de correlación de Pearson utilice las variables normalizadas de los conjuntos de datos $X$ e $Y$, asociados a dos características de la base de datos, es poder comparar las distribuciones normales de ambas variables para calcular su correlación. Cabe hacer algunas puntualizaciones sobre el coeficiente de correlación de Pearson:

- El coeficiente de Pearson toma valores entre $-1$ y $1$, donde $1$ indica correlación directa y $-1$ correlación indirecta.
- Un valor muy alto del coeficiente de Pearson no indica causalidad entre las dos variables. Por ejemplo, si tomamos en un instituto las variables altura y nivel de inglés probablemente exista una alta correlación entre ambas y sin embargo no existe ningún tipo de causalidad entre ellas. Lo que sucede es que los estudiantes más altos son de mayor edad y han estudiado más cursos de inglés. Una correlación solo es un valor matemático carente de interpretación [21].

> **Figura 12:** Gráfico de correlación de Pearson entre características de la base de datos.

En la figura 12 se puede observar que algunas de las características que más correlación tienen con una enfermedad cardiaca son el tipo de dolor pectoral, la depresión en el segmento ST o las pulsaciones por minuto máximas. Esto significa que estas características serán las que más peso tendrán en las predicciones de los modelos.

---

## 5. Métricas

El propósito de esta sección es presentar las métricas que se van a utilizar para evaluar qué tan buen rendimiento ofrecen los modelos sobre la base de datos. Recordemos que en este trabajo métrica se va a utilizar como sinónimo de indicador de la bondad de un modelo y en ningún caso tendrá que ver con el concepto matemático de distancia. Existen numerosas métricas para evaluar modelos de clasificación y de regresión; no obstante, para este trabajo nos centraremos en dos de las más utilizadas y reconocidas:

- AUC
- Valor F1

Estas métricas se construyen a partir de los conceptos precisión y exhaustividad, que a su vez se construyen empleando una serie de indicadores que contabilizan si las predicciones de un clasificador binario, es decir, positivo o negativo, se han realizado correctamente:

- **VP (Verdaderos positivos):** Casos en los que la predicción fue positiva y acertó.
- **VN (Verdaderos negativos):** Casos en los que la predicción fue negativa y acertó.
- **FP (Falsos positivos):** Casos en los que la predicción fue positiva y falló.
- **FN (Falsos negativos):** Casos en los que la predicción fue negativa y falló.

| | Positivo | Negativo |
|---|---|---|
| **Predicción positiva** | VP | FP (Error de tipo I) |
| **Predicción negativa** | FN (Error de tipo II) | VN |

Estas cuatro métricas sirven para calcular los ratios de precisión, exhaustividad, RVP (Ratio de Verdaderos Positivos) y RVN (Ratio de Verdaderos Negativos):

- **Precisión o RVP:** El ratio de precisión indica cuántos positivos del total de positivos predichos han sido correctos. La precisión sirve para entender cómo de acertado es un modelo cuando predice un positivo. Una precisión cercana a 1 indica que el modelo acierta casi siempre que predice un positivo.

$$\text{precisión} = \frac{VP}{VP + FP} \tag{25}$$

- **Exhaustividad:** El ratio de exhaustividad, también llamado sensibilidad, es el cociente entre los positivos bien predichos y el total de positivos. Una exhaustividad cercana a 1 indica que el modelo acierta en casi todos los casos en los que la predicción correcta es positivo.

$$\text{exhaustividad} = \frac{VP}{VP + FN} \tag{26}$$

- **RFP:** El ratio de falsos positivos se define como:

$$RFP = \frac{FP}{FP + VN} \tag{27}$$

Precisión o RVP, exhaustividad y RFP no son métricas independientes entre sí; existe un equilibrio entre ellas. Como se puede ver en la figura 13, si el clasificador aumenta la precisión, disminuirá la exhaustividad y viceversa. Esto se debe a que si el clasificador es muy preciso, es decir, solo predice positivo cuando está extremadamente seguro de que se trata de un positivo, entonces se predecirá negativo para muchas instancias que tenían que haber sido predichas positivo (FN). Si por el contrario el clasificador siempre predice positivo, la exhaustividad será 1 porque nunca fallará una predicción positiva pero su tasa de aciertos será muy baja (0.5 si hay igual de positivos reales que negativos reales).

> **Figura 13:** Equilibrio precisión-exhaustividad en función del umbral de decisión [34].

La curva precisión-exhaustividad (P-E) es el resultado de contraponer en una gráfica cómo varía la precisión en función de la exhaustividad. Esta gráfica permite ver a partir de qué exhaustividad la precisión se degrada. Si un modelo es bueno la curva precisión-exhaustividad se acercará mucho a la esquina superior derecha, es decir, aún teniendo una sensibilidad alta las predicciones son precisas.

Por otra parte, una de las curvas más empleadas para evaluar modelos es la curva RVP-RFP o curva ROC (Receiver Operating Characteristic). Esta curva dibuja cómo varía RVP frente a RFP cuando varía el umbral de discriminación (ver figura 14), es decir, el valor a partir del cual se clasifica como positiva una instancia.

> **Figura 14:** La curva ROC representa TPR frente a FPR [36].

AUC (Area Under the Curve) es la integral de la curva ROC entre 0 y 1, es decir, proporciona una medición agregada del rendimiento en todos los umbrales de clasificación posibles:

$$AUC = \int_0^1 C_{ROC} \tag{28}$$

AUC tiene dos características que la convierten en una métrica consistente:

- Es invariable respecto a la escala, es decir, mide qué tan bien se clasifican las predicciones en lugar de fijarse en los valores absolutos.
- Es invariable respecto al umbral de clasificación, esto es, mide la calidad de las predicciones sin tener en cuenta el umbral de clasificación elegido.

Por otra parte el valor F1 se define como la media armónica entre precisión y exhaustividad:

$$F_1 = 2 \cdot \frac{\text{precisión} \cdot \text{exhaustividad}}{\text{precisión} + \text{exhaustividad}} \tag{29}$$

El valor F asume que precisión y exhaustividad tienen la misma importancia pero como se vio en el cuadro 1 no siempre es igual de importante evitar errores de tipo I que evitar errores de tipo II. En caso de que no se quiera valorar equitativamente precisión y exhaustividad se emplea el valor $F_\beta$ donde $\beta$ es un factor que indica que se considera $\beta$ veces más importante la exhaustividad que la precisión:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precisión} \cdot \text{exhaustividad}}{(\beta^2 \cdot \text{precisión}) + \text{exhaustividad}} \tag{30}$$

En el caso que nos atañe se considerará igual de importante que el modelo sea preciso prediciendo positivos y que el modelo acierte cuando predice un positivo, por lo que se usará el valor F1 [35], [36], [37], [38].

---

## 6. Metodología

En esta sección se describe el flujo de trabajo desde una base de datos en "crudo" hasta la evaluación final atendiendo a las métricas expuestas en la anterior sección. El procedimiento básico es el siguiente:

```
Base de datos → Partición → Conjunto de entrenamiento → Entrenamiento → Modelo entrenado → Predicción → Evaluación
                          → Conjunto de prueba ↗
```

El primer paso es dividir la base de datos en dos partes:

- **Conjunto de entrenamiento:** Subconjunto utilizado para entrenar el modelo.
- **Conjunto de prueba:** Subconjunto utilizado para evaluar el modelo.

La base de datos $(X|y)$ queda pues dividida de la siguiente manera:

$$(X|y) = \begin{pmatrix} X_{\text{entr}} & y_{\text{entr}} \\ X_{\text{prueba}} & y_{\text{prueba}} \end{pmatrix} \tag{31}$$

El conjunto de entrenamiento $(X_{\text{entr}}|y_{\text{entr}})$ se utiliza para entrenar el modelo. Tras el entrenamiento se consigue un modelo entrenado capaz de emitir predicciones. El modelo predice las clases para el conjunto $X_{\text{prueba}}$; estas predicciones forman un vector $y_{\text{pred}}$ que se llamará vector de predicciones. El vector $y_{\text{pred}}$ es entonces comparado con $y_{\text{prueba}}$, que son las clases reales del conjunto de validación. En esta comparación se aplican las métricas expuestas en la sección 5 para completar la evaluación de los resultados.

No obstante este procedimiento no es suficiente para asegurar que el modelo no está sesgado, especialmente cuando se varían los parámetros del modelo. Esto se debe a que los parámetros se pueden ajustar hasta obtener métricas muy buenas para un conjunto de prueba concreto pero eso no quiere decir que el modelo sea óptimo ya que si se hiciese una nueva separación y se tuviese un conjunto de prueba distinto los resultados podrían ser malos por haber ajustado los parámetros a los datos concretos del primer conjunto de prueba. A este fenómeno se le denomina overfitting.

Nos encontramos por tanto antes dos problemas diferenciados:

- La dependencia entre los resultados y la partición entrenamiento-prueba.
- La dependencia entre la partición entrenamiento-prueba y los parámetros que optimizan el modelo. Este problema solo se presenta en los modelos que tienen parámetros como k-NN, pero no en Naive Bayes.

Para solucionar la dependencia entre los resultados y la partición entrenamiento-prueba se hará uso de un método llamado k-fold crossvalidation o validación cruzada de k iteraciones. La validación cruzada de k iteraciones consiste en repetir k veces la partición entre conjunto de entrenamiento y conjunto de prueba iterando el conjunto de prueba. De esta manera, para cada partición, se obtendrán valores distintos para las métricas. La media aritmética de las métricas será el valor tomado por válido.

> **Figura 15:** Esquema de una validación cruzada de 4 iteraciones [39].

En los modelos con parámetros, para solucionar el overfitting de los parámetros a los conjuntos de prueba se va a usar una tercera división en la base de datos completa. De esta manera se tiene un conjunto de entrenamiento en el que probar parámetros sobre el que se aplicará validación cruzada y un conjunto insesgado en el que poner a prueba el modelo final.

> **Figura 16:** Esquema de una validación cruzada de 5 iteraciones reservando algunos datos para el testeo final [40].

Para los modelos sin parámetros en los que se usará únicamente validación cruzada se realizarán 5 iteraciones, es decir, $k = 5$. Para los modelos con parámetros se usará validación cruzada y una tercera partición para evitar el sesgo de los parámetros; en este caso se reservará el 10% de los datos para el test final y se realizarán 4 iteraciones en la validación cruzada.

Siguiendo estos pasos uno se asegura de que la evaluación será insesgada, pero queda una cuestión por dirimir. La mayoría de los modelos tienen hiperparámetros, que es como se denomina a los distintos parámetros que pueden variar dentro de un modelo. Por ejemplo, KNN tiene como hiperparámetros el valor de k y la distancia que se va a usar. El resultado de tomar los 3 vecinos más cercanos según la distancia euclídea no es el mismo que el de tomar los 7 vecinos más cercanos con la distancia Manhattan.

Para conseguir los parámetros óptimos se hará uso de grid search o búsqueda en cuadrícula. Este método consiste en probar tantas combinaciones de parámetros como uno quiera y quedarse con la que mejor resultados ofrezca. La búsqueda en cuadrícula se combina con la validación cruzada sobre el conjunto de entrenamiento, y una vez se han encontrado los parámetros óptimos se evalúa el modelo sobre el conjunto de prueba.

Este proceso para encontrar los parámetros óptimos es eficaz y simple pero requiere de un gran coste computacional. Por ejemplo si se va a optimizar un modelo con 3 parámetros, cada uno de los cuales puede tomar 3 valores y se va a usar validación cruzada de 4 iteraciones, en total habría que realizar $3 \cdot 3 \cdot 3 \cdot 4 = 36$ iteraciones.

---

## 7. Marcos de trabajo

En esta sección se especifican los recursos que han sido utilizados durante la experimentación.

- **Base de datos:** Ha sido descargada desde Kaggle, una plataforma que pone a disposición de sus usuarios numerosas bases de datos y problemas relacionados con el data science, el análisis predictivo y el machine learning. La fuente original de la base de datos que se va a utilizar es el repositorio de machine learning UCI (ver referencia [41]).

- **Reentrenamiento y evaluación:** Para el entrenamiento y evaluación se ha utilizado Python 3.7. En concreto se han utilizado los siguientes scripts de elaboración propia que se pueden encontrar en mi repositorio de Github (https://github.com/sgavela/heart_disease_prediction):
  - `visualization.py`: Script empleado para elaborar algunos gráficos que ofrezcan una visión general de la base de datos.
  - `nb.py`: Script empleado para entrenar el clasificador Naive Bayes y evaluar sus resultados.
  - `knn.py`: Script empleado para entrenar y optimizar el clasificador KNN y evaluar sus resultados.
  - `lin_reg.py`: Script empleado para entrenar y optimizar el clasificador regresión linear y evaluar sus resultados.
  - `log_reg.py`: Script empleado para entrenar y optimizar el clasificador regresión logística y evaluar sus resultados.
  - `rf.py`: Script empleado para entrenar y optimizar el clasificador Random Forest y evaluar sus resultados.

  Todos los scripts hacen uso de librerías de Python populares en problemas de machine learning como pandas, matplotlib, numpy y sklearn.

- **Desarrollo de la interfaz gráfica:** Se ha utilizado también Python para desarrollar la interfaz gráfica. Además de las librerías anteriormente mencionadas también se ha usado PySimpleGUI, una librería de Python enfocada al desarrollo de interfaces gráficas.

---

## 8. Resultados

El propósito de esta sección es presentar los resultados que ha conseguido cada modelo de clasificación atendiendo a las métricas expuestas en la sección 5. Para cada clasificador habrá una pequeña subsección en la que se indiquen los resultados y los experimentos hechos con el clasificador tales como variar los parámetros o los umbrales de decisión. Esta sección concluirá con una comparativa entre los resultados.

### 8.1. Naive Bayes

El clasificador Naive Bayes se basaba en el teorema de Naive Bayes para calcular la probabilidad de cada clase asumiendo que cada característica era independiente del resto. Dado que es un clasificador que no tiene parámetros se usará únicamente validación cruzada de 5 iteraciones, es decir, se dividirá la base de datos en cinco partes y en cada iteración una de las partes hará las veces de conjunto de prueba y las otras cuatro de conjunto de entrenamiento. Para calcular la precisión, la exhaustividad, el valor F1 y el valor AUC se hará la media para estos valores en cada una de las cinco iteraciones.

Para las cinco iteraciones se ha considerado que si el clasificador asignaba a una clase una probabilidad superior a 0.5 entonces esa era la clase predicha. En la siguiente tabla se presentan los resultados obtenidos para la precisión, la exhaustividad y el valor F1 que depende de ambos.

| Naive Bayes | |
|---|---|
| Precisión | 0,8157 |
| Exhaustividad | 0,8364 |
| Valor F1 | 0,8246 |

**Cuadro 3:** Precisión, exhaustividad y valor F1 para Naive Bayes.

En cuanto al valor de AUC, como se vio en la sección 5 se calcula a partir de la curva ROC. No obstante, esta curva depende de la partición entrenamiento-prueba que se haya hecho. Por ejemplo en la figura 17 se puede ver la curva ROC para una partición aleatoria.

> **Figura 17:** Curva ROC de Naive Bayes para una partición aleatoria (AUC = 0.8448).

En la leyenda al pie del diagrama se puede ver que en este caso $\text{AUC} = 0.8448$, pero este valor dependerá de la curva. Al igual que con precisión, exhaustividad y valor F1, para cada iteración en la validación cruzada se calculará la curva ROC, el valor de AUC y se hará el promedio. Los resultados finales se pueden ver en la siguiente tabla.

| Naive Bayes | |
|---|---|
| Precisión | 0,8157 |
| Exhaustividad | 0,8364 |
| Valor F1 | 0,8246 |
| AUC | 0,8918 |

**Cuadro 4:** Precisión, exhaustividad, valor F1 y AUC para Naive Bayes.

### 8.2. KNN

El algoritmo KNN o K-Nearest Neighbours calculaba la probabilidad de que un elemento perteneciese a una clase en base a las clases de sus k vecinos más cercanos. En el caso de KNN hay tres parámetros que pueden variar:

- **k:** Se refiere al número de vecinos más cercanos que se tienen en cuenta. En este caso se tomarán como valores de k: 3, 5, 11 y 19.
- **weights:** Hace referencia a si se va a ponderar la distancia de cada uno de los k vecinos más cercanos.
  - `uniform`: Todos los elementos en el vecindario tienen igual peso en la predicción.
  - `distance`: Se pondera a cada vecino inversamente a la distancia al elemento a predecir, es decir, cuanto más cercano sea más relevante será en la predicción.
- **p:** Se utiliza para determinar qué tipo de distancia se va a usar para calcular los vecinos más cercanos. La distancia Minkowski entre dos puntos se define como:

$$d(Q, S) = \left(\sum_{i=1}^{n}|s_i - q_i|^p\right)^{1/p} \tag{32}$$

  Para el parámetro p se probarán los siguientes valores:
  - $p=1$: Distancia Manhattan.
  - $p=2$: Distancia euclídea. [44]

Para determinar cuál se va a considerar la combinación óptima de parámetros nos fijaremos en el valor F1. Tras entrenar el modelo atendiendo a las diferentes combinaciones de parámetros y utilizando validación cruzada de 4 iteraciones, los resultados obtenidos son:

| Parámetros (k, p, weights) | Valor F1 |
|---|---|
| 3, 1, uniform | 0,6792 |
| 3, 1, distance | 0,6792 |
| 3, 2, uniform | 0,6492 |
| 3, 2, distance | 0,6470 |
| 5, 1, uniform | 0,7012 |
| 5, 1, distance | 0,6983 |
| 5, 2, uniform | 0,6588 |
| 5, 2, distance | 0,6540 |
| 11, 1, uniform | 0,6724 |
| 11, 1, distance | 0,6900 |
| 11, 2, uniform | 0,6326 |
| 11, 2, distance | 0,6330 |
| 19, 1, uniform | 0,7002 |
| 19, 1, distance | 0,6936 |
| 19, 2, uniform | 0,6765 |
| 19, 2, distance | 0,6517 |

**Cuadro 5:** Valores F1 para las posibles combinaciones de parámetros de KNN.

Si se analiza el cuadro se puede ver que el mejor valor F1 se obtiene para $k = 5$, $p = 1$ y `weight = uniform`.

Antes de realizar el entrenamiento y la optimización de parámetros se ha reservado aleatoriamente un 20% la base de datos para la prueba final. Con los parámetros obtenidos en la optimización se pueden evaluar los resultados del modelo KNN sobre el conjunto de prueba:

| KNN | |
|---|---|
| Precisión | 0,7273 |
| Exhaustividad | 0,6885 |
| Valor F1 | 0,7074 |
| AUC | 0,6863 |

**Cuadro 6:** Precisión, exhaustividad, valor F1 y AUC para KNN.

Los resultados empeoran sensiblemente respecto a los conseguidos con Naive Bayes aunque siguen siendo aceptables. En la figura 18 se puede ver la curva ROC de KNN para el conjunto de prueba.

> **Figura 18:** Curva ROC del algoritmo KNN para el conjunto de prueba.

### 8.3. Regresión linear

La regresión linear, al contrario que Naive Bayes y KNN, no es un clasificador sino un modelo de regresión. Esto obliga a cambiar ligeramente la perspectiva ya que la predicción que emite el modelo no es una probabilidad sino el valor de la función de regresión, lo que obliga a fijar un umbral para distinguir qué clase está prediciendo el modelo. De esta manera se convierte un modelo de regresión en un modelo de clasificación.

La regresión linear hace uso de los siguientes parámetros:

- **fit_intercept:** Este parámetro es booleano. Si es `False` se está forzando a la recta de regresión a pasar por el origen $(0, 0)$; si es `True` la recta de regresión puede intersecar al eje $y$ en un punto que no sea el origen.
- **normalize:** Este parámetro booleano es ignorado cuando `fit_intercept` toma el valor `False`. En caso de que ambos sean `True`, los regresores $X$ serán normalizados antes de la regresión, es decir, se les restará la media y serán divididos entre la desviación estándar:

$$X_b = \frac{X - \mu}{\sigma} \tag{33}$$

Este proceso, a veces, permite eliminar los efectos de datos anormales [45].

Como métrica para la optimización de parámetros de la regresión linear se usa el error cuadrático medio (MSE). Los resultados finales son:

| | |
|---|---|
| fit_intercept | True |
| normalize | False |
| MSE | 0,1371 |

**Cuadro 7:** Parámetros óptimos para la regresión linear en nuestra base de datos.

Para transformar el modelo de regresión en un modelo de clasificación se usa un umbral de 0.5: cualquier valor en la recta de regresión por debajo de 0.5 será considerado clase 0 (sano) y cualquier valor por encima clase 1 (enfermedad cardíaca).

| Regresión linear | |
|---|---|
| Precisión | 0,8158 |
| Exhaustividad | 0,8361 |
| Valor F1 | 0,8258 |
| AUC | 0,9041 |

**Cuadro 8:** Precisión, exhaustividad, valor F1 y AUC para el modelo de regresión linear.

> **Figura 20:** Curva ROC del modelo de regresión linear para el conjunto de prueba.

### 8.4. Regresión logística

La regresión logística modelaba la probabilidad de una clase utilizando una función sigmoide:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \tag{34}$$

En el caso de la regresión logística hace uso de los siguientes parámetros [43], [46]:

- **penalty:** Se usa para especificar la norma usada en la regularización. Puede tomar los valores `l1`, `l2`, `elasticnet` y `none`.
- **C:** Indica la inversa de la fuerza de regularización; cuanto menor sea el valor de C más fuerte será la regularización.
- **solver:** Algoritmo usado para solucionar el problema de optimización:
  - `newton-cg`: Utiliza el método de Newton estimando la matriz Hessiana.
  - `lbfgs`: Limited-memory Broyden-Fletcher-Goldfarb-Shanno Algorithm.
  - `liblinear`: Library for Large Linear Classification. Utiliza un algoritmo de coordenadas descendentes.
  - `sag`: Stochastic Average Gradient.
  - `saga`: Variante del método SAG.

Tras la búsqueda en cuadrícula, la combinación óptima obtenida es:

| | |
|---|---|
| penalty | l2 |
| C | 0,6158 |
| solver | liblinear |
| MSE | 0,1693 |

**Cuadro 9:** Parámetros óptimos para la regresión logística.

| Regresión logística | |
|---|---|
| Precisión | 0,8525 |
| Exhaustividad | 0,8571 |
| Valor F1 | 0,8548 |
| AUC | 0,9096 |

**Cuadro 10:** Precisión, exhaustividad, valor F1 y AUC para el modelo de regresión logística.

### 8.5. Random Forest

El último modelo que se va a evaluar es Random Forest, que se basa en varios árboles de decisión cuyos resultados se ponen en común para obtener el resultado final (ver sección 3.5).

Para el clasificador Random Forest se han probado combinaciones con los siguientes parámetros [47]:

- **bootstrap:** Parámetro booleano que indica si se va a usar bootstrap aggregating.
- **criterion:** Función destinada a medir la calidad de cada partición en ramas del árbol.
  - `gini`: Utiliza el índice de Gini para cada nodo:

    $$\text{gini} = 1 - \sum_j p_j^2 \tag{35}$$

  - `entropy`: La entropía se calcula para cada nodo con la siguiente fórmula:

    $$\text{entropy} = -\sum_j p_j \log_2(p_j) \tag{36}$$

- **max_depth:** Máxima profundidad que podrá alcanzar el árbol. Valores probados: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 y `None`.
- **max_features:** Número de características que se tendrán en cuenta para decidir cuál es la mejor partición en cada nodo.
  - `auto`: $N$.
  - `sqrt`: $\sqrt{N}$.
  - `log2`: $\log_2(N)$.
- **min_samples_leaf:** Número mínimo de elementos que deberá haber en cada hoja del árbol. Valores probados: 1, 2, 4, 8, 16 y 32.
- **min_samples_split:** Número mínimo de elementos que deberá haber en un nodo para que este pueda ser particionado. Valores probados: 2, 5, 10, 15 y 20.
- **n_estimators:** Número de árboles en el bosque. Valores probados: 50, 100, 200 y 400.

Tras la búsqueda en cuadrícula estos son los parámetros óptimos obtenidos:

| | |
|---|---|
| bootstrap | True |
| criterion | gini |
| max_depth | None |
| max_features | log2 |
| min_samples_leaf | 16 |
| min_samples_split | 10 |
| n_estimators | 100 |
| Valor F1 | 0,8633 |

**Cuadro 11:** Parámetros óptimos para Random Forest.

| Random Forest | |
|---|---|
| Precisión | 0,8611 |
| Exhaustividad | 0,8689 |
| Valor F1 | 0,8650 |
| AUC | 0,8633 |

**Cuadro 12:** Precisión, exhaustividad, valor F1 y AUC para el modelo de Random Forest.

### 8.6. Conclusiones de la evaluación de modelos

Para seleccionar cuál es el mejor modelo hay que determinar qué criterio se va a usar para decir que un modelo es mejor que otro. En este caso se va a usar el promedio entre el valor F1 y AUC.

| | Naive Bayes | KNN | Reg. linear | Reg. logística | Random forest |
|---|---|---|---|---|---|
| Precisión | 0,8157 | 0,7273 | 0,8158 | 0,8525 | 0,8611 |
| Exhaustividad | 0,8364 | 0,6885 | 0,8361 | 0,8571 | 0,8689 |
| Valor F1 | 0,8246 | 0,7074 | 0,8258 | 0,8548 | 0,8650 |
| AUC | 0,8918 | 0,6863 | 0,9041 | 0,9096 | 0,8633 |
| **Promedio F1-AUC** | **0,8582** | **0,6969** | **0,8650** | **0,8822** | **0,8641** |

**Cuadro 13:** Comparación global de los resultados obtenidos por cada modelo.

En el cuadro se puede observar cómo a excepción de KNN, el resto de modelos ha ofrecido un rendimiento muy bueno y bastante similar. El mejor modelo atendiendo al promedio entre valor F1 y AUC es la **regresión logística**, seguida de la regresión linear y random forest. El mejor modelo, es decir, la regresión logística, será integrado en la aplicación que se va a diseñar.

---

## 9. Manual de la interfaz gráfica

### 9.1. Introducción

La calculadora de riesgo de enfermedad cardiovascular es una aplicación muy simple a la que podrían incorporarse nuevas funciones. La única función de la que dispone en este momento es pedir al usuario los datos médicos necesarios para emitir una predicción y devuelve la predicción hecha por el modelo.

La calculadora utiliza un modelo de regresión logística para, a partir de 14 parámetros médicos, predecir la probabilidad de padecer una enfermedad cardiovascular. La calculadora está diseñada para ser usada por personal sanitario dado que la mayoría de los parámetros requieren de material y conocimientos médicos para ser medidos.

### 9.2. Cómo descargar la carpeta

La aplicación puede descargarse desde el siguiente repositorio de Github:

https://github.com/sgavela/heart_disease_prediction_calculator

Para descargarla la opción más sencilla es clicar en el botón verde **Code** arriba a la derecha del repositorio y seleccionar **Descargar ZIP** en el menú desplegable. Finalmente descomprimir el archivo ZIP en el ordenador.

### 9.3. Contenido de la carpeta

En la carpeta pueden encontrarse los siguientes elementos:

- `manual`: Manual de uso en formato PDF.
- `requirements`: Script para descargar las bibliotecas de Python necesarias para ejecutar la interfaz gráfica.
- `init`: Script para ejecutar la interfaz gráfica.
- Carpeta `data`: En esta carpeta se puede encontrar la base de datos en formato CSV.
- Carpeta `utils`: En esta carpeta están los scripts de Python que se usan para desplegar la interfaz gráfica, elaborar la predicción y gestionar los posibles errores.

### 9.4. Antes de ejecutar la aplicación

Para ejecutar con éxito la aplicación para predicción de enfermedades se deben de cumplir los siguientes requerimientos:

- Ordenador con sistema operativo Windows (puede que también funcione en Linux o Mac pero no se puede asegurar).
- Tener instalada y agregada al path alguna distribución de Python. Se asegura el funcionamiento de la aplicación para las versiones 3.7 en adelante.
- Ejecutar el script `requirements`. Para ejecutarlo hacer doble click sobre el script (en la figura 21 se puede ver la ventana que se debería de abrir si el script ha funcionado correctamente. Esta ventana se cierra automáticamente).

> **Figura 21:** Correcto funcionamiento del script requirements.

### 9.5. Ejecutar la aplicación

Ejecutar la aplicación y abrir la interfaz gráfica es tan sencillo como ejecutar el script `init` (para ejecutarlo hacer doble click sobre el script).

Es muy importante para cerrar la aplicación pulsar el botón **Cerrar** y no la X arriba a la derecha de la ventana ya que esto producirá un error. Este error se debe a un fallo en la biblioteca de Python que se usa para desplegar la interfaz gráfica.

### 9.6. Introducción de datos

Se debe prestar atención a que los datos estén correctamente introducidos. Para ello deben de ajustarse a las siguientes normas:

- **Edad:** Medida en años.
- **Sexo:** 0 = Mujer, 1 = Varón.
- **Tipo de dolor pectoral:** 1 = Angina pectoral típica, 2 = Angina pectoral atípica, 3 = Dolor no producido por una angina, 0 = Asintomático.
- **Tensión en reposo:** Tensión arterial en reposo medida en mmHg.
- **Colesterol:** Nivel de colesterol en sangre medido en mg/dL.
- **Glucemia en ayunas:** 0 = $\leq 120$ mg/dL, 1 = $> 120$ mg/dL.
- **Electrocardiograma:** 0 = Normal, 1 = Onda ST-T anormal, 2 = Hipertrofia del ventrículo izquierdo.
- **Ppm máximas:** Máximas pulsaciones por minuto alcanzadas.
- **Angina inducida:** 0 = Ausencia de angina, 1 = Presencia de angina.
- **Depresión ST:** Depresión en el segmento ST (mm).
- **Pendiente:** 1 = Pendiente ascendente, 2 = Pendiente plana, 3 = Pendiente descendente.
- **Nº vasos mayores:** Número de vasos mayores coloreados mediante fluoroscopia.
- **Thal:** 1 = Normal, 2 = Defecto corregido, 3 = Defecto reversible.

Los siguientes escenarios producirán un mensaje de error:

- No se ha introducido un valor para alguno de los parámetros.
- Se ha introducido un valor no entero para un parámetro (incluidas cadenas de caracteres).
- Se ha introducido un valor no permitido para un parámetro (por ejemplo 4 para tipo de dolor pectoral o $-27$ para las ppm máximas).

> **Figura 22:** Ejemplo de error en la interfaz gráfica.

> **Figura 23:** Ejemplo de funcionamiento correcto de la interfaz gráfica (Probabilidad de enfermedad: 0.82 — Muy probable).

---

## 10. Conclusiones

A modo de conclusión se va a hacer una revisión de los objetivos que fueron establecidos al principio de este trabajo. Para cada objetivo se expondrá hasta qué punto se ha cumplido y cuáles han sido los resultados y conclusiones obtenidos.

**Identificar la correlación entre los parámetros.**
Este primer objetivo se ha cumplimentado en la sección 4. Para estudiar la correlación se ha empleado el coeficiente de Pearson. Los resultados han sido que los parámetros que más correlación tienen con la presencia de una enfermedad cardíaca son:
- El tipo de dolor pectoral.
- La presencia de una angina inducida.
- La depresión en el segmento ST del electrocardiograma.
- Las PPM máximas alcanzadas.

**Elaborar varios modelos predictivos.**
Se han elaborado con éxito cinco modelos predictivos que utilizaban la base de datos de A. Janosi et al.: Naive Bayes, KNN, regresión linear, regresión logística y Random forest. Para cada uno de estos modelos se ha asegurado la ausencia de sesgo utilizando validación cruzada y particiones entrenamiento-prueba (ver sección 6). Además, la mayoría de estos modelos constan de parámetros cuyos valores se pueden variar; para obtener los parámetros que ofrecían mejores resultados se ha empleado una técnica llamada búsqueda en cuadrícula.

**Determinar qué modelo ofrece mejores resultados.**
Se han utilizado dos métricas para evaluar los modelos (ver sección 5): Valor F1 y AUC. El criterio final para determinar qué modelo es el mejor ha sido el promedio de estas dos métricas.

| | Naive Bayes | KNN | Reg. linear | Reg. logística | Random forest |
|---|---|---|---|---|---|
| Precisión | 0,8157 | 0,7273 | 0,8158 | 0,8525 | 0,8611 |
| Exhaustividad | 0,8364 | 0,6885 | 0,8361 | 0,8571 | 0,8689 |
| Valor F1 | 0,8246 | 0,7074 | 0,8258 | 0,8548 | 0,8650 |
| AUC | 0,8918 | 0,6863 | 0,9041 | 0,9096 | 0,8633 |
| **Promedio F1-AUC** | **0,8582** | **0,6969** | **0,8650** | **0,8822** | **0,8641** |

**Cuadro 14:** Comparación global de los resultados obtenidos por cada modelo.

De los resultados se puede concluir que **la regresión logística ha sido el mejor modelo**.

**Crear una interfaz gráfica.**
La interfaz gráfica se ha programado en Python y tiene forma de calculadora. El usuario introduce los datos y la interfaz imprime por pantalla el resultado. Se ha diseñado de manera que su uso sea muy intuitivo e implementa un control de errores básicos (introducción de parámetros inválidos, parámetros incompletos, etc.). La sección 9 es un detallado manual de uso de esta interfaz con ejemplos de funcionamiento incluidos.

> **Figura 24:** Ejemplo de funcionamiento correcto de la interfaz gráfica.

---

## Referencias

[1] Organización Mundial de la Salud, *¿Qué son las enfermedades cardiovasculares?*, https://www.who.int/cardiovascular_diseases/about_cvd/es/

[2] Organización Mundial de la Salud, *The global strategy on diet, physical activity and health*, https://www.who.int/dietphysicalactivity/media/en/gsfs_general.pdf

[3] Instituto de Salud Pública de San Sebastián, *Las 10 principales causas de muerte en el mundo*, http://www.ipsuss.cl/ipsuss/actualidad/las-10-principales-causas-de-muerte-en-el-mundo/2019-06-02/230253.html

[4] Andras Janosi, William Steinbrunn, Matthias Pfisterer et al., *International application of a new probability algorithm for the diagnosis of coronary artery disease*, https://pubmed.ncbi.nlm.nih.gov/2756873/

[5] Microsoft, *Definición de un modelo de aprendizaje automático*, https://docs.microsoft.com/es-es/windows/ai/windows-ml/what-is-a-machine-learning-model, 2019.

[6] Manuel Zaforas, *Machine Learning para dummies*, https://www.paradigmadigital.com/techbiz/machine-learning-dummies, 2017.

[7] Paloma Recuero de los Santos, *Tipos de aprendizaje en Machine Learning: supervisado y no supervisado*, https://empresas.blogthinkbig.com/que-algoritmo-elegir-en-ml-aprendizaje/, 2017.

[8] Epicalsoft, *[Azure Machine Learning] Tipos de problemas en Machine Learning*, http://epicalsoft.blogspot.com/2018/11/azure-machine-learning-algoritmos-de.html

[9] Victor Roman, *Aprendizaje Supervisado: Introducción a la Clasificación y Principales Algoritmos*, https://medium.com/datos-y-ciencia/aprendizaje-supervisado-introduccion-a-la-clasificacion-y-principales-algoritmos-dadee99c9407, 2019.

[10] Macarena Estevez, *Un acercamiento a los modelos de clasificación*, https://inteligenciaanalitica.com/acercamiento-modelos-clasificacion/, 2016.

[11] Wikipedia, *Teorema de Bayes*, https://es.wikipedia.org/wiki/Teorema_de_Bayes

[12] Wikipedia, *Clasificador bayesiano ingenuo*, https://es.wikipedia.org/wiki/Clasificador_bayesiano_ingenuo

[13] Programador Clic, *Algoritmo ingenuo de Bayes con ventajas y desventajas*, https://programmerclick.com/article/9259854603/

[14] Cristina García e Irene Gómez, *Algoritmos de aprendizaje: KNN y KMEANS*, http://www.it.uc3m.es/jvillena/irc/practicas/08-09/06.pdf, 2006.

[15] Wikipedia, *k vecinos más próximos*, https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos

[16] Dhilip Subramanian, *A Simple Introduction to K-Nearest Neighbors Algorithm*, https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e, 2019.

[17] Wikipedia, *ST segment*, https://en.wikipedia.org/wiki/ST_segment

[18] D W Romhilt et al., *A critical appraisal of the electrocardiographic criteria for the diagnosis of left ventricular hypertrophy*, https://pubmed.ncbi.nlm.nih.gov/4240354/, 1969.

[19] Wikipedia, *Fluoroscopia*, https://es.wikipedia.org/wiki/Fluoroscopia

[20] Wikipedia, *Thalassemia*, https://en.wikipedia.org/wiki/Thalassemia

[21] Aleksander Dietrichson, *Métodos Cuantitativos*, https://bookdown.org/dietrichson/metodos-cuantitativos/, 2019.

[22] José Francisco López, *Homocedasticidad*, https://economipedia.com/definiciones/homocedasticidad.html

[23] Pritish Jadhav, *Proving Convexity of the MSE Loss Function*, https://medium.com/swlh/proving-convexity-of-mean-squared-error-loss-function-68b139fe6ea7, 2017.

[24] T. L. Lai, Herbert Robbins y C. Z. Wei, *Strong consistency of least squares estimates in multiple regression*, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC392707/pdf/pnas00019-0030.pdf, 1978.

[25] Wikipedia, *Linear regression*, https://en.wikipedia.org/wiki/Linear_regression

[26] Wikipedia, *Linear least squares*, https://en.wikipedia.org/wiki/Linear_least_squares

[27] Ciiia, *Regresión Logística*, https://youtu.be/SeM4Rtoa4EU, 2020.

[28] José Martínez Heras, *Gradiente Descendiente para aprendizaje automático*, https://www.iartificial.net/gradiente-descendiente-para-aprendizaje-automatico/, 2020.

[29] SitioBigData, *Árbol de decisión en Machine Learning (Parte 1)*, https://sitiobigdata.com/2019/12/14/arbol-de-decision-en-machine-learning-parte-1/

[30] Wikipedia, *Random forest*, https://en.wikipedia.org/wiki/Random_forest

[31] Wikipedia, *Bootstrap aggregating*, https://en.wikipedia.org/wiki/Bootstrap_aggregating

[32] Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome, *The Elements of Statistical Learning* (2nd ed.), Springer, 2008.

[33] Ho, Tin Kam, *Random Decision Forests*, 1995.

[34] J. Ramírez, *Curvas PR y ROC*, https://medium.com/bluekiri/curvas-pr-y-roc-1489fbd9a527, 2018.

[35] N. S. Chauan, *Métricas De Evaluación De Modelos En El Aprendizaje Automático*, https://www.datasource.ai/es/data-science-articles/metricas-de-evaluacion-de-modelos-en-el-aprendizaje-automatico, 2020.

[36] Google Developers, *Clasificación: ROC y AUC*, https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es, 2020.

[37] J. M. Heras, *Precision, Recall, F1, Accuracy en clasificación*, https://www.iartificial.net/precision-recall-f1-accuracy-en-clasificacion/, 2020.

[38] Wikipedia, *F-score*, https://en.wikipedia.org/wiki/F-score

[39] Wikipedia, *Validación cruzada*, https://es.wikipedia.org/wiki/Validaci%C3%B3n_cruzada

[40] Scikit Learn, *Cross-validation: evaluating estimator performance*, https://scikit-learn.org/stable/modules/cross_validation.html

[41] UCI Machine Learning Repository, *Heart Disease Data Set*, https://archive.ics.uci.edu/ml/datasets/Heart+Disease

[42] J. M. Heras, *Regularización Lasso L1, Ridge L2 y ElasticNet*, https://www.iartificial.net/regularizacion-lasso-l1-ridge-l2-y-elasticnet/, 2020.

[43] Stackoverflow, *Logistic regression python solvers' definitions*, https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions

[44] Sklearn, `sklearn.neighbors.KNeighborsClassifier`, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

[45] Sklearn, `sklearn.linear_model.LinearRegression`, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

[46] Sklearn, `sklearn.linear_model.LogisticRegression`, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

[47] Sklearn, `sklearn.ensemble.RandomForestClassifier`, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
