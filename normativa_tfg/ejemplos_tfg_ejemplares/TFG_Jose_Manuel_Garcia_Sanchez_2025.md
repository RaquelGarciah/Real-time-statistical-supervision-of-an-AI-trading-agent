# Aprendizaje automático para la clasificación de biopsias de próstata

**TRABAJO FIN DE GRADO**
Curso 2024/2025

FACULTAD DE CIENCIAS MATEMÁTICAS
GRADO EN INGENIERÍA MATEMÁTICA

**Autor:** José Manuel García Sánchez
**Tutor:** Ana Carpio

Madrid, 11 de febrero de 2025

---

## Índice general

- [Capítulo 1: Introducción y objetivos](#capítulo-1-introducción-y-objetivos)
- [Capítulo 2: Inteligencia artificial y redes neuronales](#capítulo-2-inteligencia-artificial-y-redes-neuronales)
- [Capítulo 3: Reconocimiento de dígitos en imágenes](#capítulo-3-reconocimiento-de-dígitos-en-imágenes)
- [Capítulo 4: Reconocimiento del cáncer de próstata en imágenes histopatológicas de biopsias](#capítulo-4-reconocimiento-del-cáncer-de-próstata-en-imágenes-histopatológicas-de-biopsias)
- [Capítulo 5: Conclusiones](#capítulo-5-conclusiones)
- [Bibliografía](#bibliografía)
- [Anexos](#anexos)

---

## Resumen

Este proyecto tiene como objetivo la aplicación de técnicas de aprendizaje profundo, en particular, redes neuronales convolucionales (CNN) para resolver un problema de clasificación de imágenes médicas. En concreto, clasificar un conjunto de imágenes histopatológicas de cáncer de próstata etiquetadas en: "cáncer" y "no cáncer".

El trabajo consta de una primera parte donde se estudian y se da una visión teórica de la inteligencia artificial y las redes neuronales. Conceptos claves, principios básicos de redes neuronales artificiales, sus tipos y aplicaciones. Además, un primer caso de uso, una red sencilla para el reconocimiento de dígitos numéricos a través de imágenes con un 98% de precisión en sus resultados.

Finalmente, se propone el modelo de estudio para el cáncer de próstata. Un primer modelo, una red capaz de clasificar nuestro conjunto de datos en dos clases: "cáncer" o "no cáncer" en función de si la imagen presenta células cancerosas o no, obteniendo una precisión del 94%. Un segundo modelo, basado en el índice de clasificación de Gleason en cuatro etiquetas y obteniendo un resultado del 80% de precisión en el modelo.

**Palabras clave:** Inteligencia Artificial (AI), Aprendizaje Automático, Redes Neuronales Convolucionales (CNN), Clasificación de Gleason, Cáncer, Imágenes histopatológicas.

---

## Abstract

This project aims to apply Deep Learning techniques, particularly convolutional neural networks (CNN), to solve a medical image classification problem. Specifically, the classification of a set of histopathological images of prostate cancer labeled as "cancer" and "no cancer."

The work consists of an initial section that studies and provides a theoretical overview of artificial intelligence and neural networks. Key concepts, basic principles of artificial neural networks, types, and applications are covered. Additionally, a preliminary use case is presented: a simple network for recognizing numeric digits in images, achieving 98% accuracy in its results.

Finally, the proposed study model for prostate cancer is presented. The first model is a network capable of classifying the dataset into two classes: "cancer" or "no cancer," depending on whether the image shows cancerous cells or not, achieving 94% accuracy. A second model is based on the Gleason classification model, categorizing into four labels and achieving 80% accuracy in the results.

**Key words:** Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), Convolutional Neural Networks (CNN), Gleason Classification, Cancer, Histopathological Images.

---

## Capítulo 1: Introducción y objetivos

La Inteligencia Artificial ha revolucionado la forma en que interactuamos con la tecnología, permitiendo que los sistemas aprendan, analicen y tomen decisiones de manera autónoma. Se ha convertido en una herramienta esencial capaz de lograr metas complejas, asemejándose al pensamiento del cerebro humano.

Lleva existiendo unos 50 años, desde sus inicios con programaciones de alto nivel para resolver problemas como el tablero de ajedrez o lógicas, pasando por la aparición del aprendizaje automático (a través de la repetición aprende y mejora la automatización) llegando al aprendizaje profundo con el desarrollo de las redes neuronales.

La IA la encontramos presente en diferentes áreas como: asistentes virtuales, detecciones de fraude, reconocimiento facial, del habla, biometría, etc. Así, nos lo hacen conocer empresas como: Microsoft, Google e IBM que han desarrollado y comercializado sus propias inteligencias. En el caso de Google nos serviremos de su lenguaje de programación: TensorFlow para construir nuestra propia red neuronal.

Por ello, este trabajo propone estudiar y conocer estas tecnologías que están en auge en estos últimos años y llevarlas a un caso práctico como es la clasificación de imágenes para la detección del cáncer de próstata. Se propone un primer modelo para realizar esta clasificación en dos categorías: "cáncer" y "no cáncer".

También, se propone un segundo modelo siguiendo el índice de clasificación de Gleason. En pocas palabras, la clasificación de Gleason es el método estándar para el diagnóstico de cáncer de próstata y esencial para determinar su pronóstico y tratamiento [1]. Tras realizar una biopsia, se toman una o varias muestras de tejido para su examen en microscopio. Esta clasificación hace referencia a cómo se ven las células cancerosas de la próstata y a qué tan probable es que el cáncer avance o se disemine. Una puntuación baja, G3, significa que el cáncer es de crecimiento lento y menos agresivo. Por el contrario, una puntuación alta, G5, es un cáncer de crecimiento rápido y más violento [2].

### 1.1 Objetivos

Los objetivos por desarrollar son los siguientes:

- Investigar sobre el aprendizaje automático y aprendizaje profundo desde sus inicios hasta su desarrollo en los últimos años.
- Investigar sobre los modelos de aprendizaje profundo más conocidos: redes neuronales artificiales (aprendizaje por imágenes).
- Resolver un problema de reconocimiento de imágenes como iniciación a la tecnología (primeros pasos con la IA).
- Resolver un problema de clasificación de biopsias de cáncer de próstata como muestra de la evolución de la tecnología. Analizar la capacidad de aprendizaje y de extracción que poseen las redes neuronales.
- Aplicar el índice de la clasificación de Gleason para evaluar el tejido canceroso en cada uno de sus grados.
- Contribuir con los resultados obtenidos en el juicio médico a la hora de sugerir un tratamiento o detección de la enfermedad.

---

## Capítulo 2: Inteligencia artificial y redes neuronales

### 2.1 Inteligencia artificial

Antes de comenzar a hablar de inteligencia artificial y redes neuronales, será conveniente definir los conceptos principales que engloban ambas ciencias de la computación.

#### 2.1.1 Inteligencia artificial

La inteligencia artificial (IA) se define como la habilidad de una máquina para replicar las mismas capacidades que tiene el ser humano: el razonamiento, el aprendizaje, la creatividad y la capacidad de planear. La máquina recibe datos, los procesa y responde a ellos siendo capaz de adoptar el comportamiento humano.

A pesar de que algunas tecnologías con inteligencia remontan su creación hace más de 50 años, la evolución en el ámbito de la informática, el trabajo con enormes cantidades de datos y nuevos algoritmos han contribuido de forma exponencial al desarrollo de estas inteligencias.

Los primeros pasos en desarrollar la IA los dieron los filósofos clásicos que intentaron describir el pensamiento humano como un sistema simbólico. Este tipo de razonamiento llevó a la computadora digital programable, una máquina basada en el razonamiento matemático abstracto.

No fue hasta 1955 cuando aparece el término "Inteligencia Artificial" de la mano de John McCarthy, un ingeniero informático de la universidad de Ivy League en Estados Unidos [3].

Según la definición de la Comisión Europea [4], encontramos dos tipos de IA:

- **Software:** asistentes virtuales, software de análisis de imágenes, motores de búsqueda, sistemas de reconocimiento de voz y rostro.
- **Inteligencia Artificial integrada:** robots, drones, vehículos autónomos.

La IA está presente en el día a día y se utiliza ampliamente en compras por internet, algoritmos de publicidad según el historial de red, asistentes digitales de atención personalizada e incluso en el ámbito de la salud. Su aplicación en este último campo abarca la investigación de líneas de ayuda, el análisis de patrones para la cura de enfermedades y la mejora de los diagnósticos (ejemplo que trataremos en este trabajo).

A su vez, la principal característica que diferencia a la Inteligencia Artificial de otros programas de ordenador es que no hay que programarla específicamente para cada escenario. Al hilo, aparecen dos nuevos conceptos íntimamente relacionados [5], pues a la Inteligencia Artificial podemos enseñarle cosas (**Machine Learning**, aprendizaje automático), pero también puede aprender por sí misma (**Deep Learning**, aprendizaje profundo). Sin olvidarnos del **Data Science** (Ciencia del dato) que a través del dato permite extraer información para ayudar a tomar mejores decisiones.

Las primeras investigaciones de la IA se centraron en las redes neuronales, inspiradas en el funcionamiento de las neuronas del cerebro humano. Posteriormente fue avanzando, aunque no tan rápido como se esperaba que llegara esa máquina capaz de replicar el pensamiento humano. Fue en la década de los 2000 cuando los gigantes tecnológicos comenzaron a desarrollar superordenadores e invertir dinero en su desarrollo.

#### 2.1.2 Aprendizaje automático

Este aprendizaje surge ante escenarios en los que las máquinas necesitan aprender grandes volúmenes de datos. Se plantea la pregunta de si un ordenador puede ir más allá de lo que nosotros le programamos y aprender por sí solo a realizar una acción sin recibir instrucciones explícitas.

Los métodos más comunes implementados para "hacer que las máquinas aprendan" son:

- **Aprendizaje supervisado:** utiliza *conjuntos de datos etiquetados* que entrenan algoritmos para clasificar datos o predecir resultados con precisión. Enseñamos al sistema los resultados que queremos obtener. Se agrupan en dos tipos:
  - **Clasificación:** agrupa datos en segmentos particulares. Algunos algoritmos más comunes son árboles de decisión y K-vecinos más cercanos.
  - **Regresión:** mide la relación entre una variable dependiente y una o más variables independientes. Algunos algoritmos comunes son la regresión de Ridge, Lasso, regresión de redes neuronales y regresión logística [6].

  Un ejemplo es un conjunto de datos de imágenes de frutas. Cada imagen está etiquetada con el nombre de fruta correspondiente (manzana, plátano, …). Se entrena al modelo para reconocer automáticamente las frutas de nuevas imágenes.

- **Aprendizaje no supervisado:** utiliza conjuntos de datos no etiquetados que entrenan algoritmos en la búsqueda de patrones desconocidos en los datos sin supervisión humana. Los modelos no aprenden a partir de los llamados datos de entrenamiento, son ellos mismos los que buscan sin supervisión patrones en el conjunto de datos. Existen tres tipos de algoritmos:
  - **Agrupación de clústers:** agrupación según las similitudes o diferencias.
  - **Asociación:** identificación de relaciones entre las variables de un conjunto de datos.
  - **Reducción de dimensionalidad.** Un ejemplo de ello es la reducción del ruido en una imagen para mejorar su claridad visual [4].

  Un ejemplo son las recomendaciones de contenido en las plataformas digitales y comercio en internet, utilizando algoritmos de filtrado colaborativo y técnicas de aprendizaje no supervisado para recomendar películas, series, música… a sus suscriptores.

- **Aprendizaje reforzado:** utilizado para aprender a través de prueba y error. Este aprendizaje transmite la retroalimentación al algoritmo después de que este genere la salida.

  Como ejemplo, un algoritmo que, tras la respuesta del paciente a un tratamiento, recibe una retroalimentación de su estado y ajusta el plan de tratamiento en consecuencia para lograr el mejor resultado posible.

Como conclusión, el aprendizaje automático utiliza modelos para construir, entrenar, probar e implementar soluciones gracias a algoritmos que permiten abordar y resolver problemas de regresión, predicción y clasificación.

#### 2.1.3 Aprendizaje profundo y redes neuronales

El aprendizaje profundo se basa en el uso de redes neuronales e intenta emular el comportamiento del cerebro humano con el fin de "aprender" teniendo como base grandes cantidades de datos [8].

El aprendizaje profundo se diferencia del aprendizaje automático por el tipo de datos con los que trabaja y los métodos mediante los cuales aprende.

Los algoritmos de aprendizaje automático aprovechan datos estructurados y etiquetados para realizar la predicción. Esto conlleva la definición de características específicas durante la entrada de datos para el modelo y su organización en tablas.

Por el contrario, el aprendizaje profundo elimina la necesidad de procesamiento y clasificación previa de los datos. Sus algoritmos pueden procesar datos no estructurados, como textos o imágenes, y automatizar la extracción de características reduciendo la dependencia de la intervención humana.

Llevados a un ejemplo, se tendría la siguiente estructura:

- **Datos de entrada:** en una clasificación de imágenes serían las imágenes.
- **Datos de salida:** en una clasificación de imágenes serían las etiquetas, como "perro", "gato", etc.
- **Una métrica para determinar la eficacia del algoritmo (función de pérdida o función objetivo):** debe medir la distancia entre la salida actual y la salida esperada. Los procesos de pendiente de gradiente y propagación inversa permiten ajustar y adaptar el algoritmo, realizando una nueva predicción con mayor precisión. Este ajuste es lo que llamamos aprendizaje.

Las redes neuronales profundas o redes artificiales tratan de imitar el cerebro a través de una combinación de entradas de datos, ponderaciones y sesgos. Estos elementos trabajan conjuntamente para reconocer, clasificar y describir con precisión los objetos dentro de los datos [9].

Estas redes están formadas por varias capas de nodos interconectados, cada uno sobre la capa anterior para refinar y optimizar la predicción o categorización. Esta progresión de cálculos a través de la red se denomina propagación hacia delante.

Las capas de entrada y salida de una red neuronal profunda se denominan capas visibles. La capa de entrada recibe los datos para el procesamiento y la capa de salida realiza la predicción o clasificación final.

En una red neuronal, la capa de entrada recibe las señales de entrada y las envía a la siguiente capa. Los parámetros, conocidos como pesos, se ajustan dentro de los nodos de las capas ocultas. Estos pesos determinan la importancia de cada característica en la predicción del valor objetivo. Es decir, cada nodo toma los datos de entrada, los multiplica por un valor de peso asignado y, en algunos casos, agrega un sesgo antes de pasar el resultado a la siguiente capa. El proceso se puede expresar como: **entrada × peso + sesgo = salida**.

#### 2.1.4 Aplicaciones del aprendizaje automático y del aprendizaje profundo

A pesar de que su gran popularidad y desarrollo se remonta a un periodo no muy lejano, la década de 2010, las áreas de investigación han conseguido superar retos y tareas para asimilar una máquina al cerebro humano y ofertar así las siguientes ventajas de estas técnicas [7]:

- **Mejorar los procesos de negocio:** el manejo del dato permite analizar la actividad empresarial y buscar las acciones más beneficiosas.
- **Adaptación a los cambios de la empresa o sector.** Los modelos pueden actualizarse con frecuencia al reentrenarlos con nuevos datos.
- **Detección automática de patrones:** identificar variables y relaciones relevantes entre los datos de diferentes ámbitos.
- **Ahorro de tiempo y costes.** La automatización de procesos obtiene resultados más precisos y rápidos que la manual.
- **Predicciones de tendencia.** A través del análisis de datos y los patrones de comportamiento se puede predecir qué productos tendrán una mayor demanda.
- **Soluciones innovadoras.** El aprendizaje automático puede aplicarse junto a otras técnicas, como el procesamiento del lenguaje natural, para extraer valores de datos y automatizar la clasificación de textos de reclamaciones o documentos legales.

---

### 2.2 Principios Básicos de Redes Neuronales Artificiales

#### 2.2.1 La neurona artificial

La neurona artificial es la base en el desarrollo de las redes neuronales. Más que una estructura física, se entiende como una función que trata de modelar matemáticamente el funcionamiento del cerebro humano.

En una primera etapa, la información recibida se representa por el vector $\vec{x} = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$. Cada entrada a su vez lleva asociada unos pesos sinápticos $w_i, \forall i = 1 \ldots n$ dando lugar al vector de pesos $\vec{w} = (w_1, w_2, \ldots, w_n) \in \mathbb{R}^n$.

La neurona realiza el producto escalar entre los vectores $\vec{x}$ y $\vec{w}$. A este producto se le dota de un umbral $w_0$, valor a partir del cual se considerará que la neurona se ha activado. Tiene como resultado una suma ponderada de las entradas por los pesos asignados más el umbral:

$$\text{Entrada} = \langle \vec{x}, \vec{w} \rangle + w_0 = \sum_{i=1}^{n} x_i w_i + w_0 = x_1 w_1 + x_2 w_2 + \cdots + x_n w_n + w_0$$

Esta suma recibe también el nombre de **entrada neta** de la neurona. La activación de la neurona vendrá determinada por lo que llamaremos **función de activación**. El uso de esta función de activación permite la concatenación de neuronas.

La entrada neta la evaluamos en esta función y obtenemos la salida de la red:

$$y = f(\text{entrada\_neta}) = f(x_1 w_1 + x_2 w_2 + \cdots + x_n w_n + w_0)$$

Aunque no existe un comportamiento biológico que establezca una equivalencia exacta con las neuronas del cerebro, la función de activación es una herramienta que permite aplicar las redes neuronales en la resolución de una amplia variedad de problemas reales.

#### 2.2.2 Funciones de activación

Las principales funciones de activación usadas por las redes neuronales artificiales, generalmente de tipo escalón, lineal o sigmoidal, son [11]:

- **Función de Salto Binaria.** Fue una de las primeras que se usaron en las estructuras de redes más primitivas como son el Perceptrón y Hopfield. Puede tomar tanto el valor 0 como 1 en la discontinuidad de la función. Es una función que anula todos los valores a la izquierda de la discontinuidad y establece en uno todos aquellos valores que están a la derecha.
- **Función Unidad Rectificada Uniforme (RELU).** Considera únicamente los valores positivos para la salida. Es la más usada por su menor coste computacional.
- **Función sigmoide.** Refleja la curva de aprendizaje de cualquier red, penalizando aquellos valores cercanos a cero o a uno. Su coste computacional es bastante elevado lo que hace que no sea usada en las capas intermedias.
- **Función tangente hiperbólica (Tanh).** Similar a la sigmoide. En este caso, devuelve valores entre (-1, 1) en lugar de (0, 1). De igual modo, refleja muy bien la curva de aprendizaje del modelo.

#### 2.2.3 Función de pérdida y optimizador

Con los datos de entrenamiento listos y la estructura de nuestro modelo definida, aún queda tomar las siguientes decisiones:

- **Función de pérdida:** indica cuánto nos hemos equivocado en nuestra predicción. Las más comunes son error cuadrático medio en problemas de regresión y entropía cruzada binaria en problemas de clasificación.
- **Optimizador:** ajusta los pesos del modelo minimizando la función de pérdida. El algoritmo "Adaptive Moment Estimation (Adam)" es el más popular para redes neuronales profundas [13].

#### 2.2.4 Métrica de rendimiento. Matriz de confusión

Es esencial elegir una métrica adecuada para medir el rendimiento de nuestra red. Esta métrica representa cómo de óptimos son los resultados obtenidos en cada modelo y servirá para hacer comparaciones y discernir a la hora de diseñar nuevos modelos.

La **matriz de confusión** es una herramienta que nos permite visualizar el rendimiento de un algoritmo de aprendizaje supervisado. Cada columna de la matriz representa el número de predicciones de cada clase, mientras que cada fila representa las etiquetas de las clases reales [14].

Para ello, la matriz de confusión utiliza los valores de:

- **Verdadero positivo (VP):** persona que tiene cáncer y el modelo lo clasifica como cáncer.
- **Verdadero negativo (VN):** persona que no tiene cáncer y el modelo lo clasifica como no cáncer.
- **Falso positivo (FP):** persona que no tiene cáncer y el modelo lo clasifica como cáncer.
- **Falso negativo (FN):** persona que tiene cáncer y el modelo lo clasifica como no cáncer.

Otras métricas utilizadas en los modelos de clasificación son [14]:

**Exactitud (accuracy):**
$$\text{Exactitud} = \frac{VP + VN}{VP + VN + FP + FN}$$

**Precisión:**
$$\text{Precisión} = \frac{VP}{VP + FP}$$

**Sensibilidad (recall):**
$$\text{Sensibilidad} = \frac{VP}{VP + FN}$$

**Valor F1 (f1-score):**
$$F1 = \frac{2 \times \text{precisión} \times \text{sensibilidad}}{\text{precisión} + \text{sensibilidad}}$$

---

### 2.3 Redes Primarias

#### 2.3.1 Perceptrón

Es el modelo más sencillo de red neuronal artificial, diseñado para clasificar datos linealmente separables. Su estructura consta de varias entradas $(x_1, x_2, \ldots, x_n)$ con pesos asociados $(w_1, w_2, \ldots, w_n)$ y una única salida $(y)$ que lleva asociada una función de activación.

#### 2.3.2 Redes Hopfield

Propuesta por John Hopfield en 1982, la red Hopfield emerge como un modelo innovador que rompe con la arquitectura tradicional utilizada hasta ese momento para almacenar y recuperar patrones de información. A diferencia de redes artificiales como el Perceptrón o ADALINE que propagan la información de las primeras capas a las últimas, la red Hopfield envía cada salida como una nueva entrada utilizando la recursividad. Este enfoque la hace adecuada para su uso en el aprendizaje no supervisado, ya que permite presentar los datos directamente a la red sin tener en cuenta información sobre la salida esperada.

Se utiliza por su complejidad para resolver problemas de optimización. El ejemplo más destacado es el **problema del viajante**, enunciado como: *"Dadas N ciudades tenemos como objetivo visitarlas todas una vez, partiendo de una cualquiera, y recorriendo la menor distancia posible"*. La red Hopfield se diseña como una red con N×N neuronas y la posibilidad de llegar a la ciudad N en un instante.

$$C_{ij} = \begin{cases} 1 & \text{si se llega a la ciudad } i \text{ en el instante } j \\ 0 & \text{en otro caso} \end{cases}$$

El objetivo es definir una función objetivo a minimizar el problema que se quiere resolver e incluir las restricciones que el planteamiento presente.

**Ejemplo: Problema del viajante**

Tomamos un total de 10 ciudades con las siguientes posiciones [15]:

| Ciudad | Posición |
|--------|----------|
| A | (0.25, 0.16) |
| B | (0.85, 0.35) |
| C | (0.65, 0.24) |
| D | (0.70, 0.50) |
| E | (0.15, 0.22) |
| F | (0.25, 0.78) |
| G | (0.40, 0.45) |
| H | (0.90, 0.65) |
| I | (0.55, 0.90) |
| J | (0.60, 0.25) |

La neurona tendrá N×N = 100 neuronas y 9900 conexiones (cada neurona se conecta con todas las demás menos consigo misma).

Por tanto, el orden en que se recorren las ciudades una vez y con la mínima distancia es: **F-I-D-B-H-C-G-E-A-J**

---

### 2.4 Redes Convolucionales (CNN)

Las redes neuronales convolucionales se distinguen de otras redes por su rendimiento superior en el tratamiento de imagen, voz o señales de audio [16]. Se componen de tres tipos principales de capas:

- **Capa convolucional.** Es la primera capa de la red y donde se realizan la mayoría de los cálculos. Su función principal es aplicar una operación matemática llamada convolución, la cual permite extraer características locales de los datos de entrada (por ejemplo, una imagen). El filtro se mueve por los campos receptivos de la imagen para comprobar si la característica está presente. El resultado final de los productos escalares se conoce como **mapa de características**. Este proceso ayuda a identificar patrones y características importantes, como bordes, texturas o formas, sin necesidad de procesar la imagen completa de manera explícita.

- **Capa de agrupación.** También conocida como submuestreo, permite disminuir la dimensión reduciendo el número de parámetros de entrada. Existen dos agrupaciones principales:
  - *Agrupación máxima:* el filtro selecciona el píxel con valor más alto y lo envía a la matriz de salida.
  - *Agrupación media:* el filtro calcula el valor promedio dentro del campo receptivo y lo envía a la matriz de salida.

  Aunque en esta capa existe la pérdida de información, se obtienen beneficios al reducir la complejidad, mejorar la eficiencia y limitar el riesgo de sobreajuste.

- **Capa totalmente conectada.** Cada nodo de la capa salida está conectado directamente a un nodo de la capa anterior. Esta capa realiza la tarea de clasificación basándose en las características extraídas de capas anteriores y sus correspondientes filtros.

Las arquitecturas más comunes de redes neuronales convolucionales son:

- **LeNet:** la más exitosa de redes convolucionales para la lectura de códigos postales, dígitos, etc.
- **AlexNet:** es la evolución de la anterior, con mayor profundidad y capas convolucionales apiladas.
- **GoogleNet, VGGNet,** etc.

---

### 2.5 Redes Recurrentes (RNN)

Una red neuronal recurrente es un tipo de red neuronal artificial donde las conexiones entre nodos forman un ciclo. Esto permite que la red pueda mantener y utilizar información previa en la secuencia (como si tuviera memoria), lo que hace que sea útil para tareas como el procesamiento del lenguaje natural y el análisis de series temporales [19].

Estas redes procesan secuencias de datos teniendo en cuenta el orden temporal de la información. A diferencia de las redes neuronales tradicionales, una red neuronal recurrente puede mantener un estado o memoria previa, lo que permite que la salida de la red en un momento dado sea influenciada por elementos anteriores de la secuencia. Esto se logra gracias a la recursividad que envía la salida de la red de vuelta a sí misma, actuando como un bucle de retroalimentación.

Los problemas de inteligencia artificial que se benefician del uso de estas redes son aquellos relacionados con datos secuenciales y temporales. Tareas como: la traducción automática, la generación de texto, y el reconocimiento de voz.

---

### 2.6 Redes Generativas (GAN)

Una red generativa antagónica (GAN) es una red con arquitectura de aprendizaje profundo utilizada para generar imágenes sintéticas. Recibe el nombre de *antagónica* porque entrena dos redes neuronales en paralelo para que compitan entre sí generando nuevos datos más auténticos a partir de un conjunto de datos de entrenamiento determinado [21].

La arquitectura GAN tiene varias aplicaciones en distintos sectores [21]:

- **Generación de imágenes.** Crear imágenes realistas mediante indicaciones basadas en texto o modificando imágenes ya existentes.
- **Generación de datos de entrenamiento para otros modelos.** El aumento de datos aumenta artificialmente el conjunto de entrenamiento mediante la creación de copias modificadas de un conjunto de datos ya existente.
- **Generación de modelos 3D a partir de datos 2D o imágenes escaneadas.** Por ejemplo, en el ámbito de la salud, las redes generativas combinan radiografías y exploraciones corporales para crear imágenes realistas de órganos destinadas a simulaciones quirúrgicas.

Su funcionamiento consiste en enfrentar dos redes:

- **Red generadora:** coge como entrada un vector aleatorio y lo codifica como una imagen sintética.
- **Red discriminadora:** coge como entrada una imagen (real o sintética) y predice si la imagen es del conjunto de entrenamiento o de la red generadora.

La competencia mejora ambas redes hasta alcanzar el equilibrio [21].

---

## Capítulo 3: Reconocimiento de dígitos en imágenes

### 3.1 Planteamiento y datos del problema

Como primer problema al uso de redes neuronales tomaremos un modelo que, dada una imagen en la que se puede ver un dígito, nos diga a qué número se corresponde. Para ello, se va a contar con un conjunto de datos formado por 7500 registros de imágenes que representan los dígitos del 0 al 9.

Las imágenes tienen un formato de 28×28 píxeles. Cada píxel toma un valor entre 0 (negro) y 255 (blanco). El conjunto de imágenes está tomado del conjunto de datos: "Modified National Institute of Standards and Technology (MNIST)". Es uno de los más conocidos y utilizados en el entorno del aprendizaje automático [23].

Además, usaremos la biblioteca de aprendizaje automático "Keras" con el fin de reconocer los dígitos escritos a mano y poder clasificar sus imágenes.

### 3.2 Modelos propuestos

**Modelo 1:** Red neuronal sencilla de clasificación secuencial o red completamente conectada. La primera capa de entrada con 784 neuronas (igual al número de columnas del conjunto de datos). La capa de salida con 10 neuronas (igual al número de dígitos a predecir). Las capas ocultas escogidas son de un total de 4, tres de ellas con 256 neuronas y una cuarta con 128 neuronas. En este modelo se encuentran conectadas todas las neuronas de una capa a otra.

Para el modelo se han llevado a cabo las siguientes hipótesis: los datos han sido escalados (divididos entre 255 para normalizar), se ha usado la función de pérdida de entropía cruzada, el optimizador ADAM y la medida del rendimiento será cuántos aciertos se obtienen a partir de las respuestas esperadas.

Las métricas estudiadas son:

- **Precisión (accuracy):** indica las predicciones correctas del modelo en el conjunto de entrenamiento.
- **Pérdida (loss):** indica el error cometido por el modelo en el conjunto de entrenamiento.
- **Pérdida de validación (val_loss):** indica el error en el conjunto de prueba.
- **Precisión de validación (val_accuracy):** indica las predicciones correctas en el conjunto de prueba.

A partir de los resultados obtenidos, se ha obtenido una precisión de prueba del **97%** en el conjunto de datos MNIST.

**Modelo 2:** Red Convolucional de un bloque. Formada por una capa convolucional de tamaño 32 con ventanas 3×3 totalmente solapadas. Están seguidas de una capa de agrupación máxima de 2×2 sin solapamiento. A continuación, una capa de aplanado, tras ella una capa densa de tamaño 32 con función de activación RELU. Por último, otra capa densa con 10 neuronas y función de activación softmax.

**Modelo 3:** Red Convolucional de dos bloques. Análogo al modelo 2 pero duplicando la capa convolucional y la de agrupación máxima.

Resultados obtenidos:

| Modelo | Precisión |
|--------|-----------|
| Modelo 2 (1 bloque) | 98.50% |
| Modelo 3 (2 bloques) | 98.98% |

El modelo 3 obtiene mejores resultados a pesar de que el tiempo de procesamiento es mayor (casi el doble de tiempo en cada iteración).

Variando el tamaño de las capas convolucionales en el modelo 3:

| Tamaño de capas | Precisión | Pérdida |
|-----------------|-----------|---------|
| 32 | 98.94% | 3.58% |
| 64 | 98.89% | 5.47% |
| 128 | 98.91% | 3.93% |
| 256 | 98.91% | 3.64% |

Se concluye que, dada la sencillez del modelo, se obtienen los mejores resultados con un modelo de 32 capas, ya que combina buen valor de precisión, mayor rapidez y eficiencia.

---

## Capítulo 4: Reconocimiento del cáncer de próstata en imágenes histopatológicas de biopsias

### 4.1 Marco de trabajo

La clasificación supervisada consiste en asignar la etiqueta o clase correcta a una observación, tras haber procesado un conjunto de ellas ya etiquetadas [25]. En particular, se aborda un problema de clasificación supervisada, cuyos datos son imágenes.

Para ello, se ha empleado un ordenador Intel Core i7 - 7ª Generación con memoria RAM de 16 GB y un disco duro SDD de 256 GB. El sistema operativo usado es Windows 10 y con un tiempo medio de 4 horas en el desarrollo de cada modelo.

Los lenguajes de programación utilizados son: Python, Matlab y Anaconda. Principalmente se usa Anaconda, una aplicación que configura un cuaderno virtual llamado Jupyter Notebook permitiendo la ejecución de código por celdas e impresión de resultados por pantalla. Las librerías empleadas incluyen: TensorFlow, ImageDataGenerator, Keras, Matplotlib.pyplot, Numpy, Pandas, Seaborn, etc.

### 4.2 Planteamiento del problema

Los pasos para entrenar una red neuronal son:

1. **Recopilación de datos:** se recopilan imágenes histopatológicas de pacientes anónimos para el diagnóstico del cáncer de próstata.
2. **Preprocesamiento de datos:** las imágenes se convierten a escala de grises y se realiza un balanceo del conjunto de datos.
3. **Diseño de la red neuronal:** se elige una CNN capaz de procesar y analizar imágenes médicas histopatológicas.
4. **División de datos:** el conjunto de datos se divide en dos tercios para entrenamiento y un tercio para prueba.
5. **Entrenamiento de la red neuronal:** la red ajusta sus pesos y sesgos a medida que se le presentan más ejemplos.
6. **Evaluación del rendimiento:** se analiza la precisión de las predicciones y se construye la matriz de confusión.

### 4.3 Datos del problema

Para afrontar el problema, se usa el conjunto de imágenes histopatológicas de cáncer de próstata "DiagSet-A" [26]. Este conjunto contiene un total de 192 imágenes.

El conjunto venía desbalanceado: 52 imágenes de no cáncer frente a 140 de cáncer. Por ello, se recurrió a la librería `RandomOverSampler`, que implementa una técnica de sobremuestreo replicando instancias existentes hasta que todas las clases tienen la misma cantidad de muestras (140 imágenes cada clase).

El siguiente paso fue realizar pequeñas modificaciones a grupos de imágenes: rotación, ampliación e inclinación. El valor de la modificación se define de forma aleatoria para cada imagen, con el fin de generar mayor número total de imágenes e introducir algo de ruido para verificar la robustez del modelo.

Usando la función `train_test_split` se divide el conjunto de datos en dos tercios para entrenamiento y un tercio para prueba.

### 4.4 Red CNN. Estructura

Se plantearon diferentes modelos, de más sencillo a más complejo, para valorar el rendimiento de la red.

#### Modelo 1: Clasificación en dos categorías: "cáncer" y "no cáncer" (sin ruido)

El conjunto de datos está compuesto por 280 imágenes. La arquitectura de la red es la siguiente:

- **Entrada:** imágenes de tamaño 1024×1024 en un único canal (escala de grises), normalizadas al intervalo [0, 1].
- **Capas Convolucionales:** primera capa convolucional de tamaño 8 con ventanas 3×3. El resto son bloques formados por dos capas convolucionales con función de activación RELU y de tamaño creciente (de 8 a 128), seguidas de una capa de agrupación máxima.
- **Capas densas:** dos capas densas tras el aplanamiento. Una de tamaño 256 con función de activación RELU.
- **Salida:** última capa con función de activación sigmoide para devolver una probabilidad entre 0 y 1 ("cáncer" o "no cáncer").

Función de pérdida: entropía cruzada binaria. Optimizador: ADAM con tasa de aprendizaje $1 \times 10^{-4}$.

**Resultados del Modelo 1:**

|  | Precisión | Recall | F1-Score | Support |
|--|-----------|--------|----------|---------|
| cancer | 0.96 | 0.98 | 0.97 | 47 |
| no_cancer | 0.98 | 0.96 | 0.97 | 47 |
| **accuracy** | | | **0.97** | 94 |

#### Modelo 2: Clasificación en dos categorías: "cáncer" y "no cáncer" (con ruido)

El conjunto de datos está compuesto por 560 imágenes (incluidas imágenes con modificaciones). La arquitectura de la red es la misma que en el modelo 1.

**Resultados del Modelo 2:**

|  | Precisión | Recall | F1-Score | Support |
|--|-----------|--------|----------|---------|
| cancer | 0.96 | 0.93 | 0.94 | 94 |
| no_cancer | 0.93 | 0.96 | 0.94 | 93 |
| **accuracy** | | | **0.94** | 187 |

#### Modelo 3: Clasificación en cuatro categorías: "cáncer G3", "cáncer G4", "cáncer G5" y "no cáncer"

En este modelo se aumenta la dificultad y, siguiendo la clasificación de Gleason, se entrena la red neuronal para clasificar en cuatro categorías. La última capa de salida tiene tamaño 4. La función de pérdida se actualiza a `sparse_categorical_crossentropy`.

**Resultados del Modelo 3:**

|  | Precisión | Recall | F1-Score | Support |
|--|-----------|--------|----------|---------|
| cancer G3 | 0.79 | 0.86 | 0.83 | 22 |
| cancer G4 | 0.68 | 0.83 | 0.75 | 23 |
| cancer G5 | 0.94 | 0.65 | 0.77 | 23 |
| no_cancer | 0.77 | 0.77 | 0.77 | 22 |
| **accuracy** | | | **0.78** | 90 |

**Conclusiones de los modelos:**

- La comparativa entre el modelo 1 y modelo 2 muestra que la red convolucional de 5 bloques mantiene buenos resultados a pesar de introducir ruido. El modelo 1 tiene una tasa de acierto del 97% y el modelo 2 del 94%; la pérdida del 3% en la precisión es asumible dado el ruido introducido.
- Los valores de sensibilidad en los modelos 1 y 2 son mayores al 90%, lo que indica que ambos son muy efectivos en la identificación de casos con cáncer.
- El modelo 3 obtiene un rendimiento de casi el 80%, lo que refleja la mayor dificultad de clasificar cuatro categorías.
- La eficacia disminuye cuando se quiere profundizar en el estado en que se manifiesta el cáncer dentro de la clasificación "cáncer".

---

## Capítulo 5: Conclusiones

La inteligencia artificial se ha convertido en una herramienta indispensable hoy en día, en un mundo donde la información crece a un ritmo exponencial y las decisiones más críticas dependen cada vez más de la capacidad para procesar y analizar datos, aprender patrones, automatizar procesos y generar soluciones innovadoras. En el ámbito de la salud, estas decisiones que se toman pueden implicar la diferencia entre la vida y la muerte.

En particular, las redes convolucionales han emergido como una de las herramientas más poderosas, inspiradas en el funcionamiento del cerebro humano para identificar patrones y relaciones en datos, a priori, desestructurados. Han resultado ser especialmente eficaces en la tarea de clasificación de imágenes, donde su capacidad de abstracción y aprendizaje de características a partir de simples imágenes nos han ayudado a identificar imágenes de células cancerosas frente a aquellas que no lo son.

Se comenzó con una introducción teórica a los conceptos claves, seguida de un problema propuesto sencillo para el reconocimiento de dígitos numéricos donde se obtuvieron unos resultados con bastante éxito (98% de precisión) tratado por una red convolucional sencilla de dos bloques.

Seguidamente, se consideró un caso práctico aplicado en el ámbito de la medicina: la clasificación de imágenes de biopsias de próstata. El primer paso fue preprocesar las imágenes, transformándolas en matrices de una dimensión y a escala de grises. A continuación, se aplicaron técnicas para equilibrar un conjunto de datos desbalanceado. Como resultado, se logró una precisión del 94% en la clasificación binaria, distinguiendo entre imágenes de tejidos cancerosos y no cancerosos.

Al ampliar el problema a una clasificación de cuatro clases con el objetivo de diferenciar tipos específicos de tejidos cancerosos, el modelo alcanzó un 80% de precisión.

Se espera que este trabajo dé apoyo a todos los informes y estudios que existen sobre las redes neuronales, sirviendo de guía a cualquier persona que desee aprender sobre la inteligencia artificial y, en concreto, cómo aplicar modelos de redes convolucionales en la rama sanitaria para la resolución del problema de clasificación de imágenes de tejido canceroso. Los resultados obtenidos subrayan la importancia de seguir explorando y perfeccionando estas técnicas.

---

## Bibliografía

[1] F. Marginean e I. Arvidsson, "An Artificial Intelligence-based Support Tool for Automation and Standardisation of Gleason Grading in Prostate Biopsies," Division of Urological Cancers, Lund University, Malmö, November 14, 2020.

[2] S. M. Shah, "Sistema de puntuación de Gleason." Department of Urology, The Icahn School of Medicine at Mount Sinai, New York.

[3] G. Boy, "Diferencias entre Data Science, Inteligencia Artificial, Machine Learning y Deep Learning," Programmatic Spain (PS). Píldoras Educativas, Nivel Medio, vol. I, 9/08/2021.

[4] Parlamento Europeo, "Temas Digital: Inteligencia Artificial," 8 Septiembre 2020. [En línea]. Disponible: https://www.europarl.europa.eu/topics/es/article/20200827STO85804/que-es-la-inteligencia-artificial-y-como-se-usa

[5] R. Alonso, "IA, Machine Learning y Deep Learning ¿Cuál es la diferencia?" HZ Hard Zone, 17 Septiembre 2024.

[6] Empresa Alteryx, "Artículo: Aprendizaje supervisado frente a no supervisado." [En línea]. Disponible: https://www.alteryx.com/es/glossary/supervised-vs-unsupervised-learning

[7] Asociación para el Desarrollo de la Ingeniería del Conocimiento, "Los sistemas de IA aprenden de tus datos. Machine & Deep Learning," IIC - Instituto de ingeniería del conocimiento.

[8] D. Valverde Menasalvas, "Reconocimiento de imágenes médicas mediante aprendizaje automático," Trabajo Fin de Grado, UCM, 2021.

[9] C. Stryker y E. Kavlakoglu, "What is artificial intelligence (AI)?" IBM, Think Topics, 9 Agosto 2024.

[10] A. de Francisco, "Artificial: La Nueva Inteligencia y su aplicación en Nefrología," Nefrología al día. Sociedad Española de Nefrología, Act. 29/01/2024.

[11] F. S. Caparrini, "Redes Neuronales: una visión superficial," Depart. de Ciencias de la Computación e Inteligencia Artificial. Universidad de Sevilla, 16 Marzo 2022.

[12] S. Navarro, "¿Qué es una función de activación en Deep Learning?" Keepcoding, 22 enero 2025.

[13] S. Bock y M. Weiß, "A proof of local convergence for the adam optimizer," International Joint Conference on Neural Networks, 2019.

[14] J. I. B. Arce, "La matriz de confusión y sus métricas," Health BIG DATA, 26 julio 2019.

[15] J. M. S. Ramos, "Introducción a las redes neuronales artificiales," Trabajo Fin de Grado, UCM, 2022.

[16] IBM, "¿Qué son las redes neuronales convolucionales?" IBM Think Topics.

[17] J. Lapeña-Motilva y Á. Sanchez-Ferro, "Inteligencia Artificial en la enfermedad de Parkinson y otros trastornos del movimiento," Revista Kranion, vol. 18, p. 60, 14/06/2023.

[18] S. Navarro, "Arquitectura VGG16 y VGG19 en Deep Learning," keepcoding, 1 julio 2024.

[19] F. Analytics, "RNN (Red Neuronal Recurrente)," Foqum, Empowers your success, 2023.

[20] R. Cañadas, "Tecnología. Redes Neuronales Recurrentes," Ab Datum, 22/11/2021.

[21] A. Amazon, "¿Qué es una GAN?" [En línea]. Disponible: https://aws.amazon.com/es/what-is/gan/

[22] "Redes GAN: ¿Qué son? Características, funciones y ventajas," REDESiNFORMATICAS, 2023.

[23] J. McCaffrey, "Uso del conjunto de datos de reconocimiento de imágenes MNIST," MSDN Magazine Issues, vol. 29, nº 6, 06/2014.

[24] D. Calvo, "Definición de red neuronal artificial. Aprendizaje automático," Julio 2017.

[25] M. T. Barón, "Clasificación de Imágenes Médicas de Rayos-X mediante Redes Neuronales Convolucionales," TFG. Universidad de Valladolid, 2021.

[26] M. Koziarski et al., "DiagSet: a dataset for prostate cancer histopathological image classification," Nature. Scientific Reports, vol. I, nº 6780, p. 14, 2024.

---

## Anexos

### Figura A.1: Código modelo 1 del capítulo 3

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Modelo 1
# Construcción del modelo secuencial
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      # Aplanar la entrada de imágenes de 28x28
    layers.Dense(256, activation="relu"),       # Primera capa oculta con 256 neuronas
    layers.Dense(256, activation="relu"),       # Segunda capa oculta con 256 neuronas
    layers.Dense(256, activation="relu"),       # Tercera capa oculta con 256 neuronas
    layers.Dense(128, activation="relu"),       # Cuarta capa oculta con 128 neuronas
    layers.Dense(10, activation="softmax")      # Capa de salida con 10 clases (softmax para clasificación)
])

# Compilación del modelo
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Resumen del modelo
model.summary()

# Entrenamiento del modelo
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Pérdida en test: {test_loss}")
print(f"Precisión en test: {test_accuracy}")
```

### Figura A.2: Código modelo 2 del capítulo 3

```python
# MODELO 2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Cargar los datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos entre 0 y 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Cambiar la forma para agregar un canal (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Crear el modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 1 capa convolucional
    layers.MaxPooling2D((2, 2)),          # MaxPooling
    layers.Flatten(),                     # Aplanar la salida
    layers.Dense(32, activation='relu'),  # Capa densa intermedia
    layers.Dense(10, activation='softmax')  # Capa de salida (10 clases)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluar en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Pérdida en test: {test_loss}")
print(f"Precisión en test: {test_acc}")
```

### Figura A.3: Código modelo 3 del capítulo 3

```python
# Modelo 3
model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 1 capa convolucional
    layers.MaxPooling2D((2, 2)),          # MaxPooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 1 capa convolucional
    layers.MaxPooling2D((2, 2)),          # MaxPooling
    layers.Flatten(),                     # Aplanar la salida
    layers.Dense(32, activation='relu'),  # Capa densa intermedia
    layers.Dense(10, activation='softmax')  # Capa de salida (10 clases)
])

# Compilar el modelo
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Resumen del modelo
model1.summary()

# Entrenar el modelo
history = model1.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluar en el conjunto de prueba
test_loss, test_acc = model1.evaluate(x_test, y_test, verbose=2)
print(f"Pérdida en test: {test_loss}")
print(f"Precisión en test: {test_acc}")
```

### Figura A.5: Código modelo 1 del capítulo 4

```python
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Lectura datos
mypath ="C:/Users/user/imagen prostate cancer black balanceada /cancer"
cancer_files= [f for f in os.listdir(mypath)]

mypath ="C:/Users/user/imagen prostate cancer black balanceada /no_cancer"
no_cancer_files= [f for f in os.listdir(mypath)]

clase_c =['cancer' for i in range(0, len(cancer_files))]
clase_nc = ['no_cancer' for i in range(0, len(no_cancer_files))]

files=cancer_files + no_cancer_files
clase= clase_c + clase_nc

df =pd.DataFrame(list(zip(files, clase)), columns = ['FILENAME','CLASS'])

train, test = train_test_split(df, test_size=1/3 , stratify=df.CLASS)  # Discriminación 70-30

# [... resto del código de preparación de datos ...]

# Modelo CNN - 5 Bloques
from tensorflow.keras.models import import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(layers.Conv2D(8, (3,3), activation = "relu", input_shape= (1024,1024,1)))
model.add(layers.Conv2D(8, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(16, (3,3), activation = "relu"))
model.add(layers.Conv2D(16, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation = "relu"))
model.add(layers.Conv2D(32, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dense(2, activation="softmax"))

# Compilación del modelo
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])
model.summary()

# Entrenamiento de red
history= model.fit(
    train_dataset,
    epochs=100,
    verbose=1,
    validation_data=test_dataset
)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

Y_pred = model.predict(test_dataset)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['cancer', 'no_cancer']
c_matrix = confusion_matrix(test_dataset.classes, y_pred)
c_report = classification_report(test_dataset.classes, y_pred, target_names=target_names)
```

### Figura A.7: Código modelo 3 del capítulo 4

```python
# Modelo Red CNN - 5 bloques
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(layers.Conv2D(8, (3,3), activation = "relu", input_shape= (1024,1024,1)))
model.add(layers.Conv2D(8, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(16, (3,3), activation = "relu"))
model.add(layers.Conv2D(16, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation = "relu"))
model.add(layers.Conv2D(32, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dense(4, activation="softmax"))  # 4 clases: G3, G4, G5, no_cancer

# Compilación del modelo
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])
model.summary()

# Entrenamiento
history= model.fit(
    train_dataset,
    epochs=100,
    verbose=1,
    validation_data=test_dataset
)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

Y_pred = model.predict(test_dataset)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['cancer G3', 'cancer G4', 'cancer G5', 'no_cancer']
c_matrix = confusion_matrix(test_dataset.classes, y_pred)
c_report = classification_report(test_dataset.classes, y_pred, target_names=target_names)
```
