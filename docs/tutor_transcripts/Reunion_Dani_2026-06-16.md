# Transcripción — Reunión Dani 2

*Audio: Reunión Dani 2.m4a · duración 31.5 min · transcrito con faster-whisper (medium, es). Calidad de transcripción automática, revisar.*

**[00:32]** oyes es que está silenciado creo si ahora sí sí ahora sí sí es que la
**[00:46]** acabo de ver que he visto ahí en mi truco vale vale vale voy a ver si puedo
**[00:50]** compartirte lo ves ahora joder pues entonces no no te la puedo
**[01:01]** es que no sé es que yo creo que es por algo de permisos del ordenador pero
**[01:11]** ahora mismo se están compartiendo pantalla pero no parece ser que no va
**[01:21]** si no he planteado todo en tema de acuras y sigue sin verlo no
**[01:31]** perfecto vale lo que te estaba diciendo que he justificado de la decisión de
**[01:38]** por qué tres regímenes sino cuatro por digamos por interpretar y edad pero es
**[01:42]** que yo pensé como que la vale yo he hecho tres lectores y hay uno que está
**[01:49]** basado en la en la en los regímenes entonces ese ese detector que es una
**[01:57]** puntuación de cero a uno me daba es una probabilidad me daban casi todo en era
**[02:05]** como que salía o muy cerca del 0 muy cerca del 1
**[02:10]** y ese ese ese score lo que lo que te dice es cuánto de probable
**[02:17]** es la probabilidad que si yo te digo que está en crisis si la gente dice
**[02:22]** que está en crisis y mide o se acoge la probabilidad de el régimen que apuesta
**[02:30]** por si crisis va a corto coge la probabilidad que del régimen que va a
**[02:34]** largo sabes el contrario como el régimen y la probabilidad de que ahora
**[02:39]** mismo el régimen sea incompatible con lo que tú me estás diciendo o sea con
**[02:42]** lo que la gente me está diciendo entonces si me da que la gente supone
**[02:48]** que está en crisis y en verdad hay un 90 por ciento de probabilidad de
**[02:51]** que está en calma ahí es cuando interviene el detector
**[02:54]** vale entonces como lo que te digo me salía todo en uno o en cero casi todo
**[03:01]** casi todo y eso es porque el régimen normalmente o está en calma o está en
**[03:07]** crisis y hay muy pocas veces que salen estrés
**[03:11]** entonces yo pensé y porque y porque en vez de coger 3 cojo 2k
**[03:18]** pero me salía peor porque lo que hace que el estrés es como amorti o sea la
**[03:24]** los momentos que hay muchísima volatilidad los amortigua 6 los mete
**[03:29]** ahí como en un comodín que no hace nada y eso al final hace bien a la
**[03:33]** decisión final porque está no actúa sin saber sabes
**[03:37]** y eso de nuestra regimen es tienes una referencia bibliográfica si si si
**[03:43]** si si eso está eso está así además hay mucha como literatura de esto de los
**[03:48]** regímenes y tal no me he inventado yo aquí vale pues eso aquí he calibrado
**[03:55]** cada activo se calibra o sea cada régimen esto se cabía sobre el activo
**[04:00]** y aquí esto es el pedido de prueba vale luego lo mismo los umbrales de
**[04:06]** de los otros detectores que uno es el de la volatilidad y otro es el de
**[04:12]** cuánto cambia o sea la de si es coherente con el agente la decisión de
**[04:17]** hoy es coherente con el pasado vale te cuento esto el psa es este de
**[04:23]** coherente con el pasado vale entonces tú ves aquí que cuando
**[04:28]** salta es en el 064 que es el presentil 99 claro o sea este lo que
**[04:39]** me sale a mí es que este detector no interviene porque necesita mucho para que
**[04:44]** salte pero es que si lo bajo o sea he estado probando vale y cuando ha
**[04:50]** intervenido este detector y tampoco ha mejorado mucho la
**[04:57]** claro es que yo estoy tomando decisiones a lo que yo no puedo hacer
**[05:00]** es tomar decisiones en función a lo que a cómo me dan los resultados no
**[05:05]** claro no no poste yo te digo la ventana de test por eso por eso por eso o sea
**[05:13]** por eso no he cogido porque si fuera o sea si fuera así yo hubiera cogido el
**[05:18]** percentil 50
**[05:20]** pero lo que hay que hacer es tomar la decisión de preceptil en otro año y
**[05:25]** llevar de su antes y ver que funciona o sea en vez de su dividir su dividir la
**[05:30]** ventana de calibración correcto
**[05:35]** el último año para test por ejemplo para válida
**[05:41]** vale vale vale bueno pero eso no pasa nada eso es cambiarlo si eso es
**[05:48]** meterlo el este antes y ver qué pasa
**[05:59]** claro pero imagina vale vale la cosa es que si si veo que no acierta ya
**[06:05]** estoy haciendo como trampas
**[06:19]** porque que haría para el año que viene si yo quiero invertir más salió el 87
**[06:23]** vale por eso por eso y si ves que una decisión tomada sobre nuestros
**[06:33]** indicadores abierta sobre lo que tú quieres llamarlo si no te funciona ese
**[06:37]** tiene que ir fuera de tu estrategia porque no funciona
**[06:41]** vale tú mira tengo estas tres palancas pero estas tres palancas yo
**[06:44]** pongo tres percentiles donde también en validación y en test en verdad que
**[06:48]** está otra me columpio yo digo esto en validación pero luego esta es mentira por
**[06:53]** lo cual está en la búsqueda ya ya ya vale vale pero también o sea
**[07:02]** también de interpretabilidad este detecto sea esto es como de
**[07:07]** sustentibles a los cambios y si yo le pongo que el indicador este es
**[07:11]** imagínate el percentil 10 a la mínima va a saltar sabes o sea esto este
**[07:16]** indicador significa que cuando el agente diga algo muy incoherente con el mismo si
**[07:24]** yo le bajo mucho el indicador ya deja de ser esa lógica no
**[07:29]** si yo le pongo el percentil 50 igual es la mitad de las
**[07:35]** sabes lo que te digo sabes como que esto está pensado como para que sea
**[07:40]** la cola del final o sea el percentil 90 95 del final pero ninguno
**[07:45]** con ninguno de esos me salta porque son muy altos
**[07:49]** con el 95 entre el 95 del 99 no salta no interviene en ningún caso este este
**[07:57]** indicador a ver no vale 50 interviene pero yo lo veo no lo veo interpretable
**[08:13]** no sé qué opinas o sea yo quiero decir que sin saber cómo está hecho yo creo yo
**[08:22]** creo o sea yo podría tener una tabla un gráfico lo que sea que me enseñara
**[08:31]** según cambio el umbral en el presentil si cambio el umbral como repercute en
**[08:39]** la cura así de validación y test de variación y si veo que me van de la
**[08:47]** vale vale vale vale vale si tener una gráfica con los dos líneas una que sea
**[08:53]** si no sí porque si siguen el patrón o si claro pero claro
**[09:05]** no sé si se puede hacer el gráfico en el eje x en el eje x sí sí
**[09:14]** 50 55 60 65 75 en el eje y el valor del error de la cura así y por cada 50 55
**[09:27]** dos barritas sí sí sí sí sí sí un histograma
**[09:31]** y yo digo que trevín y variación van de la mano o sea trevín variación y
**[09:37]** test pero vale entonces cuál es el presentil óptimo que yo puedo poner
**[09:43]** para que no sé qué vale vale porque si no no generaliza ya si tú me dices
**[09:52]** que si hubieras invertido este año así pues mira que bien y me la juego
**[09:57]** el año que viene para hacerlo como me dice me dice que este año ha
**[09:59]** cambiado a mí siempre ha arruinado ya me digo vale muy bien
**[10:03]** esto no vale ya ya ya ya
**[10:08]** vale tienes que obsesionar tú puedes tener tus tres métodos esto es el
**[10:14]** montencano y si uno de los tres no funciona pues uno de los tres no
**[10:18]** funciona no pasa nada
**[10:22]** a ver aquí el método este principal es el de los regímeres aquí el que
**[10:28]** quieras el de los regímenes porque es el que el que da el signo
**[10:34]** sacaron según como yo porque he hecho como dos cosas ni el modelo que yo
**[10:40]** he establecido porque al final es un modelo como determinístico en verdad
**[10:42]** porque son reglas que yo puesto y lo que me dijiste tú de meterle un
**[10:46]** boosting con las probabilidades que se acaba de cada cosa y tal entonces
**[10:50]** esas son como las dos cosas que yo he hecho
**[10:54]** y eso y al final el ram o sea el de los regímenes lo que hace es darte la
**[11:01]** dirección y la posición
**[11:05]** y el psa lo que haría sería recortar
**[11:12]** pero como no salta es que no interviene y el del garch da la
**[11:18]** volatilidad que se usa en el en el otro en el ram o sea en el de los
**[11:24]** regímenes para dar la posición
**[11:32]** vale vale vale vale pues nada y luego que a curas y mira te digo el agente por
**[11:42]** sí solo da 0 con 37 creo que era 0 con 36 con mi modelo en plan estable como
**[11:49]** determinístico 0 con 43 y con el el boosting 0 53 y las variables que más
**[12:01]** las variables que más importancia tienen en el boosting son las mías no
**[12:06]** son las de la gente sabes son las de la del detector este del régimen la
**[12:11]** volatilidad anualizada la probabilidad de calma son lo que yo uso
**[12:18]** para modelo
**[12:23]** sólo dos cosas tiene un poco hasta medir si mañana de un día para otro
**[12:28]** verdad que la bolsa va a subir o bajar no en ese pequeño y sí sí sí sí que
**[12:33]** por frente creamos y de cerros pues pues pues pues de unos en cuanto
**[12:39]** cuando sube cuando baja solo tengo por el momento
**[12:42]** ay sí sí y así cuántos y cuando y nada mismo no lo sé pero
**[12:47]** está lo he sacado, lo he sacado por alguna vez, no sé dónde está, sí porque luego también he comprobado, he
**[12:58]** probado como con distintos regímenes, he probado de distintas ventanas de calibración de test
**[13:05]** para ver si era casualidad porque es un periodo alcista y va bien, sabes o en cuanto es que lo
**[13:14]** está, ahora no, pero está esto, aunque tengas una matrícula con función no vale
**[13:29]** no es que tenía un grafo, quería enseñarte, mira esto es la parte del p-valor, esto es el
**[13:42]** p-valor para ver si el, esto es el m10, lo probé con el m8, el m8 es el mío y el m10 es el boosting
**[13:49]** mira aquí está, aquí está, no ves la pantalla, vale, vale pues te digo, el régimen alcista
**[14:03]** sube 278 días y baja 123, 278 veces, 278.1 o 623, sí, sí, sí, sí, a ver según me dices
**[14:40]** 278 más 123 me has dicho, sí, sí o sea son dos años de 278 y 123, 278.1, sí, sí, sí,
**[14:59]** o sea 278, sí, imagínate que te pregunto, dime, dime, 278 entre 401 es el total, 278 más 123 es 401, 278 entre 401 es 69
**[15:18]** vale, o sea un 269 quiero decir, sí, 70% es alcista, si alguien te dice oye mira yo he hecho un modelo que dice siempre 1, siempre digo 1, siempre
**[15:29]** claro, el buy and hold, ese es el buy and hold, esa es la estrategia como clásica, por eso es abierta, más que el mío
**[15:37]** claro y entonces pedirán y entonces tomar todo lo que aportan, claro pero es que yo he cogido este como un ejemplo, o sea
**[15:44]** mi método, lo único que yo quiero que aporte, o sea lo que yo quiero demostrar con lo que yo estoy haciendo es que si yo le meto esto, me va a mejorar a la gente
**[15:55]** ya, ¿sabes? porque esto no es para este activo, esto es para más activos y para más agentes, ¿sabes? yo solo necesito como la probabilidad, yo de la gente lo que cojo es la dirección, la posición, ya
**[16:10]** no me parezca mal, pero te van a decir, yo te digo porque preparé la respuesta, te van a decir, creo yo, vale, es que para este ejemplo que tú te has traído la gente es una castaña
**[16:23]** pero es que tengo otro ejemplo
**[16:24]** y es fácil ganarlo, y el otro, o sea tú necesitas, es que una de las cosas que se critican en estos métodos es que tú vayas por otro lado, es que por ejemplo quiero decir
**[16:36]** el típico ejemplo de, quiero identificar si un tío es moroso, ¿vale? muy bien, ¿qué porcentaje de morosos tienes? un 5%, ¿cuánto en un moroso? un 95, ¿vale?
**[16:48]** pues entonces yo digo que nadie es moroso y voy a ganar, en un 95% de los casos y te dicen, ya hombre, claro, eso ya lo sé yo
**[16:57]** entonces por ahí, o sea, eso es, o sea, eso te lo van a decir, pero vamos seguro, eso no es que yo digo, a ver si tienes suerte no te lo dirán
**[17:03]** porque en estos modelos de predicción binaria siempre se comparan con predicciones naïve, la predicción naïve es decir la clara
**[17:13]** no estarán a decir, jo, qué malo es el método de la gente IA, porque aciertan un 30% de los casos cuando si siempre, si siempre digo uno, acierto en un 70
**[17:27]** claro, pero es lo que te digo, con otro activo el método de la gente IA será, esto es casualidad, literalmente, es casualidad que este activo es artista
**[17:39]** vale, pero... si es bajista, si tú dices tú lo vas a perder y, ¿sabes?
**[17:43]** necesitaría presentar un ejemplo en el que tú... que yo gani, vale, sí, sí
**[17:49]** y a la predicción naïve, pero lo único que a mí me pasaba en reuniones cuando yo iba, yo iba a Iberia y yo decía en Iberia
**[17:58]** mira, el modelo que yo he hecho es dos veces mejor que un modelo aleatorio
**[18:04]** y me decían, bueno, muy bien Dani, pero ¿quién te dice a ti que aquí en Iberia las predicciones las hacemos aleatoriamente?
**[18:13]** ay, perdona, ¿me das tus predicciones Iberia? sí, ay, pues entonces me ganas Iberia
**[18:19]** y claro, Iberia me decía, no te contrato, porque es verdad que tienes un método que es mejor que la gente IA
**[18:26]** pero es que el método que yo tengo en la compañía también es mejor que la gente IA
**[18:31]** con lo cual, ¿tú método qué me aporta? es que lo malo, lo malo es que... o sea, lo malo que yo veo, si no encuentras ese ejemplo
**[18:39]** sí, sí, pero creo que lo tengo, ¿eh? tú pierdes contra una predicción trivial
**[18:44]** que es decir siempre uno, si tú no tienes perfecto, pero si no te van a decir, vale, tienes un método que gana la gente IA
**[18:52]** y lo único que sigues a decir es que la gente IA, por lo menos este, es una mierda, vale, perfecto
**[18:57]** pero tu método es bueno, tu método tampoco es bueno, es que si digo siempre uno, es mejor que tu método
**[19:02]** y tú dices, ya, pero por lo menos es tu método
**[19:04]** eso no está justificado con que yo te diga que mi objetivo no es... o sea, porque yo no te puedo dar una estrategia
**[19:12]** o sea, yo lo que pretendo no es darte la mejor estrategia de inversión, porque si no sería millonario, me refiero
**[19:20]** sí, pero imagínate que yo hago un TFG y yo mira, el acierto de la gente IA tiene un 30
**[19:25]** y mi método es decir siempre uno y tengo un 70
**[19:28]** no, tiene un 59 y lo mío 53
**[19:31]** el que tiene un 59
**[19:34]** tiene un 59 de acura, sí
**[19:38]** si digo siempre uno, si digo siempre uno
**[19:41]** sí, das a un 69, ya, ya
**[19:43]** 69, yo digo, mira, que tengo un método, el trabajo se llama...
**[19:49]** humillando a los agentes IA con Machine Learning
**[19:52]** y te lo presento y digo, mira, 20 páginas de la gente IA con una transformer y con una LLM y todo lo que tu quieras
**[19:58]** y luego pongo mi metodología
**[20:00]** mi metodología consiste siempre en decir uno
**[20:04]** 4.1, comparativa
**[20:06]** yo decir un 70 y la gente IA un 30
**[20:09]** soy mejor, gracias a mi Machine Learning, que la gente IA
**[20:14]** y qué le digo yo
**[20:15]** no, pero espérate, yo tengo un boosting
**[20:17]** ah, sí, el boosting que un 70, no el boosting un 50
**[20:20]** pero si el boosting también es peor
**[20:24]** o sea, quiero decir
**[20:25]** vale, pero si yo, vale, una pregunta
**[20:28]** tu trabajo tiene que empezar, cuando tú presentes esto
**[20:31]** si no presentas más acciones
**[20:32]** la predicción de la gente IA es un 30
**[20:34]** si hiciéramos una predicción naïve
**[20:36]** sería un 70
**[20:38]** voy a intentar batir a los dos
**[20:40]** si no, es sino
**[20:42]** y con qué, vale, vale
**[20:44]** y con que yo encuentre un activo
**[20:47]** que va, que sea el contra ejemplo de eso, ya gano
**[20:51]** convendría en ese caso
**[20:53]** que el título de tu TFG
**[20:55]** tuviera el nombre del activo
**[20:57]** porque si no
**[20:59]** claro, te pueden decir
**[21:00]** cuántas veces lo has probado, 100
**[21:01]** y de las 100 veces, cuántas veces has encontrado un activo que qué
**[21:04]** 1
**[21:05]** ya
**[21:06]** o sea, si tú lo vienes con una metodología aplicada por los activos
**[21:09]** alguien que te lo compra, te querría decir
**[21:11]** en cuántos activos has probado
**[21:13]** claro
**[21:15]** salvo que tú digas, no, no, no, es un caso de estudio
**[21:17]** el SP500
**[21:18]** y así gano
**[21:19]** entonces tu título es
**[21:21]** así que a lo mejor en vez de lo que deberías centrar
**[21:23]** en el activo que gano y ya está
**[21:25]** claro, por lo menos
**[21:27]** y cuando te preguntes las pruebas con otros activos
**[21:29]** y tú no, me centrarás
**[21:31]** y callao
**[21:33]** claro, vale, es que yo lo tenía como
**[21:35]** más general, pero es verdad que
**[21:37]** que sí, tienes razón
**[21:39]** o sea, a la gente
**[21:41]** lo gano
**[21:43]** a la gente
**[21:45]** lo gano siempre
**[21:47]** al Bayern Hall, no
**[21:49]** a ti te interesa encontrar un activo
**[21:51]** que esté 50-50
**[21:53]** 50-50
**[21:55]** sí, para que
**[21:57]** la predicción esa mayoritaria
**[21:59]** siempre 1-0
**[22:01]** sí, que estén balanceadas las clases
**[22:03]** para que ya, una predicción mayoritaria va a decir 50-50
**[22:05]** o sea, va a centrarse en los casos
**[22:07]** y si lo dices aleatoriamente, sería
**[22:09]** un medio por medio, un cuarto
**[22:11]** a la gente le mete un 53
**[22:13]** y tú le metes un 55
**[22:15]** o sea, eso
**[22:17]** bueno, ni tan mal
**[22:19]** un 50, la gente un 53
**[22:21]** y tú un 55, vale
**[22:23]** vale, vale, vale
**[22:25]** es una de las cosas racistas
**[22:27]** o bueno, o el que tengas que lo analices en otro periodo
**[22:29]** que el periodo tenga 50-50
**[22:31]** y que se haga mejor la foto
**[22:33]** sí, que a lo mejor cojo
**[22:35]** en vez de esto, claro, es que estoy aquí
**[22:37]** cojo esto
**[22:39]** este catch
**[22:41]** claro, tienes una acción que pones para arriba
**[22:43]** deberías coger tu opción más irregular
**[22:45]** que a lo mismo sube que baje
**[22:47]** porque la tuya general va para arriba
**[22:49]** sí, no, no, esto es racista
**[22:51]** píjate una machunga
**[22:53]** píjate una de...
**[22:55]** vale, ya sé lo que tengo que buscar, lo que acabas de decir
**[22:57]** 50-50 o...
**[22:59]** malo
**[23:01]** porque si no vamos a pasar canutas
**[23:03]** porque es que la pregunta primero te va a hacer esa
**[23:05]** y te van a partir el discurso
**[23:07]** yo estoy muy nerviosa
**[23:09]** pero espérate, espérate
**[23:11]** claro, es que yo pensaba que era justificable
**[23:13]** pero si tienes razón lo que me estás diciendo
**[23:15]** sí, qué más da
**[23:17]** que bata algo malo
**[23:19]** si yo tengo algo mejor
**[23:21]** claro
**[23:23]** y para qué me estoy de una carrera de 4 años
**[23:25]** sabe
**[23:27]** se lo digo a mi madre o a mi abuela
**[23:29]** yo voy a decir siempre uno
**[23:31]** o con el boosting
**[23:33]** claro, o sea
**[23:35]** mi cosa era eso, que era como general
**[23:37]** sabes que sí, que en este
**[23:39]** tira más vayanholt
**[23:41]** porque es alcista, pero uno bajista
**[23:43]** te lo va, ya
**[23:45]** vale, pues sí, yo creo que es mejor
**[23:47]** yo creo que es mejor centrarlo en uno
**[23:51]** es que no he sacado de la cura
**[23:53]** así y todo eso
**[23:55]** porque hizo al principio como 10
**[23:57]** y es verdad que había
**[23:59]** no me acuerdo si uno o varios
**[24:01]** que sí que superaba el vayanholt
**[24:03]** el sarpe
**[24:05]** y el sarpe
**[24:07]** va como acompañado
**[24:09]** está como relacionado con la cura
**[24:11]** entonces creo que sé
**[24:13]** por dónde tengo que probarlo
**[24:15]** oye, yo tengo una duda que tenía
**[24:17]** lo del GARTS
**[24:19]** ¿tú predices la volatilidad de mañana?
**[24:21]** yo predijo
**[24:23]** la volatilidad
**[24:25]** de mañana anualizada
**[24:27]** porque, o sea
**[24:29]** no dividas por lo que tú me decías
**[24:31]** sí, sí
**[24:33]** de mañana, sí, sí
**[24:35]** si tú coges el GARTS
**[24:37]** y coges la predicción de volatilidad
**[24:39]** no la predicción del retorno
**[24:41]** no, de la volatilidad
**[24:43]** sí, de la volatilidad de mañana
**[24:45]** y eso lo haces con Python
**[24:47]** vale, o sea, tú lanzas
**[24:49]** la función de Python y te repredites
**[24:51]** la volatilidad de mañana
**[24:53]** tú con la de mañana
**[24:55]** y la de la predicción de la gente de mañana
**[24:57]** lo combinas todo
**[24:59]** y así saco la posición
**[25:01]** he visto un gráfico ahí que no entendía bien
**[25:03]** uno que ya se hace un momentito
**[25:05]** que
**[25:07]** se ve ahí
**[25:09]** era
**[25:11]** un gráfico de línea
**[25:13]** que iba como de la base
**[25:15]** no sé si estaba bajando o subiendo
**[25:17]** era un gráfico de líneas que salían tres o cuatro líneas
**[25:19]** no, otro
**[25:21]** era como la set
**[25:23]** si, si
**[25:25]** el de abajo
**[25:27]** que me
**[25:33]** está más bajo
**[25:38]** sí, no, pero esto es que hice
**[25:43]** las pruebas con lo de el K
**[25:45]** para ver que hay que acoger
**[25:47]** vale
**[25:53]** no, lo voy a quitar
**[25:55]** esto era para el K
**[25:57]** no le voy a dar más vueltas
**[25:59]** lo ves
**[26:01]** lo tienes en la pantalla
**[26:03]** lo tengo
**[26:05]** ya lo veo
**[26:07]** ya lo ves
**[26:09]** esto era para elegir el K
**[26:11]** o sea esto
**[26:13]** ¿pero qué representa la curva monza?
**[26:15]** pues representa la logero similitud
**[26:17]** ah, vale, perfecto
**[26:19]** era como
**[26:21]** haciéndolo cogiendo como cachos
**[26:23]** como
**[26:25]** la logero similitud
**[26:27]** sí, no, sí
**[26:29]** sí, sí, sí
**[26:31]** vale, bueno
**[26:33]** lo voy a quitar
**[26:35]** porque es que no voy a darle más vueltas
**[26:37]** el K es igual a tres ya solo por la literatura
**[26:39]** y porque es más interpretable
**[26:41]** mi única pregunta era
**[26:43]** K igual a dos o K igual a tres?
**[26:45]** porque cuatro no voy a poner en cuatro
**[26:47]** y K igual a dos da peor a cura
**[26:49]** ya está, lo voy a quitar
**[26:51]** eso era que estaba probando cosas
**[26:53]** y daba también K igual a tres
**[26:55]** vale, entonces
**[26:57]** nada más que consideras un caso
**[26:59]** en el que
**[27:01]** un socio naíf
**[27:03]** no gane
**[27:05]** o no gane de paliza
**[27:07]** a la gente guía
**[27:09]** y tú ganes a los dos
**[27:11]** me lo informas
**[27:13]** vale, mañana
**[27:15]** mañana tienes
**[27:17]** hueco y te
**[27:19]** no tienes nada mañana
**[27:21]** no tengo hueco
**[27:23]** y pasado
**[27:27]** martes
**[27:29]** a lo mejor a esta hora
**[27:31]** el martes a esta hora
**[27:33]** a lo mejor hasta ahora
**[27:35]** si, a lo mejor
**[27:41]** hace un calor
**[27:43]** aquí está lloviendo
**[27:45]** tengo un calor en la habitación
**[27:47]** entonces el martes a esta hora
**[27:49]** si tienes
**[27:51]** yo creo que el martes a esta hora puede
**[27:53]** vale
**[27:55]** vale
**[27:59]** yo creo que ahora
**[28:01]** lo tengo más
**[28:03]** más
**[28:05]** las decisiones mejor tomadas
**[28:07]** porque también lo de
**[28:09]** lo de
**[28:11]** el umbral del ram
**[28:13]** el que te dije que el otro día
**[28:15]** que había puesto
**[28:17]** cero con dos, cero con cuatro, cero con siete
**[28:19]** así, claro al final
**[28:21]** hice como el histograma este
**[28:23]** y me di cuenta a lo que te he dicho
**[28:25]** que estaba todo en el cero, en el uno
**[28:27]** para
**[28:29]** entonces sabes que da igual poner
**[28:31]** los medios
**[28:33]** lo he puesto al
**[28:35]** al cero con cinco
**[28:37]** vale, tu no te preocupes
**[28:41]** consigue ahora cierto ese
**[28:43]** me escribes o me lo coquetamos
**[28:45]** que yo voy a ser crítico con lo que leas
**[28:47]** no te preocupes que yo voy a ser crítico con lo que leas
**[28:49]** si yo leo alguna cosa que no me convence te lo voy a decir
**[28:51]** vale
**[28:53]** para evitarte el matraco
**[28:55]** vale, si hay una cosa que la veo estupenda
**[28:57]** te la diré, si hay una cosa que no lo digas así
**[28:59]** yo te lo diré
**[29:01]** vale
**[29:03]** vale
**[29:05]** vale
**[29:07]** pues voy a buscarlo ahora
**[29:09]** y para el martes hablamos
**[29:11]** y te mando todo en base al ejemplo ese
**[29:13]** y mejor
**[29:15]** vale, si, pero lo voy a encontrar porque ya te digo que
**[29:17]** cuando estuve mirando había cosas
**[29:19]** que si que batía el
**[29:21]** el baianjuel, pero no es algo de eso
**[29:23]** porque todas decisiones
**[29:25]** con validación
**[29:27]** y
**[29:29]** se comprueban en test
**[29:31]** vale
**[29:33]** vale
**[29:35]** vale
**[29:37]** vale
**[29:39]** vale
**[29:41]** vale
**[29:43]** vale
**[29:45]** vale
**[29:47]** vale
**[29:49]** vale
**[29:51]** vale
**[29:53]** vale
**[29:55]** vale
**[29:57]** vale
**[29:59]** vale
**[30:01]** vale
**[30:03]** vale
**[30:05]** vale
**[30:07]** vale
**[30:09]** vale
**[30:11]** vale
**[30:13]** vale
**[30:15]** vale
**[30:17]** vale
**[30:19]** vale
**[30:21]** vale
**[30:23]** vale
**[30:25]** va a saltar a la mínima sabes como que deja de ser interpretable pero si me
**[30:30]** funciona mejor es que cuando lo hice me estuve
**[30:36]** radiando con eso digo no sé
**[30:40]** y luego ya lo interpretamos
**[30:44]** funciona mejor en el 30 o en el 40 me lo enseñas intentamos justificar
**[30:50]** vale vale vale vale vale
**[30:54]** vale pues hablamos y el martes sobre esta
**[31:01]** hora en principio si te mando un correo y si el martes del restaurante yo creo que
**[31:08]** bueno así vale pues perfecto
**[31:12]** muchas gracias Dani
**[31:14]** Adiós
