# Estilo de Raquel + anti-IA (el redactor la lee SIEMPRE)

> Objetivo doble: que la memoria **suene a Raquel** (estudiante de matemáticas de la UCM) y que **pase el
> detector de IA y el de plagio**. Esta es la checklist que usan `redactor-tesis`, `estilo-raquel` y
> `detector-ia`.

## 1. PROHIBIDO (lista explícita)

**Puntuación / tics de IA:**
- **Guion largo `—` (raya) como muletilla** y **` - ` como conector o inciso**. Sustituir por **coma,
  paréntesis o punto**. (Tic de IA que Raquel quiere evitar a toda costa.)
- Punto y coma en cadena para encadenar tres ideas calcadas.

**Vocabulario AI-typical (no usar):**
- Inglés: *delve, delving, moreover, furthermore, in essence, it's worth noting*.
- Español: *cabe destacar, cabe mencionar, es importante mencionar/señalar, conviene subrayar/recordar, es
  esencial que, en esencia, podemos observar que, se puede afirmar que, vale la pena señalar, abordaremos,
  exploraremos* (tono de manual), *en resumen* al cierre de cada sección. Estos meta-comentarios anuncian la
  redacción; el énfasis va dentro de la frase (*lo decisivo es, esto subraya, merece atención*).

**Estructura:**
- **Conectores-etiqueta** al inicio de cada frase (*además, por otro lado* sin contraste real, *en conclusión*).
  Preferir **conexión implícita por orden lógico**.
- **Frases de longitud uniforme** (desviación típica baja → huele a IA).
- **Estructura tripartita** repetitiva (cada párrafo exactamente 3 frases).
- **Viñetas en el cuerpo** de la memoria (solo en apéndices). Prosa académica.

## 2. La voz de Raquel (cómo SÍ escribe)

De las muestras en `tesis_assets/estilo_raquel/` (Estructuras, Geometría Lineal, TFB):
- **Longitud de frase variable**: mezcla frases cortas (8–12 palabras) con largas (30+). σ alta, no uniforme.
- **Primera persona ocasional y natural**: "definimos", "observamos", "durante la calibración vi que…". Nada de
  "we" mayestático constante.
- **Voz activa** predominante ("denotamos", "tomamos"), pasiva solo cuando es técnica.
- **Rigor**: cada símbolo definido antes de usarse; demostraciones completas, no "por el Teorema X".
- **Conectores naturales**: "así", "de ahí que", "acabamos de ver que", "por tanto" usados con sentido, no como
  etiqueta en cada frase. **Banco completo por función en `tesis_assets/conectores_raquel.md`**, con tres reglas:
  variedad obligatoria (ningún conector dos veces seguidas en un párrafo), conexión implícita siempre que se
  pueda, densidad razonable.
- **Explica con propósito**: transmite comprensión ("α+β mide la persistencia de la volatilidad: cuán despacio
  se reabsorbe un golpe"), no enuncia mecánicamente.

## 3. Reglas de cita (anti-plagio)
- Toda afirmación **no original** lleva cita verificada (existe y dice lo que se le atribuye).
- Paráfrasis lejana, nunca cercana; **>40 palabras seguidas de una fuente → cita literal entre comillas**, no
  paráfrasis.
- Definiciones estándar (GARCH, HMM, McNemar…) se citan a su fuente.

## 4. Cifras
- Siempre desde JSON o tabla autogenerada, **nunca a mano**. Decimal con coma en el texto (0,552).
