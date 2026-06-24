# Portada TFG — UCM Facultad de Matemáticas
 
Replica **exactamente** esta portada en LaTeX. El logo está en `logo_UCM.png`.
El ejemplo es `ejemplo_portada.png`.
 
---
 
## Estructura y espaciado
 
La portada es una página única sin número de página, con todo el contenido centrado horizontalmente. El orden vertical, de arriba a abajo, es:
 
1. `UNIVERSIDAD COMPLUTENSE DE MADRID` — en la parte superior, con margen top generoso (~3–4 cm desde el borde)
2. `FACULTAD DE MATEMÁTICAS` — inmediatamente debajo, sin espacio extra entre las dos líneas de institución
3. Espacio vertical grande (~2.5 cm)
4. Logo `logo_UCM.png` — centrado, ancho ~4.5 cm
5. Espacio vertical grande (~2.5 cm)
6. `TRABAJO DE FIN DE GRADO`
7. Espacio vertical muy grande (~3.5–4 cm) — este es el hueco más grande de la página
8. Título del trabajo (dos líneas, centrado)
9. Espacio pequeño (~0.8 cm)
10. `Supervisor: Ana Carpio`
11. Espacio vertical grande (~3 cm)
12. Nombre del autor en negrita y mayor tamaño
13. Espacio pequeño (~0.4 cm)
14. Grado
15. Espacio pequeño (~0.4 cm)
16. Curso académico
---
 
## Tipografía exacta
 
Usa `\usepackage{lmodern}` o Computer Modern (la fuente por defecto de LaTeX, con serifa). **No usar sans-serif.**
 
| Elemento | Comando LaTeX | Características |
|---|---|---|
| `UNIVERSIDAD COMPLUTENSE DE MADRID` | `\large\textbf` | Mayúsculas, negrita |
| `FACULTAD DE MATEMÁTICAS` | `\large\textbf` | Mayúsculas, negrita |
| `TRABAJO DE FIN DE GRADO` | `\large\textbf` | Mayúsculas, negrita |
| Título del trabajo | `\large` (sin negrita) | Dos líneas, centrado, fuente normal |
| `Supervisor: Ana Carpio` | `\normalsize` | Normal, centrado |
| Nombre del autor | `\Large\textbf` | Negrita, tamaño mayor que el resto |
| Grado | `\normalsize` | Normal |
| Curso académico | `\normalsize` | Normal |
 
---
 
## Código LaTeX
 
```latex
\begin{titlepage}
    \centering
    \vspace*{1cm}
 
    {\large\textbf{UNIVERSIDAD COMPLUTENSE DE MADRID}}\\[0.3cm]
    {\large\textbf{FACULTAD DE MATEMÁTICAS}}
 
    \vspace{2.5cm}
 
    \includegraphics[width=4.5cm]{logo_UCM}
 
    \vspace{2.5cm}
 
    {\large\textbf{TRABAJO DE FIN DE GRADO}}
 
    \vspace{4cm}
 
    {\large Desarrollo de aplicación basada en modelos de clasificación para diagnóstico\\
    de enfermedades cardiovasculares.}
 
    \vspace{0.8cm}
 
    {\normalsize Supervisor: Ana Carpio}
 
    \vspace{3cm}
 
    {\Large\textbf{Sergio Gil Gavela}}
 
    \vspace{0.4cm}
 
    {\normalsize Doble grado en Matemáticas e Ingeniería Informática}
 
    \vspace{0.4cm}
 
    {\normalsize Curso académico 2020-21}
 
\end{titlepage}
```
 
---
 
## Paquetes necesarios en el preámbulo
 
```latex
\usepackage{graphicx}   % para \includegraphics
\usepackage{lmodern}    % fuente con serifa, calidad PDF
\usepackage[T1]{fontenc}
```
 
---
 
## Notas para la adaptación
 
- Sustituye el **título**, **supervisor**, **autor**, **grado** y **curso** por los de tu TFG.
- El `\vspace*{1cm}` inicial usa asterisco para que funcione aunque sea la primera línea de la página.
- Si la página tiene márgenes distintos a los por defecto, ajusta los `\vspace` proporcionalmente — el hueco más grande es siempre el que queda entre `TRABAJO DE FIN DE GRADO` y el título.
- `logo_UCM` sin extensión: LaTeX busca el archivo `logo_UCM.png` automáticamente si usas `pdflatex`.