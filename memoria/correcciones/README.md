# memoria/correcciones/ — registro de las correcciones de Raquel por capítulo

Aquí va, por capítulo, el registro de **cada cambio que Raquel hace** sobre la prosa redactada, con su razón y
categoría. Es la **entrada** del agente `aprendiz-correcciones`, que destila de aquí las reglas generalizables
hacia `memoria/correcciones_aprendidas.md`.

## Cómo se genera (Método C, sin sintaxis para Raquel)
1. Existe un borrador commiteado (la "versión redactada").
2. Raquel edita el capítulo a su aire (Overleaf o local).
3. El hilo principal hace el **diff** redactada→corregida y extrae cada cambio.
4. Por cada cambio, propone razón + categoría; Raquel confirma o ajusta.
5. Se escribe `capN.md` con una entrada por cambio.

## Formato de cada entrada (`capN.md`)
```
### [cap3 · §3.2 RAM] estilo
- Original:  "<frase tal como la redacté>"
- Corregido: "<frase tal como la dejó Raquel>"
- Razón:     "<por qué lo cambió>"
- Categoría: estilo | rigor | cifras | énfasis | exclusión | estructura | citas | claridad
```

## Categorías
| Categoría | Qué captura |
|---|---|
| estilo | voz, longitud de frase, guiones, conectores, primera persona |
| claridad / estructura | orden, exposición, dependencias |
| rigor | imprecisión o afirmación demasiado fuerte |
| cifras | número mal o sin traza a JSON |
| **énfasis** | desarrollar más/menos algo; con qué liderar |
| **exclusión** | no mencionar X; no enmarcar como Y |
| citas | preferencia de citación |

Las **exclusiones** y el **énfasis** se vuelven regla de forma inmediata (son órdenes). El **estilo** se
refuerza con la repetición.
