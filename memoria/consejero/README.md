# Registro de sesiones del consejero

Aquí se guarda **todo lo que hablas con el consejero-revisor**, una sesión por fichero, para que puedas
comentarlo y pedir que los demás agentes de la tesis (rigor, estilo, citas, tutor…) te ayuden a partir de ahí.

## Cómo se llena

- **Automático (hook).** Al terminar cada respuesta del consejero, un hook del sistema añade el intercambio
  (tu mensaje + la respuesta) al fichero de la sesión. Captura garantizada, aunque el modelo se despiste.
  Script: `.claude/hooks/consejero_log.py`; se dispara con el evento `Stop` (ver `.claude/settings.json`).
- **Hitos (consejero).** Además, el propio consejero anota los **hitos** de la sesión —decisiones tomadas,
  anotaciones `% >>> CONSEJERO` insertadas o aplicadas, subagentes llamados— en una sección `### Hitos` del
  mismo fichero, para tener una vista resumida sin releer todo.

Cada sesión es `sesiones/consejero_HHMM_DDMM.md` (hora y día de inicio, tu convención del histórico). El
`INDICE.md` las lista.

## Cómo comentas (y pides ayuda a otros agentes)

Edita el fichero de la sesión y añade líneas con estas marcas donde quieras:

```
> COMENTARIO: <tu nota libre sobre lo que dijo el consejero>
> AYUDA: <algo que quieres revisar; deja que el consejero elija el especialista>
> AYUDA[@rigor-matematico]: <pídeselo a un agente concreto>
> DUDA: <algo que no entiendes y quieres que te expliquen>
```

Después, en un chat (con el consejero encendido) di **«revisa mis comentarios de la última sesión»** o usa el
comando `/consejero-sesion`: lee tus marcas y enruta cada una al agente que toca (`rigor-matematico`,
`estilo-raquel`, `experto-citas`, `defensa-tutor`, `oraculo-tesis`…), trayéndote el dictamen.

## Privacidad

- No se registra el razonamiento interno del modelo (`thinking`), ni las llamadas a herramientas, solo la
  conversación legible.
- El estado interno del hook vive en `.state.json` (ignorado por git).
- Los `.md` de sesión **sí** se versionan: son parte de tu cuaderno de trabajo y defendibles ante el tribunal.
