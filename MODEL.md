# Modelo de predicción de nivel de biomasa

## Modelo de crecimiento de materia seca

Desde el punto de vista predictivo, el objetivo del modelo entrenado para estimación de crecimiento de la biomasa puede ser representado como sigue:

`b = f(b0 , t , r , i, h)`

Donde:

- b es el nivel de biomasa estimado al final del periodo de predicción en kg/Ha
- b0 es el nivel de biomasa inicial (medido mediante un platímetro) en kg/Ha
- t es la temperatura cumulativa durante el periodo de tiempo analizado en grados centígrados.
- r es el nivel de precipitación cumulativo durante el periodo de tiempo analizado en milímetros.
- i es el nivel de iluminación cumulativo durante el periodo de tiempo en lux.
- h es el nivel de humedad relativa del aire cumulativo durante el periodo de tiempo en %.

Si tomamos b como estimación de nivel de biomasa, y lo multiplicamos por el número de hectáreas del total de los potreros, tenemos una estimación de biomasa disponible en el módulo productivo.

## Modelo de consumo de materia seca

El modelo del National Research Council (NRC) 2016 proporciona un método integral para calcular los requerimientos nutricionales del ganado de carne. Se describe el cálculo del consumo de materia seca (CMS) utilizando la fórmula del NRC 2016, considerando los datos de peso vivo (PV), ganancia media diaria (GMD) y calidad del forraje (K_CNE).

La fórmula para calcular el consumo de materia seca (CMS) en kg/día es la siguiente:

`CMS (kg/día) = (0.025 × PV + 0.1 × GMD) × K_CNE`

Donde:

- CMS: Consumo de materia seca en kg/día
- PV: Peso vivo del animal en kg
- GMD: Ganancia media diaria en kg/día (se estima en 0.3 kg)
- K_CNE: Factor de ajuste para la concentración de energía neta del forraje (1 asumiendo pastura de calidad media)

## Predicción de nivel de materia seca

Para la predicción final de materia seca (teniendo en cuenta el crecimiento y el consumo de la misma), se precisan los siguientes datos adicionales:

- Tamaño del potrero utilizado (Ha)
- Porcentaje de pastura del potrero (para descontar la maleza)
- Número de cabezas de ganado para el pastoreo
- Peso promedio por cabeza de ganado (kg)
