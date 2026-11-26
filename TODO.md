# Guía para presentar Miotubos

wNov 20, 2026 · @Diego

## El proyecto en pocas frases

Entrenamos tres redes para marcar, píxel por píxel, dónde hay miotubos en imágenes de microscopía, y la U-Net con bloques residuales obtuvo el mayor Dice: 0.5887.

- **Qué entra y qué sale:** entra un recorte en escala de grises de 256 × 256 píxeles y sale una máscara del mismo tamaño, donde cada píxel dice "miotubo" o "fondo".
- **Con qué datos:** 243 pares de imagen y máscara de la carpeta `vt`, 194 para entrenar y 49 para validar.
- **Qué comparamos:** tres redes que solo se diferencian en la arquitectura. Todo lo demás es igual.
- **Qué resultó:** las tres quedan muy cerca, entre 0.573 y 0.589 de Dice.
- **Qué hay que decir con honestidad:** las anotaciones son parciales, la validación es provisional y cada red se entrenó una sola vez.

## Conceptos

Cada término que aparece en las diapositivas, en palabras simples.

### El problema y los datos

| Término | Qué significa |
| --- | --- |
| Miotubo | Célula muscular alargada que se ve en las imágenes de microscopía. |
| Segmentación | Decidir, para cada píxel de la imagen, si es miotubo o fondo. |
| Máscara | Imagen en blanco y negro del mismo tamaño que la original: 1 donde hay miotubo, 0 donde hay fondo. |
| Anotación | La máscara hecha a mano por una persona. Es contra lo que comparamos al modelo. |
| Anotación parcial | Solo se marcaron algunos miotubos. Lo que no está marcado no es necesariamente fondo. |
| Recorte | Un pedazo de 256 × 256 píxeles de la imagen completa, que mide 2592 × 1944. No se cambia la resolución. |
| Entrenamiento y validación | Con las imágenes de entrenamiento la red aprende. Con las de validación, que nunca vio, medimos qué tan bien lo hace. |

### Cómo está hecha la red

| Término | Qué significa |
| --- | --- |
| Convolución | Un filtro pequeño, aquí de 3 × 3 píxeles, que recorre la imagen buscando un patrón, como un borde. |
| Canal | Una "capa" de información: cada canal resalta un patrón distinto. La entrada tiene 1 canal, el brillo. |
| ReLU | Paso que convierte en cero los valores negativos. Permite a la red aprender formas complejas. |
| MaxPooling | Reduce la imagen a la mitad quedándose con el valor más alto de cada grupo de 2 × 2 píxeles. |
| Encoder | La primera mitad de la red: achica la imagen y sube los canales (16, 32 y 64) para entender qué hay. |
| Decoder | La segunda mitad: vuelve a agrandar hasta el tamaño original para dibujar la máscara. |
| Convolución transpuesta | La operación que usa el decoder para agrandar la imagen al doble. |
| Conexión de salto | Atajo que copia información del encoder al decoder del mismo tamaño, para recuperar los detalles que se perdieron al achicar. |
| Bloque residual | Bloque con un atajo interno: su entrada se suma a su salida. Ayuda a que la red entrene de forma más estable. |
| Sigmoid y umbral | Sigmoid convierte la salida en un valor entre 0 y 1. Si es 0.50 o más, el píxel se marca como miotubo. |

### Cómo se entrena

| Término | Qué significa |
| --- | --- |
| Pesos | Los números internos que la red ajusta al aprender. Empiezan al azar: eso es entrenar "desde cero". |
| Época | Una pasada completa por los datos de entrenamiento. Hicimos 10. |
| Lote | Cuántos recortes ve la red antes de ajustar sus pesos. Usamos 2. |
| Función de pérdida | El número que mide qué tan equivocada está la red. Entrenar es hacerlo bajar. |
| BCE ponderada | Parte de la pérdida que revisa píxel por píxel. "Ponderada" porque un miotubo no detectado cuenta 5 veces más, ya que hay menos píxeles de miotubo que de fondo. |
| Adam | El método que decide cómo ajustar los pesos en cada paso. |
| Tasa de aprendizaje | Qué tan grande es cada ajuste. Usamos 0.001. |

### Cómo se mide

| Término | Qué significa |
| --- | --- |
| Dice | Qué tanto coincide lo que marcó el modelo con la anotación. Va de 0 a 1, y 1 es coincidencia total. Es nuestra medida principal. |
| IoU | Mide lo mismo que Dice con otra fórmula, más estricta. Siempre da un número menor o igual. |
| Precisión | De todo lo que marcó el modelo, qué parte estaba anotada. |
| Sensibilidad | De todo lo anotado, qué parte encontró el modelo. |

## Qué decir en cada diapositiva

La idea principal de cada una y los datos que conviene tener a la mano.

| # | Diapositiva | Qué explicar |
| --- | --- | --- |
| 1 | Portada | El tema en una frase: enseñamos a una red a marcar miotubos en imágenes de microscopía. |
| 2 | Propósito | Qué es segmentar y por qué es difícil aquí: poco contraste, ruido y estructuras parecidas. El objetivo es comparar tres arquitecturas y elegir la mejor en validación. |
| 3 | Datos y preparación | De 306 pares usamos 243: 194 para entrenar y 49 para validar. Los 63 excluidos fueron 34 máscaras vacías, 28 etiquetas inválidas y 1 polígono mayor al 70 %. Trabajamos con recortes de 256 × 256 sin cambiar la resolución. Avisar desde aquí que las anotaciones son parciales. |
| 4 | Modelos | Las tres redes forman una escalera. El encoder–decoder no tiene conexiones de salto. La U-Net las agrega. La U-Net residual agrega además atajos dentro de cada bloque. Como todo lo demás es igual, cualquier diferencia viene de la arquitectura. |
| 5 | La U-Net | Seguir el camino de izquierda a derecha: entra el recorte, el encoder lo achica (16 y 32 canales), el centro tiene 64 canales a 64 × 64, el decoder lo agranda usando las conexiones de salto, y sale un valor por píxel. Luego el entrenamiento: 10 épocas, Adam, tasa de 0.001, pérdida de BCE ponderada más Dice. |
| 6 | Tabla de comparación | La U-Net residual tiene el mayor Dice, 0.589, pero las tres quedan muy cerca. Hay un intercambio: el encoder–decoder encuentra más de lo anotado (sensibilidad 0.814) pero marca más de la cuenta (precisión 0.442); la residual es la más precisa (0.491). |
| 7 | Dice por época | La residual sube de forma constante hasta la época 10. La U-Net pequeña llega a su mejor valor, 0.582, en la época 6 y ya no mejora. El encoder–decoder casi no avanza las primeras cuatro épocas. |
| 8 | Ejemplos | Tres recortes de validación: uno difícil, uno típico y uno fácil. Blanco es lo marcado como miotubo. Comparar cada modelo contra la columna de anotación. |
| 9 | Conclusiones | Elegimos la U-Net residual por tener el mayor Dice, 0.5887 en la época 10. La diferencia es pequeña y viene de un solo entrenamiento de cada red. Cerrar con las limitaciones: anotaciones parciales y validación provisional. |

Dos cosas para no confundir al hablar:

- En la diapositiva 7 la U-Net pequeña aparece con 0.579, su valor en la época 10. En la tabla aparece con 0.582, su mejor época. Los dos números son correctos.
- Los recortes de la diapositiva 8 se eligieron por el Dice promedio de las tres redes: uno de los más bajos, el de en medio y uno de los más altos. No son los mejores casos escogidos a mano.

## Preguntas que pueden hacer

Las más probables, con una respuesta corta y honesta.

| Pregunta | Respuesta |
| --- | --- |
| ¿Por qué el Dice es apenas 0.59? | Porque las anotaciones son parciales. Si la red marca un miotubo real que nadie anotó, se cuenta como error. Por eso el Dice tiene un tope que no depende solo del modelo. |
| ¿Por qué los modelos marcan más área que la anotada? | Marcan entre 30.7 % y 37.7 % y lo anotado es 20.5 %. Puede ser que encuentren miotubos sin anotar o que marquen de más. Con anotaciones parciales no podemos distinguir cuánto es cada cosa. |
| ¿La U-Net residual es realmente mejor? | Tiene el mayor Dice, pero por menos de 0.01. Con un solo entrenamiento y 49 recortes no podemos asegurarlo. Lo que sí se ve es que entrena de forma más estable. |
| ¿Por qué incluir el encoder–decoder? | Es la U-Net sin conexiones de salto. Compararlos muestra cuánto aportan esas conexiones. |
| ¿Por qué la validación es "provisional"? | Las imágenes se repartieron al azar. Si dos imágenes vienen de la misma muestra y caen una en cada grupo, el resultado puede salir mejor de lo real. Tampoco hay un tercer grupo de prueba aparte. |
| ¿Por qué se excluyeron 63 pares? | 34 tenían la máscara vacía, 28 tenían la etiqueta en un formato inválido y 1 tenía un polígono que cubría más del 70 % de la imagen. Todos quedaron registrados en un archivo. |
| ¿Por qué recortes y no la imagen completa? | Las imágenes miden 2592 × 1944 píxeles. Los recortes de 256 × 256 conservan el detalle original y caben en una computadora con poca memoria. |
| ¿Por qué el peso 5 en la pérdida? | Hay menos píxeles de miotubo que de fondo. Sin ese peso, a la red le conviene marcar casi todo como fondo. |
| ¿Por qué solo 10 épocas? | Fue una primera prueba pensada para una computadora modesta. La U-Net residual seguía mejorando en la época 10, así que probablemente le sirvan más. |
| ¿Qué sigue? | Entrenar más épocas, repetir cada entrenamiento varias veces para confirmar la diferencia, y repartir las imágenes por muestra biológica. |
