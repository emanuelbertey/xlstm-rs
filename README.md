# xLSTM-RS: Extended Long Short-Term Memory in Rust

[Versión en Español abajo](#español)

xLSTM-RS is a high-performance implementation of the **xLSTM (Extended Long Short-Term Memory)** architecture, developed in Rust using the **Burn** deep learning framework. This project focuses on financial time series forecasting using pre-computed embeddings.

## Key Features

*   **xLSTM Architecture**: Implements both **sLSTM** (scalar LSTM with exponential gating) and **mLSTM** (matrix LSTM with covariance memory).
*   **Flexible Blocks**: Support for stacking blocks in sLSTM, mLSTM, or alternating patterns to capture different sequence dependencies.
*   **Burn Framework**: Leverages Rust's safety and performance with multiple backend support (NdArray, WGPU/Vulkan).
*   **Memory Optimized**: Efficient handling of the computation graph during validation and inference to prevent memory leaks.

## Visualization Results

### Predictions vs Actual
Comparison of the predicted relative price changes against the real market movements.
![Predictions vs Actual](predictions_vs_actual.png)

### Accuracy Scatter Plot
Distribution of predictions relative to the ideal line (y=x).
![Prediction Scatter Plot](prediction_scatter.png)

## Getting Started

1.  **Train a model**: `cargo run --release -- train`
2.  **Run inference**: `cargo run --release -- infer xlstm_model`
3.  **Continue training**: `cargo run --release -- continue xlstm_model`

---

<a name="español"></a>
# xLSTM-RS: Extended Long Short-Term Memory en Rust

xLSTM-RS es una implementación de alto rendimiento de la arquitectura **xLSTM (Extended Long Short-Term Memory)**, desarrollada en Rust utilizando el framework de deep learning **Burn**. Este proyecto se centra en la predicción de series temporales financieras utilizando embeddings precalculados.

## Características Principales

*   **Arquitectura xLSTM**: Implementa tanto **sLSTM** (LSTM escalar con compuertas exponenciales) como **mLSTM** (LSTM matricial con memoria de covarianza).
*   **Bloques Flexibles**: Soporte para apilar bloques en patrones sLSTM, mLSTM o alternados para capturar diferentes dependencias de secuencias.
*   **Framework Burn**: Aprovecha la seguridad y el rendimiento de Rust con soporte para múltiples backends (NdArray, WGPU/Vulkan).
*   **Memoria Optimizada**: Gestión eficiente del grafo de computación durante la validación e inferencia para evitar fugas de memoria.

## Resultados Visuales

### Predicciones vs Actual
Comparativa de los cambios de precio relativos predichos frente a los movimientos reales del mercado.
![Predicciones vs Real](predictions_vs_actual.png)

### Gráfico de Dispersión (Accuracy)
Distribución de las predicciones respecto a la línea ideal (y=x).
![Gráfico de Dispersión](prediction_scatter.png)

## Cómo empezar

1.  **Entrenar un modelo**: `cargo run --release -- train`
2.  **Ejecutar inferencia**: `cargo run --release -- infer xlstm_model`
3.  **Continuar entrenamiento**: `cargo run --release -- continue xlstm_model`

## License / Licencia

**English**: The improvements and modifications made in this fork are dual-licensed under the **MIT License** and **Apache License 2.0**, following the Rust ecosystem standards. We encourage the original creator to define a global license for the project.

**Español**: Las mejoras y modificaciones realizadas en este fork tienen una licencia doble **MIT** y **Apache 2.0**, siguiendo los estándares del ecosistema de Rust. Se anima al creador original a definir una licencia global para el proyecto.

---

Official Repository: [https://github.com/thmasq/xlstm-rs](https://github.com/thmasq/xlstm-rs)

---

### Personal Project / Proyecto Personal: **Laurelia**
For a more advanced implementation focused on LLMs and Chatbots using the **Hugging Face Candle** framework, check out:
Para una implementación más avanzada enfocada en LLMs y Chatbots usando el framework **Candle** de Hugging Face, visita:

**[Laurelia (emanuelbertey/LaurelIA)](https://github.com/emanuelbertey/LaurelIA.git)**
*Uses advanced optimization techniques such as fused projections, matrix memory normalization, and parallel kernels.*
*Utiliza técnicas avanzadas de optimización como proyecciones fusionadas, normalización de memoria matricial y kernels paralelos.*
